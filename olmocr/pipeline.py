import argparse
import asyncio
import atexit
import base64
from collections import deque
import datetime
import hashlib
import json
import logging
import multiprocessing
import os
import random
import re
import shutil
import ssl
import sys
import tarfile
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cache
from io import BytesIO
from urllib.parse import urlparse

import boto3
import httpx
from botocore.exceptions import ClientError
from huggingface_hub import snapshot_download
from PIL import Image, ImageFilter
from pypdf import PdfReader
from tqdm import tqdm

from olmocr.check import (
    check_poppler_version,
    check_torch_gpu_available,
)
from olmocr.data.renderpdf import get_png_dimensions_from_base64, render_pdf_to_base64png
from olmocr.filter.filter import Language, PdfFilter
from olmocr.image_utils import convert_image_to_pdf_bytes, is_jpeg, is_png
from olmocr.metrics import MetricsKeeper, WorkerTracker
from olmocr.prompts import PageResponse, build_no_anchoring_v4_yaml_prompt
from olmocr.prompts.anchor import PageReport, _merge_image_elements, _pdf_report, get_anchor_text
from olmocr.s3_utils import (
    download_directory,
    download_zstd_csv,
    expand_s3_glob,
    get_s3_bytes,
    get_s3_bytes_with_backoff,
    parse_s3_path,
)
from olmocr.train.front_matter import FrontMatterParser
from olmocr.version import VERSION
from olmocr.work_queue import LocalBackend, S3Backend, WorkQueue

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

server_logger = logging.getLogger("vllm")
server_logger.propagate = False

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

# Add console handler to loggers (file handler added later if disk logging enabled)
logger.addHandler(console_handler)
server_logger.addHandler(console_handler)

# Quiet logs from pypdf
logging.getLogger("pypdf").setLevel(logging.ERROR)

# Global s3 clients fo the whole script, we have two separate ones in case your workspace and your pdfs are in different accounts
workspace_s3 = boto3.client("s3")
pdf_s3 = boto3.client("s3")

# Global variables for token statistics
metrics = MetricsKeeper(window=60 * 5)
tracker = WorkerTracker()

# Global variable for vLLM queue status (updated by vllm_server_task)
vllm_queued_requests = None

# Temperature values for retry attempts - higher temperature helps overcome repetition issues
TEMPERATURE_BY_ATTEMPT = [0.1, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 1.0]

pdf_render_max_workers_limit = asyncio.BoundedSemaphore(int(float(os.environ.get("BEAKER_ASSIGNED_CPU_COUNT", max(1, multiprocessing.cpu_count() - 2)))))
max_concurrent_requests_limit = asyncio.BoundedSemaphore(1)  # Actual value set by args in main()

# Filter object, cached so it will only get loaded when/if you need it
get_pdf_filter = cache(lambda: PdfFilter(languages_to_keep={Language.ENGLISH, None}, apply_download_spam_check=True, apply_form_check=True))


@dataclass(frozen=True)
class PageResult:
    s3_path: str
    page_num: int
    response: PageResponse

    input_tokens: int
    output_tokens: int
    is_fallback: bool
    is_valid: bool


async def build_page_query(local_pdf_path: str, page: int, target_longest_image_dim: int, image_rotation: int = 0, model_name: str = "olmocr") -> dict:
    MAX_TOKENS = 8000
    assert image_rotation in [0, 90, 180, 270], "Invalid image rotation provided in build_page_query"

    # Allow the page rendering to process in the background, but limit the number of workers otherwise you can overload the system
    async with pdf_render_max_workers_limit:
        image_base64 = await asyncio.to_thread(render_pdf_to_base64png, local_pdf_path, page, target_longest_image_dim=target_longest_image_dim)

    if image_rotation != 0:
        image_bytes = base64.b64decode(image_base64)
        with Image.open(BytesIO(image_bytes)) as img:
            if image_rotation == 90:
                tranpose = Image.Transpose.ROTATE_90
            elif image_rotation == 180:
                tranpose = Image.Transpose.ROTATE_180
            else:
                tranpose = Image.Transpose.ROTATE_270

            rotated_img = img.transpose(tranpose)

            # Save the rotated image to a bytes buffer
            buffered = BytesIO()
            rotated_img.save(buffered, format="PNG")

        # Encode the rotated image back to base64
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_no_anchoring_v4_yaml_prompt(*get_png_dimensions_from_base64(image_base64))},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,  # This will get overridden later
    }


async def try_single_page(
    args,
    pdf_orig_path: str,
    pdf_local_path: str,
    page_num: int,
    attempt: int,
    rotation: int,
) -> PageResult | None:
    """
    Try processing a single page once. Returns PageResult on success, None on failure.
    Does NOT handle retries - caller is responsible for retry logic.
    """
    COMPLETION_URL = f"{args.server.rstrip('/')}/chat/completions"
    MODEL_MAX_CONTEXT = 16384

    temp_idx = min(attempt, len(TEMPERATURE_BY_ATTEMPT) - 1)
    temperature = TEMPERATURE_BY_ATTEMPT[temp_idx]

    api_key = args.api_key if args.server and hasattr(args, "api_key") else None

    try:
        query = await build_page_query(
            pdf_local_path,
            page_num,
            args.target_longest_image_dim,
            image_rotation=rotation,
            model_name=args.model,
        )
        query["temperature"] = temperature

        if args.guided_decoding:
            query["guided_regex"] = (
                r"---\nprimary_language: (?:[a-z]{2}|null)\nis_rotation_valid: (?:True|False|true|false)\nrotation_correction: (?:0|90|180|270)\nis_table: (?:True|False|true|false)\nis_diagram: (?:True|False|true|false)\n(?:---|---\n[\s\S]+)"
            )

        async with max_concurrent_requests_limit:
            status_code, response_body = await apost(COMPLETION_URL, json_data=query, api_key=api_key)

        if status_code != 200:
            logger.warning(
                f"Server returned {status_code} for {pdf_orig_path}-{page_num} attempt {attempt}: {response_body[:500] if response_body else 'empty response'}"
            )
            return None

        base_response_data = json.loads(response_body)

        metrics.add_metrics(
            server_input_tokens=base_response_data["usage"].get("prompt_tokens", 0),
            server_output_tokens=base_response_data["usage"].get("completion_tokens", 0),
        )

        is_valid = True

        if base_response_data["usage"]["total_tokens"] > MODEL_MAX_CONTEXT:
            is_valid = False

        if base_response_data["choices"][0]["finish_reason"] != "stop":
            is_valid = False

        model_response_markdown = base_response_data["choices"][0]["message"]["content"]

        parser = FrontMatterParser(front_matter_class=PageResponse)
        front_matter, text = parser._extract_front_matter_and_text(model_response_markdown)
        page_response = parser._parse_front_matter(front_matter, text)

        return PageResult(
            pdf_orig_path,
            page_num,
            page_response,
            input_tokens=base_response_data["usage"].get("prompt_tokens", 0),
            output_tokens=base_response_data["usage"].get("completion_tokens", 0),
            is_fallback=False,
            is_valid=is_valid,
        )
    except asyncio.CancelledError:
        raise
    except (ConnectionError, OSError, asyncio.TimeoutError):
        # Re-raise connection errors so caller can apply exponential backoff
        raise
    except Exception as e:
        logger.warning(f"try_single_page failed for {pdf_orig_path}-{page_num} attempt {attempt}: {type(e).__name__}: {e}")
        return None


def make_fallback_result(pdf_orig_path: str, pdf_local_path: str, page_num: int) -> PageResult:
    """Create a fallback PageResult using pdftotext."""
    return PageResult(
        pdf_orig_path,
        page_num,
        PageResponse(
            natural_text=get_anchor_text(pdf_local_path, page_num, pdf_engine="pdftotext"),
            primary_language=None,
            is_rotation_valid=True,
            rotation_correction=0,
            is_table=False,
            is_diagram=False,
        ),
        input_tokens=0,
        output_tokens=0,
        is_fallback=True,
        is_valid=True,
    )


async def try_single_page_with_backoff(
    args,
    pdf_orig_path: str,
    pdf_local_path: str,
    page_num: int,
    attempt: int,
    rotation: int,
) -> PageResult | None:
    """
    Wrapper around try_single_page that handles connection errors with exponential backoff.
    """
    MAX_BACKOFF_ATTEMPTS = 10

    for backoff_count in range(MAX_BACKOFF_ATTEMPTS):
        try:
            return await try_single_page(args, pdf_orig_path, pdf_local_path, page_num, attempt, rotation)
        except (ConnectionError, OSError, asyncio.TimeoutError) as e:
            sleep_delay = 10 * (2**backoff_count)
            logger.warning(
                f"Connection error on {pdf_orig_path}-{page_num} attempt {attempt}: {type(e).__name__}: {e}. "
                f"Backoff {backoff_count + 1}/{MAX_BACKOFF_ATTEMPTS}, sleeping {sleep_delay}s"
            )
            await asyncio.sleep(sleep_delay)

    logger.error(f"Max backoff attempts reached for {pdf_orig_path}-{page_num}, terminating job")
    sys.exit(1)


async def process_page(args, worker_id: int, pdf_orig_path: str, pdf_local_path: str, page_num: int) -> PageResult:
    """
    Process a single page with retry logic:
    1. Try first attempt
    2. If success: return result
    3. If rotation error: retry sequentially (need model feedback for rotation correction)
    4. If other error: fire all remaining retries in parallel (if queue empty) or sequential
    """
    MAX_RETRIES = args.max_page_retries
    retry_attempts = list(range(1, MAX_RETRIES))
    cumulative_rotation = 0

    await tracker.track_work(worker_id, f"{pdf_orig_path}-{page_num}", "started")

    # === First attempt ===
    result = await try_single_page_with_backoff(args, pdf_orig_path, pdf_local_path, page_num, attempt=0, rotation=cumulative_rotation)

    if result is not None and not result.response.is_rotation_valid:
        cumulative_rotation = result.response.rotation_correction % 360

    # Success on first try
    if result is not None and result.is_valid and result.response.is_rotation_valid:
        metrics.add_metrics(**{"completed_pages": 1, "finished_on_attempt_0": 1})
        await tracker.track_work(worker_id, f"{pdf_orig_path}-{page_num}", "finished")
        return result

    # === Rotation error path: sequential retries with model feedback ===
    if result is not None and not result.response.is_rotation_valid:
        logger.info(f"Rotation error for {pdf_orig_path}-{page_num}, retrying sequentially with rotation={cumulative_rotation}")

        for attempt in retry_attempts:
            result = await try_single_page_with_backoff(args, pdf_orig_path, pdf_local_path, page_num, attempt, cumulative_rotation)

            if result is not None and result.is_valid and result.response.is_rotation_valid:
                metrics.add_metrics(**{"completed_pages": 1, f"finished_on_attempt_{attempt}": 1})
                await tracker.track_work(worker_id, f"{pdf_orig_path}-{page_num}", "finished")
                return result

            if result is not None:  # Another rotation correction needed
                cumulative_rotation = (cumulative_rotation + result.response.rotation_correction) % 360

        # If you tried many times and all rotations were invalid, but you at least had a valid response, then return that in the end
        if result is not None and result.is_valid:
            metrics.add_metrics(**{"completed_pages": 1, f"finished_on_attempt_{MAX_RETRIES}": 1})
            await tracker.track_work(worker_id, f"{pdf_orig_path}-{page_num}", "finished")
            return result

        # Otherwise you can do a full fallback
        logger.error(f"Failed {pdf_orig_path}-{page_num} after {MAX_RETRIES} rotation retries")
        metrics.add_metrics(failed_pages=1)
        await tracker.track_work(worker_id, f"{pdf_orig_path}-{page_num}", "errored")
        return make_fallback_result(pdf_orig_path, pdf_local_path, page_num)

    # === Non-rotation error path: sequential, but switch to parallel if queue empties ===
    for i, attempt in enumerate(retry_attempts):
        result = await try_single_page_with_backoff(args, pdf_orig_path, pdf_local_path, page_num, attempt, rotation=cumulative_rotation)

        if result is not None and result.is_valid and result.response.is_rotation_valid:
            metrics.add_metrics(**{"completed_pages": 1, f"finished_on_attempt_{attempt}": 1})
            await tracker.track_work(worker_id, f"{pdf_orig_path}-{page_num}", "finished")
            return result

        # After each failed attempt, check if queue is empty - if so, fire remaining in parallel
        remaining_attempts = retry_attempts[i + 1 :]
        if remaining_attempts and vllm_queued_requests == 0:
            logger.info(f"Queue empty, firing {len(remaining_attempts)} parallel retries for {pdf_orig_path}-{page_num}")
            tasks = [
                asyncio.create_task(try_single_page_with_backoff(args, pdf_orig_path, pdf_local_path, page_num, a, rotation=cumulative_rotation))
                for a in remaining_attempts
            ]

            for coro in asyncio.as_completed(tasks):
                try:
                    result = await coro
                    if result is not None and result.is_valid and result.response.is_rotation_valid:
                        for t in tasks:
                            t.cancel()
                        metrics.add_metrics(**{"completed_pages": 1, "finished_on_parallel_retry": 1})
                        await tracker.track_work(worker_id, f"{pdf_orig_path}-{page_num}", "finished")
                        return result
                except asyncio.CancelledError:
                    continue
            break  # Parallel attempts exhausted

    # If you tried many times and a least had a valid response, then return that in the end
    if result is not None and result.is_valid:
        metrics.add_metrics(**{"completed_pages": 1, f"finished_on_attempt_{MAX_RETRIES}": 1})
        await tracker.track_work(worker_id, f"{pdf_orig_path}-{page_num}", "finished")
        return result

    # All retries exhausted
    logger.error(f"Failed {pdf_orig_path}-{page_num} after {MAX_RETRIES} attempts")
    metrics.add_metrics(failed_pages=1)
    await tracker.track_work(worker_id, f"{pdf_orig_path}-{page_num}", "errored")
    return make_fallback_result(pdf_orig_path, pdf_local_path, page_num)


# Manual simple implementation of HTTP Post
# It feels strange perhaps, but httpx and aiohttp are very complex beasts
# Ex. the sessionpool in httpcore has 4 different locks in it, and I've noticed
# that at the scale of 100M+ requests, that they deadlock in different strange ways
async def apost(url, json_data, api_key=None):
    parsed_url = urlparse(url)
    host = parsed_url.hostname
    # Default to 443 for HTTPS, 80 for HTTP
    if parsed_url.scheme == "https":
        port = parsed_url.port or 443
        use_ssl = True
    else:
        port = parsed_url.port or 80
        use_ssl = False
    path = parsed_url.path or "/"

    writer = None
    try:
        if use_ssl:
            ssl_context = ssl.create_default_context()
            reader, writer = await asyncio.open_connection(host, port, ssl=ssl_context)
        else:
            reader, writer = await asyncio.open_connection(host, port)

        json_payload = json.dumps(json_data)

        headers = [
            f"POST {path} HTTP/1.1",
            f"Host: {host}",
            f"Content-Type: application/json",
            f"Content-Length: {len(json_payload)}",
        ]

        if api_key:
            headers.append(f"Authorization: Bearer {api_key}")

        headers.append("Connection: close")

        request = "\r\n".join(headers) + "\r\n\r\n" + json_payload
        writer.write(request.encode())
        await writer.drain()

        status_line = await reader.readline()
        if not status_line:
            raise ConnectionError("No response from server")
        status_parts = status_line.decode().strip().split(" ", 2)
        if len(status_parts) < 2:
            raise ValueError(f"Malformed status line: {status_line.decode().strip()}")
        status_code = int(status_parts[1])

        # Read headers
        headers = {}
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            key, _, value = line.decode().partition(":")
            headers[key.strip().lower()] = value.strip()

        # Read response body
        if "content-length" in headers:
            body_length = int(headers["content-length"])
            response_body = await reader.readexactly(body_length)
        elif headers.get("transfer-encoding", "") == "chunked":
            chunks = []
            while True:
                # Read chunk size line
                size_line = await reader.readline()
                chunk_size = int(size_line.strip(), 16)  # Hex format

                if chunk_size == 0:
                    await reader.readline()  # Read final CRLF
                    break

                chunk_data = await reader.readexactly(chunk_size)
                chunks.append(chunk_data)

                # Read trailing CRLF after chunk data
                await reader.readline()

            response_body = b"".join(chunks)
        elif headers.get("connection", "") == "close":
            # Read until connection closes
            response_body = await reader.read()
        else:
            raise ConnectionError("Cannot determine response body length")

        return status_code, response_body
    except Exception as e:
        # Pass through errors
        raise e
    finally:
        # But just make sure to close the socket on your way out
        if writer is not None:
            try:
                writer.close()
                await writer.wait_closed()
            except:
                pass


def is_tarball_path(path: str) -> bool:
    """Check if a path is a tarball based on extension."""
    lower = path.lower()
    return lower.endswith(".tar.gz") or lower.endswith(".tgz")


async def process_tarball(args, worker_id: int, tarball_path: str) -> list:
    """Process all PDFs inside a tarball concurrently and return list of Dolma documents."""
    logger.info(f"Worker {worker_id} processing tarball {tarball_path}")

    tarball_bytes = await asyncio.to_thread(lambda: get_s3_bytes_with_backoff(pdf_s3, tarball_path))

    # Extract all PDFs to a temp directory
    temp_dir = tempfile.mkdtemp()
    try:
        pdf_files = []  # (source_path, local_path)
        with tarfile.open(fileobj=BytesIO(tarball_bytes), mode="r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.lower().endswith(".pdf"):
                    local_path = os.path.join(temp_dir, os.path.basename(member.name))
                    with open(local_path, "wb") as f:
                        extracted = tar.extractfile(member)
                        if extracted:
                            f.write(extracted.read())
                            pdf_files.append((f"{tarball_path}::{member.name}", local_path))

        logger.info(f"Worker {worker_id} extracted {len(pdf_files)} PDFs from {tarball_path}")

        # Process all PDFs concurrently
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(process_single_pdf(args, worker_id, src, local)) for src, local in pdf_files]

        dolma_docs = [t.result() for t in tasks if t.result() is not None]
        logger.info(f"Worker {worker_id} processed {len(dolma_docs)} PDFs from tarball {tarball_path}")
        return dolma_docs
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


async def process_single_pdf(args, worker_id: int, pdf_orig_path: str, local_pdf_path: str):
    """Process a single PDF that's already on disk.

    Args:
        args: Pipeline arguments
        worker_id: Worker ID for logging
        pdf_orig_path: Original path (for metadata, can be tarball::internal format)
        local_pdf_path: Local path to the PDF file

    Returns:
        Dolma document or None
    """
    try:
        try:
            reader = PdfReader(local_pdf_path)
            num_pages = reader.get_num_pages()
        except:
            logger.exception(f"Could not count number of pages for {pdf_orig_path}, aborting document")
            return None

        logger.debug(f"Got {num_pages} pages to do for {pdf_orig_path} in worker {worker_id}")

        if args.apply_filter and get_pdf_filter().filter_out_pdf(local_pdf_path):
            logger.info(f"Filtering out pdf {pdf_orig_path}")
            return None

        # List to hold the tasks for processing each page
        page_tasks = []
        page_results = []

        async with asyncio.TaskGroup() as tg:
            for page_num in range(1, num_pages + 1):
                task = tg.create_task(process_page(args, worker_id, pdf_orig_path, local_pdf_path, page_num))
                page_tasks.append(task)

        # Collect the results from the entire task group, assuming no exceptions, if there is an exception propagated to this point in any page, it will abort the PDF itself
        page_results = [task.result() for task in page_tasks]
        assert all(page_result.is_valid for page_result in page_results)

        num_fallback_pages = sum(page_result.is_fallback for page_result in page_results)

        if num_fallback_pages / num_pages > args.max_page_error_rate:
            logger.error(
                f"Document {pdf_orig_path} has {num_fallback_pages} fallback pages out of {num_pages} exceeding max_page_error_rate of {args.max_page_error_rate}, discarding document."
            )
            return None
        elif num_fallback_pages > 0:
            logger.warning(
                f"Document {pdf_orig_path} processed with {num_fallback_pages} fallback pages out of {num_pages}, proceeding to build Dolma document."
            )

        return build_dolma_document(pdf_orig_path, page_results)
    except Exception as e:
        logger.exception(f"Exception in process_single_pdf for {pdf_orig_path}: {e}")
        return None


async def process_pdf(args, worker_id: int, pdf_orig_path: str):
    """Process a single PDF from S3/local path and return a Dolma document."""
    with tempfile.NamedTemporaryFile("wb+", suffix=".pdf", delete=False) as tf:
        try:
            data = await asyncio.to_thread(lambda: get_s3_bytes_with_backoff(pdf_s3, pdf_orig_path))
            tf.write(data)
            tf.flush()
        except ClientError as ex:
            if ex.response["Error"]["Code"] == "NoSuchKey":
                logger.info(f"S3 File Not found, skipping it completely {pdf_orig_path}")
                return None
            else:
                raise

        if is_png(tf.name) or is_jpeg(tf.name):
            logger.info(f"Converting {pdf_orig_path} from image to PDF format...")
            tf.seek(0)
            tf.write(convert_image_to_pdf_bytes(tf.name))
            tf.flush()

    try:
        return await process_single_pdf(args, worker_id, pdf_orig_path, tf.name)
    finally:
        if os.path.exists(tf.name):
            os.unlink(tf.name)


def build_dolma_document(pdf_orig_path, page_results):
    # Build the document text and page spans
    document_text = ""
    pdf_page_spans = []
    current_char_pos = 0

    for index, page_result in enumerate(page_results):
        if page_result.response.natural_text is not None:
            content = page_result.response.natural_text + ("\n\n" if index < len(page_results) - 1 else "")
        else:
            content = ""

        start_pos = current_char_pos
        document_text += content
        current_char_pos = len(document_text)
        pdf_page_spans.append([start_pos, current_char_pos, page_result.page_num])

    if not document_text:
        logger.info(f"No document text for {pdf_orig_path}")
        return None  # Return None if the document text is empty

    # Build the Dolma document
    metadata = {
        "Source-File": pdf_orig_path,
        "olmocr-version": VERSION,
        "pdf-total-pages": len(page_results),
        "total-input-tokens": sum(page.input_tokens for page in page_results),
        "total-output-tokens": sum(page.output_tokens for page in page_results),
        "total-fallback-pages": sum(page.is_fallback for page in page_results),
    }

    id_ = hashlib.sha1(document_text.encode()).hexdigest()

    dolma_doc = {
        "id": id_,
        "text": document_text,
        "source": "olmocr",
        "added": datetime.datetime.now().strftime("%Y-%m-%d"),
        "created": datetime.datetime.now().strftime("%Y-%m-%d"),
        "metadata": metadata,
        "attributes": {
            "pdf_page_numbers": pdf_page_spans,
            "primary_language": [p.response.primary_language for p in page_results],
            "is_rotation_valid": [p.response.is_rotation_valid for p in page_results],
            "rotation_correction": [p.response.rotation_correction for p in page_results],
            "is_table": [p.response.is_table for p in page_results],
            "is_diagram": [p.response.is_diagram for p in page_results],
        },
    }
    return dolma_doc


_IMAGE_REF_RE = re.compile(r"!\[[^\]]*\]\((page_(?:\d+_)?\d+_\d+_\d+_\d+\.png)\)")
_MARKDOWN_IMAGE_TAG_RE = re.compile(r"!\[([^\]]*)\]\((page_(?:\d+_)?\d+_\d+_\d+_\d+\.png)\)")
_FIGURE_CAPTION_RE = re.compile(r"(?m)^\s*Figure\s+(\d+(?:\.\d+)?)\b", re.IGNORECASE)


@dataclass(frozen=True)
class LayoutDetection:
    label: str
    score: float
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class DetectedFigureRef:
    page_num: int
    box: tuple[int, int, int, int]
    filename: str
    discovery_source: str


def _resolve_layout_device(device: str | None, torch_module) -> str:
    requested = (device or "auto").strip()
    normalized = requested.lower()

    if normalized in ("auto", "gpu"):
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if normalized.startswith("cuda") and not torch_module.cuda.is_available():
        logger.warning(f"Figure layout detector requested device '{requested}' but CUDA is unavailable, falling back to cpu")
        return "cpu"
    return requested


def _load_layout_detector_model(model_loader, model_name: str):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*copying from a non-meta parameter in the checkpoint to a meta parameter in the current model.*",
            category=UserWarning,
            module=r"torch\.nn\.modules\.module",
        )
        return model_loader.from_pretrained(model_name)


class FigureLayoutDetector:
    FIGURE_LABEL_TOKENS = ("picture", "figure", "chart", "diagram", "graphic", "image")

    def __init__(self, model_name: str, device: str, score_threshold: float):
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        device = _resolve_layout_device(device, torch)

        self.device = torch.device(device)
        self.score_threshold = score_threshold
        self.torch = torch
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = _load_layout_detector_model(AutoModelForObjectDetection, model_name)
        self.model.to(self.device)
        self.model.eval()

    def detect(self, image: "Image.Image") -> list[LayoutDetection]:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with self.torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = self.torch.tensor([image.size[::-1]], device=self.device)
        results = self.processor.post_process_object_detection(outputs, threshold=self.score_threshold, target_sizes=target_sizes)[0]

        detections: list[LayoutDetection] = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            label_name = str(self.model.config.id2label.get(int(label), int(label))).lower()
            if not any(token in label_name for token in self.FIGURE_LABEL_TOKENS):
                continue

            x0, y0, x1, y1 = [int(round(v)) for v in box.tolist()]
            detections.append(LayoutDetection(label=label_name, score=float(score), box=(x0, y0, x1, y1)))

        return detections


@cache
def get_figure_layout_detector(model_name: str | None, device: str, score_threshold: float) -> FigureLayoutDetector | None:
    if not model_name or model_name.lower() == "none":
        return None

    try:
        import torch

        resolved_device = _resolve_layout_device(device, torch)
        return FigureLayoutDetector(model_name, resolved_device, score_threshold)
    except Exception as exc:
        logger.warning(f"Could not initialize figure layout detector '{model_name}', falling back to heuristic crop refinement: {exc}")
        return None


def _clamp_box(box: tuple[int, int, int, int], iw: int, ih: int) -> tuple[int, int, int, int]:
    return (
        max(0, min(box[0], iw)),
        max(0, min(box[1], ih)),
        max(0, min(box[2], iw)),
        max(0, min(box[3], ih)),
    )


def _box_area(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _intersection_area(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> int:
    return max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])) * max(0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))


def _box_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    intersection = _intersection_area(box_a, box_b)
    union = _box_area(box_a) + _box_area(box_b) - intersection
    return intersection / max(union, 1)


def _expand_box(box: tuple[int, int, int, int], margin_x: int, margin_y: int, iw: int, ih: int) -> tuple[int, int, int, int]:
    return _clamp_box((box[0] - margin_x, box[1] - margin_y, box[2] + margin_x, box[3] + margin_y), iw, ih)


def _is_page_sized_box(box: tuple[int, int, int, int], iw: int, ih: int, min_fraction: float = 0.85) -> bool:
    return _box_area(box) / max(iw * ih, 1) >= min_fraction


def _box_to_ref_filename(page_num: int, box: tuple[int, int, int, int]) -> str:
    x0, y0, x1, y1 = box
    return f"page_{page_num}_{x0}_{y0}_{max(0, x1 - x0)}_{max(0, y1 - y0)}.png"


def _normalize_box(box: tuple[int, int, int, int], iw: int, ih: int) -> tuple[int, int, int, int] | None:
    normalized = _clamp_box(box, iw, ih)
    if normalized[2] <= normalized[0] or normalized[3] <= normalized[1]:
        return None
    return normalized


def _append_deduped_box(existing_boxes: list[tuple[int, int, int, int]], candidate_box: tuple[int, int, int, int], iw: int, ih: int) -> bool:
    normalized = _normalize_box(candidate_box, iw, ih)
    if normalized is None:
        return False

    candidate_area = _box_area(normalized)
    if candidate_area == 0:
        return False

    for existing in existing_boxes:
        if _box_iou(existing, normalized) >= 0.5:
            return False
        if _intersection_area(existing, normalized) / candidate_area >= 0.85:
            return False

    existing_boxes.append(normalized)
    return True


def _count_true_runs(values: list[bool]) -> int:
    runs = 0
    in_run = False
    for value in values:
        if value and not in_run:
            runs += 1
            in_run = True
        elif not value:
            in_run = False
    return runs


def _is_probable_text_fragment(box: tuple[int, int, int, int], img: Image.Image) -> bool:
    left, upper, right, lower = box
    width = max(0, right - left)
    height = max(0, lower - upper)
    if width == 0 or height == 0:
        return True

    page_width, page_height = img.size
    area_fraction = (width * height) / max(page_width * page_height, 1)
    aspect_ratio = width / max(height, 1)
    if area_fraction >= 0.004 or aspect_ratio < 6.0 or height > max(64, int(page_height * 0.12)):
        return False

    crop = img.crop((left, upper, right, lower)).convert("L")
    threshold_table = [255 if pixel > 235 else 0 for pixel in range(256)]
    mask = crop.point(threshold_table, mode="1")
    pixels = mask.tobytes()
    row_active: list[bool] = []
    col_active: list[bool] = []

    for row in range(height):
        ink_pixels = 0
        for col in range(width):
            if pixels[row * width + col] == 0:
                ink_pixels += 1
        row_active.append(ink_pixels >= max(1, int(width * 0.02)))

    for col in range(width):
        ink_pixels = 0
        for row in range(height):
            if pixels[row * width + col] == 0:
                ink_pixels += 1
        col_active.append(ink_pixels >= max(1, int(height * 0.02)))

    row_runs = _count_true_runs(row_active)
    col_runs = _count_true_runs(col_active)
    active_rows = sum(row_active)
    return row_runs <= 3 and col_runs <= 2 and active_rows <= max(12, int(height * 0.45))


def _accept_figure_candidate(box: tuple[int, int, int, int], img: Image.Image, source: str) -> bool:
    page_width, page_height = img.size
    width = box[2] - box[0]
    height = box[3] - box[1]
    area_fraction = _box_area(box) / max(page_width * page_height, 1)
    if width <= 0 or height <= 0:
        return False
    if _is_page_sized_box(box, page_width, page_height):
        return False
    if source.startswith("pdf-anchor"):
        return True
    if source == "page-components":
        if area_fraction < 0.015:
            return False
        if width < int(page_width * 0.15) or height < max(24, int(page_height * 0.085)):
            return False
    if width < max(20, int(page_width * 0.025)) or height < max(20, int(page_height * 0.025)):
        return False
    return not _is_probable_text_fragment(box, img)


def _component_text_penalty(
    component_box: tuple[int, int, int, int],
    original_foreground: list[bool],
    window_w: int,
    window_h: int,
) -> float:
    comp_w = max(component_box[2] - component_box[0], 1)
    comp_h = max(component_box[3] - component_box[1], 1)
    aspect_ratio = comp_w / comp_h
    ink_pixels = 0
    dense_rows = 0
    dense_cols = 0

    for y in range(component_box[1], component_box[3]):
        row_ink = 0
        row_offset = y * window_w
        for x in range(component_box[0], component_box[2]):
            if original_foreground[row_offset + x]:
                ink_pixels += 1
                row_ink += 1
        if row_ink >= max(1, int(comp_w * 0.18)):
            dense_rows += 1

    for x in range(component_box[0], component_box[2]):
        col_ink = 0
        for y in range(component_box[1], component_box[3]):
            if original_foreground[y * window_w + x]:
                col_ink += 1
        if col_ink >= max(1, int(comp_h * 0.18)):
            dense_cols += 1

    dense_row_fraction = dense_rows / comp_h
    dense_col_fraction = dense_cols / comp_w
    fill_ratio = ink_pixels / max(comp_w * comp_h, 1)
    touches_top_edge = component_box[1] <= max(2, int(window_h * 0.02))
    touches_bottom_edge = component_box[3] >= window_h - max(2, int(window_h * 0.02))

    penalty = 0.0
    if aspect_ratio >= 2.5 and dense_row_fraction >= 0.18 and dense_col_fraction <= 0.35:
        penalty += 1.5
    if aspect_ratio >= 4.0 and dense_row_fraction >= 0.1:
        penalty += 1.0
    if fill_ratio <= 0.22 and dense_row_fraction >= 0.2 and dense_col_fraction <= 0.3:
        penalty += 0.75
    if (touches_top_edge or touches_bottom_edge) and aspect_ratio >= 2.0 and dense_row_fraction >= 0.12:
        penalty += 0.75

    return penalty


def _refine_component_box_to_original_foreground(
    component_box: tuple[int, int, int, int], original_foreground: list[bool], width: int, height: int
) -> tuple[int, int, int, int] | None:
    refined_min_x, refined_min_y = width, height
    refined_max_x, refined_max_y = -1, -1

    for y in range(component_box[1], component_box[3]):
        row_offset = y * width
        for x in range(component_box[0], component_box[2]):
            if original_foreground[row_offset + x]:
                refined_min_x = min(refined_min_x, x)
                refined_min_y = min(refined_min_y, y)
                refined_max_x = max(refined_max_x, x)
                refined_max_y = max(refined_max_y, y)

    if refined_max_x < refined_min_x or refined_max_y < refined_min_y:
        return None

    return (refined_min_x, refined_min_y, refined_max_x + 1, refined_max_y + 1)


def _enumerate_page_component_boxes(img: "Image.Image") -> list[tuple[int, int, int, int]]:
    page_width, page_height = img.size
    gray = img.convert("L")
    original_foreground = [value < 235 for value in gray.tobytes()]
    mask = Image.frombytes(
        "L",
        (page_width, page_height),
        bytes(255 if value else 0 for value in original_foreground),
    )
    mask = mask.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MaxFilter(5))

    foreground = [value > 0 for value in mask.tobytes()]
    visited = bytearray(page_width * page_height)
    candidate_boxes: list[tuple[float, tuple[int, int, int, int]]] = []
    page_area = max(page_width * page_height, 1)

    for start_idx, is_foreground in enumerate(foreground):
        if not is_foreground or visited[start_idx]:
            continue

        queue = deque([start_idx])
        visited[start_idx] = 1
        min_x = max_x = start_idx % page_width
        min_y = max_y = start_idx // page_width

        while queue:
            idx = queue.popleft()
            x = idx % page_width
            y = idx // page_width

            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= nx < page_width and 0 <= ny < page_height:
                    n_idx = ny * page_width + nx
                    if foreground[n_idx] and not visited[n_idx]:
                        visited[n_idx] = 1
                        queue.append(n_idx)

        component_box = (min_x, min_y, max_x + 1, max_y + 1)
        area_fraction = _box_area(component_box) / page_area
        if area_fraction < 0.01:
            continue

        refined_box = _refine_component_box_to_original_foreground(component_box, original_foreground, page_width, page_height)
        if refined_box is None:
            continue

        penalty = _component_text_penalty(refined_box, original_foreground, page_width, page_height)
        if penalty >= 1.5:
            continue
        if not _accept_figure_candidate(refined_box, img, "page-components"):
            continue

        score = area_fraction - (penalty * 0.1)
        candidate_boxes.append((score, refined_box))

    candidate_boxes.sort(key=lambda item: (-item[0], item[1][1], item[1][0]))
    return [box for _, box in candidate_boxes[:8]]


def _report_looks_scanned(report) -> bool:
    if report is None or not report.image_elements:
        return False

    merged_images = _merge_image_elements(report.image_elements)
    if len(merged_images) != 1:
        return False

    page_area = max((report.mediabox.x1 - report.mediabox.x0) * (report.mediabox.y1 - report.mediabox.y0), 1.0)
    image_bbox = merged_images[0].bbox
    image_area = max((image_bbox.x1 - image_bbox.x0) * (image_bbox.y1 - image_bbox.y0), 0.0)
    return image_area / page_area >= 0.85


def _get_cached_page_report(pdf_path: str, page_num: int, report_cache: dict[int, PageReport | None]) -> PageReport | None:
    if page_num not in report_cache:
        try:
            report_cache[page_num] = _pdf_report(pdf_path, page_num)
        except Exception:
            report_cache[page_num] = None
    return report_cache[page_num]


def _find_best_anchor_box(
    model_box: tuple[int, int, int, int],
    iw: int,
    ih: int,
    pdf_path: str,
    page_num: int,
    report_cache: dict[int, PageReport | None],
) -> tuple[tuple[int, int, int, int] | None, PageReport | None]:
    report = _get_cached_page_report(pdf_path, page_num, report_cache)
    if report is None or not report.image_elements:
        return None, report

    pw = report.mediabox.x1 - report.mediabox.x0
    ph = report.mediabox.y1 - report.mediabox.y0
    if pw <= 0 or ph <= 0:
        return None, report

    sx = iw / pw
    sy = ih / ph

    best_overlap = 0
    best_box: tuple[int, int, int, int] | None = None
    for elem in _merge_image_elements(report.image_elements):
        anchor_box = (
            int((elem.bbox.x0 - report.mediabox.x0) * sx),
            int((ph - (elem.bbox.y1 - report.mediabox.y0)) * sy),
            int((elem.bbox.x1 - report.mediabox.x0) * sx),
            int((ph - (elem.bbox.y0 - report.mediabox.y0)) * sy),
        )
        overlap = _intersection_area(model_box, anchor_box)
        if overlap > best_overlap:
            best_overlap = overlap
            best_box = _clamp_box(anchor_box, iw, ih)

    if best_box is None:
        return None, report

    model_area = _box_area(model_box)
    if model_area <= 0 or best_overlap / model_area <= 0.1:
        return None, report

    return best_box, report


def _get_anchor_candidate_boxes(
    iw: int,
    ih: int,
    pdf_path: str,
    page_num: int,
    report_cache: dict[int, PageReport | None],
) -> tuple[list[tuple[int, int, int, int]], PageReport | None]:
    report = _get_cached_page_report(pdf_path, page_num, report_cache)
    if report is None or not report.image_elements:
        return [], report

    pw = report.mediabox.x1 - report.mediabox.x0
    ph = report.mediabox.y1 - report.mediabox.y0
    if pw <= 0 or ph <= 0:
        return [], report

    sx = iw / pw
    sy = ih / ph
    boxes: list[tuple[int, int, int, int]] = []
    for elem in _merge_image_elements(report.image_elements):
        box = (
            int((elem.bbox.x0 - report.mediabox.x0) * sx),
            int((ph - (elem.bbox.y1 - report.mediabox.y0)) * sy),
            int((elem.bbox.x1 - report.mediabox.x0) * sx),
            int((ph - (elem.bbox.y0 - report.mediabox.y0)) * sy),
        )
        normalized = _normalize_box(box, iw, ih)
        if normalized is None:
            continue
        if _is_page_sized_box(normalized, iw, ih):
            continue
        if _box_area(normalized) / max(iw * ih, 1) < 0.002:
            continue
        boxes.append(normalized)

    return boxes, report


def _pick_layout_detection(model_box: tuple[int, int, int, int], detections: list[LayoutDetection]) -> tuple[int, int, int, int] | None:
    if not detections:
        return None

    seed_cx = (model_box[0] + model_box[2]) / 2
    seed_cy = (model_box[1] + model_box[3]) / 2
    seed_w = max(model_box[2] - model_box[0], 1)
    seed_h = max(model_box[3] - model_box[1], 1)

    best_score = None
    best_box = None
    for detection in detections:
        overlap = _intersection_area(model_box, detection.box)
        contains_center = detection.box[0] <= seed_cx <= detection.box[2] and detection.box[1] <= seed_cy <= detection.box[3]
        if overlap == 0 and not contains_center:
            continue

        det_cx = (detection.box[0] + detection.box[2]) / 2
        det_cy = (detection.box[1] + detection.box[3]) / 2
        center_distance = abs(det_cx - seed_cx) / seed_w + abs(det_cy - seed_cy) / seed_h
        candidate_score = (overlap / max(_box_area(model_box), 1)) * 4.0 + detection.score - center_distance

        if best_score is None or candidate_score > best_score:
            best_score = candidate_score
            best_box = detection.box

    return best_box


def _enumerate_page_figure_refs(
    img: "Image.Image",
    pdf_path: str,
    page_num: int,
    report_cache: dict[int, PageReport | None],
    layout_model_name: str | None,
    layout_model_device: str,
    layout_model_score_threshold: float,
) -> list[DetectedFigureRef]:
    iw, ih = img.size
    refs: list[DetectedFigureRef] = []
    candidate_boxes: list[tuple[int, int, int, int]] = []

    anchor_boxes, report = _get_anchor_candidate_boxes(iw, ih, pdf_path, page_num, report_cache)
    if report is not None and not _report_looks_scanned(report):
        for anchor_box in anchor_boxes:
            if _accept_figure_candidate(anchor_box, img, "pdf-anchor-enum") and _append_deduped_box(candidate_boxes, anchor_box, iw, ih):
                refs.append(
                    DetectedFigureRef(
                        page_num=page_num,
                        box=candidate_boxes[-1],
                        filename=_box_to_ref_filename(page_num, candidate_boxes[-1]),
                        discovery_source="pdf-anchor-enum",
                    )
                )

    detector = get_figure_layout_detector(layout_model_name, layout_model_device, layout_model_score_threshold)
    if detector is not None:
        try:
            for detection in detector.detect(img):
                refined = _local_component_crop(detection.box, img, window_box=_expand_box(detection.box, 24, 24, iw, ih))
                candidate = refined if refined is not None else detection.box
                normalized = _normalize_box(candidate, iw, ih)
                if normalized is None:
                    continue
                source = "layout-detector-refined" if refined is not None else "layout-detector"
                if not _accept_figure_candidate(normalized, img, source):
                    continue
                if _append_deduped_box(candidate_boxes, normalized, iw, ih):
                    refs.append(
                        DetectedFigureRef(
                            page_num=page_num,
                            box=candidate_boxes[-1],
                            filename=_box_to_ref_filename(page_num, candidate_boxes[-1]),
                            discovery_source=source,
                        )
                    )
        except Exception as exc:
            logger.warning(
                f"Figure layout detection failed for {pdf_path} page {page_num} during page enumeration, falling back to structural extraction only: {exc}"
            )

    return refs


def detect_page_figure_refs(
    natural_text: str,
    pdf_path: str,
    page_spans: list | None = None,
    dim: int = 2048,
    layout_model_name: str | None = None,
    layout_model_device: str = "cpu",
    layout_model_score_threshold: float = 0.35,
) -> dict[int, list[DetectedFigureRef]]:
    ref_boxes_by_page = _extract_ref_boxes_by_page(natural_text, page_spans)
    page_text_by_number = _extract_page_texts(natural_text, page_spans)
    report_cache: dict[int, PageReport | None] = {}
    page_cache: dict[int, Image.Image] = {}

    if page_spans:
        page_numbers = [page_num for _, _, page_num in page_spans]
    else:
        page_numbers = sorted(ref_boxes_by_page) or [1]

    detected_refs_by_page: dict[int, list[DetectedFigureRef]] = {}
    for page_num in page_numbers:
        if page_num not in page_cache:
            b64 = render_pdf_to_base64png(pdf_path, page_num, target_longest_image_dim=dim)
            page_cache[page_num] = Image.open(BytesIO(base64.b64decode(b64)))

        img = page_cache[page_num]
        page_refs = _enumerate_page_figure_refs(
            img,
            pdf_path,
            page_num,
            report_cache,
            layout_model_name,
            layout_model_device,
            layout_model_score_threshold,
        )

        if not page_refs:
            fallback_boxes: list[tuple[int, int, int, int]] = []
            for box in ref_boxes_by_page.get(page_num, []):
                normalized = _normalize_box(box, img.width, img.height)
                if normalized is None:
                    continue
                refined_box, crop_source = _refine_figure_crop(
                    normalized[0],
                    normalized[1],
                    normalized[2] - normalized[0],
                    normalized[3] - normalized[1],
                    img,
                    pdf_path,
                    page_num,
                    report_cache,
                    layout_model_name=layout_model_name,
                    layout_model_device=layout_model_device,
                    layout_model_score_threshold=layout_model_score_threshold,
                )
                normalized_refined = _normalize_box(refined_box, img.width, img.height)
                if normalized_refined is None or not _accept_figure_candidate(normalized_refined, img, crop_source):
                    continue
                if _append_deduped_box(fallback_boxes, normalized_refined, img.width, img.height):
                    page_refs.append(
                        DetectedFigureRef(
                            page_num=page_num,
                            box=normalized_refined,
                            filename=_box_to_ref_filename(page_num, normalized_refined),
                            discovery_source=f"vlm-ref-fallback:{crop_source}",
                        )
                    )

        figure_mentions = _count_page_figure_mentions(page_text_by_number.get(page_num, ""))
        if figure_mentions > len(page_refs):
            existing_boxes = [ref.box for ref in page_refs]
            component_candidates_added = 0
            for component_box in _enumerate_page_component_boxes(img):
                if any(
                    _box_iou(component_box, existing_box) >= 0.35 or _intersection_area(component_box, existing_box) / max(_box_area(component_box), 1) >= 0.75
                    for existing_box in existing_boxes
                ):
                    continue
                existing_boxes.append(component_box)
                page_refs.append(
                    DetectedFigureRef(
                        page_num=page_num,
                        box=component_box,
                        filename=_box_to_ref_filename(page_num, component_box),
                        discovery_source="page-components",
                    )
                )
                component_candidates_added += 1
                if len(page_refs) >= figure_mentions:
                    break

            if component_candidates_added:
                logger.info(
                    f"Added {component_candidates_added} page-component figure refs on page {page_num} of {pdf_path} to match {figure_mentions} figure mentions"
                )

        if page_refs:
            detected_refs_by_page[page_num] = page_refs
            logger.info(
                f"Detected {len(page_refs)} canonical figure refs on page {page_num} of {pdf_path}: " + ", ".join(ref.discovery_source for ref in page_refs)
            )

    return detected_refs_by_page


def _local_component_crop(
    model_box: tuple[int, int, int, int],
    img: "Image.Image",
    window_box: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int, int] | None:
    iw, ih = img.size
    if window_box is None:
        margin_x = max(32, int((model_box[2] - model_box[0]) * 0.75))
        margin_y = max(32, int((model_box[3] - model_box[1]) * 0.75))
        window = _expand_box(model_box, margin_x, margin_y, iw, ih)
    else:
        window = _clamp_box(window_box, iw, ih)

    gray = img.convert("L").crop(window)
    window_w, window_h = gray.size
    if window_w == 0 or window_h == 0:
        return None

    # A small dilation helps merge nearby diagram fragments without jumping across
    # the larger whitespace gaps that typically separate captions and body text.
    original_foreground = [value < 235 for value in gray.tobytes()]
    mask = Image.frombytes(
        "L",
        (window_w, window_h),
        bytes(255 if value else 0 for value in original_foreground),
    )
    mask = mask.filter(ImageFilter.MaxFilter(5))

    pixels = mask.tobytes()
    foreground = [value > 0 for value in pixels]
    visited = bytearray(window_w * window_h)

    seed_local = (
        model_box[0] - window[0],
        model_box[1] - window[1],
        model_box[2] - window[0],
        model_box[3] - window[1],
    )
    seed_expanded = _expand_box(seed_local, 12, 12, window_w, window_h)
    seed_area = max(_box_area(seed_local), 1)
    seed_cx = (seed_local[0] + seed_local[2]) // 2
    seed_cy = (seed_local[1] + seed_local[3]) // 2
    window_area = max(window_w * window_h, 1)

    nearest_box: tuple[int, int, int, int] | None = None
    nearest_distance: float | None = None
    best_box: tuple[int, int, int, int] | None = None
    best_score: float | None = None

    for start_idx, is_foreground in enumerate(foreground):
        if not is_foreground or visited[start_idx]:
            continue

        queue = deque([start_idx])
        visited[start_idx] = 1
        min_x = max_x = start_idx % window_w
        min_y = max_y = start_idx // window_w

        while queue:
            idx = queue.popleft()
            x = idx % window_w
            y = idx // window_w

            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= nx < window_w and 0 <= ny < window_h:
                    n_idx = ny * window_w + nx
                    if foreground[n_idx] and not visited[n_idx]:
                        visited[n_idx] = 1
                        queue.append(n_idx)

        component_box = (min_x, min_y, max_x + 1, max_y + 1)
        overlap = _intersection_area(component_box, seed_expanded)
        contains_center = component_box[0] <= seed_cx <= component_box[2] and component_box[1] <= seed_cy <= component_box[3]

        comp_w = max(component_box[2] - component_box[0], 1)
        comp_h = max(component_box[3] - component_box[1], 1)
        aspect_ratio = max(comp_w / comp_h, comp_h / comp_w)
        area_fraction = _box_area(component_box) / window_area
        distance_norm = abs(((component_box[0] + component_box[2]) / 2) - seed_cx) / max(seed_local[2] - seed_local[0], 1) + abs(
            ((component_box[1] + component_box[3]) / 2) - seed_cy
        ) / max(seed_local[3] - seed_local[1], 1)
        text_like_penalty = _component_text_penalty(component_box, original_foreground, window_w, window_h)
        if aspect_ratio > 9 and min(comp_w, comp_h) < max(seed_local[2] - seed_local[0], seed_local[3] - seed_local[1]) * 0.2:
            text_like_penalty += 0.35

        if overlap > 0 or contains_center:
            component_score = (overlap / seed_area) * 4.0 + area_fraction * 1.5 - distance_norm - text_like_penalty
            if best_score is None or component_score > best_score:
                best_score = component_score
                best_box = component_box
            continue

        comp_cx = (component_box[0] + component_box[2]) / 2
        comp_cy = (component_box[1] + component_box[3]) / 2
        distance = abs(comp_cx - seed_cx) + abs(comp_cy - seed_cy)
        if nearest_distance is None or distance < nearest_distance:
            nearest_distance = distance
            nearest_box = component_box

    if best_box is None and nearest_box is not None:
        best_box = nearest_box

    if best_box is None:
        return None

    refined_min_x, refined_min_y = window_w, window_h
    refined_max_x, refined_max_y = -1, -1
    for y in range(best_box[1], best_box[3]):
        row_offset = y * window_w
        for x in range(best_box[0], best_box[2]):
            if original_foreground[row_offset + x]:
                refined_min_x = min(refined_min_x, x)
                refined_min_y = min(refined_min_y, y)
                refined_max_x = max(refined_max_x, x)
                refined_max_y = max(refined_max_y, y)

    if refined_max_x >= refined_min_x and refined_max_y >= refined_min_y:
        best_box = (refined_min_x, refined_min_y, refined_max_x + 1, refined_max_y + 1)

    return _clamp_box((window[0] + best_box[0], window[1] + best_box[1], window[0] + best_box[2], window[1] + best_box[3]), iw, ih)


def _refine_figure_crop(
    model_x: int,
    model_y: int,
    model_w: int,
    model_h: int,
    img: "Image.Image",
    pdf_path: str,
    page_num: int,
    report_cache: dict[int, PageReport | None],
    layout_model_name: str | None,
    layout_model_device: str,
    layout_model_score_threshold: float,
) -> tuple[tuple[int, int, int, int], str]:
    iw, ih = img.size
    model_box = _clamp_box((model_x, model_y, model_x + model_w, model_y + model_h), iw, ih)

    anchor_box, report = _find_best_anchor_box(model_box, iw, ih, pdf_path, page_num, report_cache)
    if anchor_box is not None and report is not None and not _report_looks_scanned(report) and not _is_page_sized_box(anchor_box, iw, ih):
        return anchor_box, "pdf-anchor"

    detector = get_figure_layout_detector(layout_model_name, layout_model_device, layout_model_score_threshold)
    if detector is not None:
        try:
            layout_box = _pick_layout_detection(model_box, detector.detect(img))
            if layout_box is not None:
                refined_layout_box = _local_component_crop(model_box, img, window_box=_expand_box(layout_box, 24, 24, iw, ih))
                if refined_layout_box is not None and _intersection_area(refined_layout_box, model_box) > 0:
                    return refined_layout_box, "layout-detector-refined"
                return _clamp_box(layout_box, iw, ih), "layout-detector"
        except Exception as exc:
            logger.warning(f"Figure layout detection failed for {pdf_path} page {page_num}, falling back to heuristic crop refinement: {exc}")

    local_box = _local_component_crop(model_box, img)
    if local_box is not None:
        return local_box, "local-components"

    if anchor_box is not None and not _is_page_sized_box(anchor_box, iw, ih):
        return anchor_box, "pdf-anchor-fallback"

    return model_box, "model-bbox"


def _anchor_crop(
    model_x: int, model_y: int, model_w: int, model_h: int, iw: int, ih: int, pdf_path: str, page_num: int, report_cache: dict
) -> tuple[int, int, int, int]:
    """Refine an approximate figure bbox using PDF image-element geometry when available."""
    if page_num not in report_cache:
        try:
            report_cache[page_num] = _pdf_report(pdf_path, page_num)
        except Exception:
            report_cache[page_num] = None

    report = report_cache[page_num]
    if report and report.image_elements:
        pw = report.mediabox.x1 - report.mediabox.x0
        ph = report.mediabox.y1 - report.mediabox.y0
        sx = iw / pw
        sy = ih / ph

        best_overlap = 0
        best_box: tuple[int, int, int, int] | None = None

        for elem in report.image_elements:
            a_left = int((elem.bbox.x0 - report.mediabox.x0) * sx)
            a_right = int((elem.bbox.x1 - report.mediabox.x0) * sx)
            a_upper = int((ph - (elem.bbox.y1 - report.mediabox.y0)) * sy)
            a_lower = int((ph - (elem.bbox.y0 - report.mediabox.y0)) * sy)

            ov_w = max(0, min(model_x + model_w, a_right) - max(model_x, a_left))
            ov_h = max(0, min(model_y + model_h, a_lower) - max(model_y, a_upper))
            overlap = ov_w * ov_h

            if overlap > best_overlap:
                best_overlap = overlap
                best_box = (a_left, a_upper, a_right, a_lower)

        model_area = model_w * model_h
        if best_box and model_area > 0 and best_overlap / model_area > 0.1:
            return (
                max(0, min(best_box[0], iw)),
                max(0, min(best_box[1], ih)),
                max(0, min(best_box[2], iw)),
                max(0, min(best_box[3], ih)),
            )

    return (
        max(0, min(model_x, iw)),
        max(0, min(model_y, ih)),
        max(0, min(model_x + model_w, iw)),
        max(0, min(model_y + model_h, ih)),
    )


def _parse_image_ref_filename(filename: str) -> tuple[int | None, int, int, int, int]:
    stem = os.path.splitext(filename)[0]
    parts = stem.split("_")
    if parts[0] != "page":
        raise ValueError(f"Unsupported image reference filename: {filename}")
    if len(parts) == 5:
        _, sx, sy, sw, sh = parts
        return None, int(sx), int(sy), int(sw), int(sh)
    if len(parts) == 6:
        _, spage, sx, sy, sw, sh = parts
        return int(spage), int(sx), int(sy), int(sw), int(sh)
    raise ValueError(f"Unsupported image reference filename: {filename}")


def _resolve_image_ref_page(ref_pos: int, page_spans: list | None) -> int:
    if page_spans:
        for span_start, span_end, page_num in page_spans:
            if span_start <= ref_pos < span_end:
                return page_num
        return page_spans[-1][2]
    return 1


def _get_page_qualified_image_ref(filename: str, ref_pos: int, page_spans: list | None = None) -> str:
    page_num, x, y, w, h = _parse_image_ref_filename(filename)
    if page_num is None:
        page_num = _resolve_image_ref_page(ref_pos, page_spans)
    return f"page_{page_num}_{x}_{y}_{w}_{h}.png"


def _qualify_page_markdown_image_refs(natural_text: str, page_num: int) -> str:
    matches = list(_IMAGE_REF_RE.finditer(natural_text))
    if not matches:
        return natural_text

    output_parts = []
    last_index = 0
    for match in matches:
        output_parts.append(natural_text[last_index : match.start(1)])
        _, x, y, w, h = _parse_image_ref_filename(match.group(1))
        output_parts.append(f"page_{page_num}_{x}_{y}_{w}_{h}.png")
        last_index = match.end(1)
    output_parts.append(natural_text[last_index:])
    return "".join(output_parts)


def _qualify_markdown_image_refs(natural_text: str, page_spans: list | None = None) -> str:
    matches = list(_IMAGE_REF_RE.finditer(natural_text))
    if not matches:
        return natural_text

    output_parts = []
    last_index = 0
    for match in matches:
        output_parts.append(natural_text[last_index : match.start(1)])
        output_parts.append(_get_page_qualified_image_ref(match.group(1), match.start(), page_spans))
        last_index = match.end(1)
    output_parts.append(natural_text[last_index:])
    return "".join(output_parts)


def _qualify_markdown_image_refs_with_page_spans(natural_text: str, page_spans: list | None = None) -> tuple[str, list | None]:
    if not page_spans:
        return _qualify_markdown_image_refs(natural_text, page_spans), page_spans

    qualified_parts = []
    qualified_page_spans = []
    current_char_pos = 0

    for span_start, span_end, page_num in page_spans:
        page_text = natural_text[span_start:span_end]
        qualified_page_text = _qualify_page_markdown_image_refs(page_text, page_num)
        qualified_parts.append(qualified_page_text)

        next_char_pos = current_char_pos + len(qualified_page_text)
        qualified_page_spans.append([current_char_pos, next_char_pos, page_num])
        current_char_pos = next_char_pos

    return "".join(qualified_parts), qualified_page_spans


def _extract_ref_boxes_by_page(natural_text: str, page_spans: list | None = None) -> dict[int, list[tuple[int, int, int, int]]]:
    page_boxes: dict[int, list[tuple[int, int, int, int]]] = {}
    for match in _IMAGE_REF_RE.finditer(natural_text):
        page_num, x, y, w, h = _parse_image_ref_filename(match.group(1))
        if page_num is None:
            page_num = _resolve_image_ref_page(match.start(), page_spans)
        page_boxes.setdefault(page_num, []).append((x, y, x + w, y + h))
    return page_boxes


def _extract_page_texts(natural_text: str, page_spans: list | None) -> dict[int, str]:
    if not page_spans:
        return {1: natural_text}
    return {page_num: natural_text[span_start:span_end] for span_start, span_end, page_num in page_spans}


def _escape_markdown_image_alt_text(alt_text: str) -> str:
    escaped: list[str] = []
    idx = 0
    while idx < len(alt_text):
        char = alt_text[idx]
        if char == "\\" and idx + 1 < len(alt_text):
            escaped.append(char)
            escaped.append(alt_text[idx + 1])
            idx += 2
            continue
        if char in {"_", "*", "[", "]"}:
            escaped.append("\\")
        escaped.append(char)
        idx += 1
    return "".join(escaped)


def _count_page_figure_mentions(page_text: str) -> int:
    return len({match.lower() for match in _FIGURE_CAPTION_RE.findall(page_text)})


def _rewrite_markdown_with_detected_refs(
    natural_text: str,
    page_spans: list | None,
    detected_refs_by_page: dict[int, list[DetectedFigureRef]],
) -> str:
    if not detected_refs_by_page:
        return natural_text

    existing_refs_by_page: dict[int, list[tuple[str, tuple[int, int, int, int]]]] = {}
    for match in _MARKDOWN_IMAGE_TAG_RE.finditer(natural_text):
        filename = match.group(2)
        page_num, x, y, w, h = _parse_image_ref_filename(filename)
        if page_num is None:
            page_num = _resolve_image_ref_page(match.start(), page_spans)
        existing_refs_by_page.setdefault(page_num, []).append((match.group(1).strip(), (x, y, x + w, y + h)))

    def _page_text_with_refs(page_text: str, page_num: int) -> str:
        cleaned = _MARKDOWN_IMAGE_TAG_RE.sub("", page_text).rstrip("\n")
        page_refs = detected_refs_by_page.get(page_num, [])
        if not page_refs:
            return cleaned

        unmatched_existing = list(existing_refs_by_page.get(page_num, []))
        rendered_refs: list[str] = []
        for detected_ref in page_refs:
            alt_text = "Figure"
            best_index = None
            best_score = 0.0
            for idx, (existing_alt, existing_box) in enumerate(unmatched_existing):
                score = _box_iou(detected_ref.box, existing_box)
                if score > best_score:
                    best_score = score
                    best_index = idx
            if best_index is not None and best_score >= 0.35:
                matched_alt, _ = unmatched_existing.pop(best_index)
                alt_text = matched_alt or "Figure"
            rendered_refs.append(f"![{_escape_markdown_image_alt_text(alt_text)}]({detected_ref.filename})")

        addition_text = "\n\n".join(rendered_refs)
        return cleaned + ("\n\n" if cleaned else "") + addition_text

    if not page_spans:
        return _page_text_with_refs(natural_text, 1)

    parts = []
    for span_start, span_end, page_num in page_spans:
        parts.append(_page_text_with_refs(natural_text[span_start:span_end], page_num))

    return "\n\n".join(parts)


def _augment_markdown_with_detected_refs(
    natural_text: str,
    page_spans: list | None,
    detected_refs_by_page: dict[int, list[str]],
) -> str:
    if not detected_refs_by_page:
        return natural_text
    canonical_refs_by_page = {
        page_num: [
            DetectedFigureRef(
                page_num=page_num,
                box=(x, y, x + w, y + h),
                filename=filename,
                discovery_source="compat",
            )
            for filename in refs
            for _, x, y, w, h in [_parse_image_ref_filename(filename)]
        ]
        for page_num, refs in detected_refs_by_page.items()
    }
    return _rewrite_markdown_with_detected_refs(natural_text, page_spans, canonical_refs_by_page)


def detect_missing_figure_refs(
    natural_text: str,
    pdf_path: str,
    page_spans: list | None = None,
    dim: int = 2048,
    layout_model_name: str | None = None,
    layout_model_device: str = "cpu",
    layout_model_score_threshold: float = 0.35,
) -> dict[int, list[str]]:
    canonical_refs_by_page = detect_page_figure_refs(
        natural_text,
        pdf_path,
        page_spans=page_spans,
        dim=dim,
        layout_model_name=layout_model_name,
        layout_model_device=layout_model_device,
        layout_model_score_threshold=layout_model_score_threshold,
    )
    return {page_num: [ref.filename for ref in refs] for page_num, refs in canonical_refs_by_page.items()}


def extract_page_images(
    natural_text: str,
    markdown_dir: str,
    pdf_path: str,
    page_spans: list | None = None,
    dim: int = 2048,
    layout_model_name: str | None = None,
    layout_model_device: str = "cpu",
    layout_model_score_threshold: float = 0.35,
    detected_refs_by_page: dict[int, list[str]] | None = None,
) -> None:
    """Crop and save figure images referenced in olmocr markdown output.

    The model emits references like ![caption](page_x_y_w_h.png) where
    startx/starty are pixel coordinates with top-left origin in the rendered
    image.  Page number is resolved from page_spans ([start, end, page_num]).
    Crop boundaries are refined using PDF image-element geometry for digital PDFs
    and layout/image analysis for scanned pages.
    """
    page_cache: dict[int, Image.Image] = {}
    report_cache: dict[int, PageReport | None] = {}

    filenames: list[str] = []
    seen_filenames: set[str] = set()
    for match in _IMAGE_REF_RE.finditer(natural_text):
        filename = match.group(1)
        if filename not in seen_filenames:
            seen_filenames.add(filename)
            filenames.append(filename)

    if detected_refs_by_page is None:
        detected_refs_by_page = {
            page_num: [ref.filename for ref in refs]
            for page_num, refs in detect_page_figure_refs(
                natural_text,
                pdf_path,
                page_spans=page_spans,
                dim=dim,
                layout_model_name=layout_model_name,
                layout_model_device=layout_model_device,
                layout_model_score_threshold=layout_model_score_threshold,
            ).items()
        }

    for refs in detected_refs_by_page.values():
        for filename in refs:
            if filename not in seen_filenames:
                seen_filenames.add(filename)
                filenames.append(filename)

    auto_detected_filenames = {ref for refs in detected_refs_by_page.values() for ref in refs}

    if not filenames:
        return

    for filename in filenames:
        dest = os.path.join(markdown_dir, filename)
        if os.path.exists(dest):
            continue

        page_num, x, y, w, h = _parse_image_ref_filename(filename)
        if page_num is None:
            page_num = 1

        if page_num not in page_cache:
            b64 = render_pdf_to_base64png(pdf_path, page_num, target_longest_image_dim=dim)
            page_cache[page_num] = Image.open(BytesIO(base64.b64decode(b64)))

        img = page_cache[page_num]
        (left, upper, right, lower), crop_source = _refine_figure_crop(
            x,
            y,
            w,
            h,
            img,
            pdf_path,
            page_num,
            report_cache,
            layout_model_name=layout_model_name,
            layout_model_device=layout_model_device,
            layout_model_score_threshold=layout_model_score_threshold,
        )

        if right > left and lower > upper:
            img.crop((left, upper, right, lower)).save(dest, format="PNG")
            ref_origin = "auto-detected" if filename in auto_detected_filenames else "vlm-ref"
            logger.info(f"Extracted figure {filename} from {pdf_path} page {page_num} via {crop_source} ({ref_origin})")


def get_markdown_path(workspace: str, source_file: str) -> str:
    """
    Calculate the markdown output path for a given source file.

    Args:
        workspace: The workspace directory path
        source_file: The original source file path (can be S3, local, or tarball::internal_path)

    Returns:
        The full path where the markdown file should be written
    """
    # Handle tarball paths (format: tarball_path::internal_path)
    if "::" in source_file:
        tarball_path, internal_path = source_file.split("::", 1)
        # Use tarball basename + internal path structure
        tarball_basename = os.path.splitext(os.path.basename(tarball_path))[0]
        if tarball_basename.endswith(".tar"):
            tarball_basename = tarball_basename[:-4]
        relative_path = os.path.join(tarball_basename, internal_path)
    elif source_file.startswith("s3://"):
        # Extract the path after the bucket name for S3 sources
        parsed = urlparse(source_file)
        relative_path = parsed.path.lstrip("/")
    else:
        # For local files, strip leading slash to make it relative
        relative_path = source_file.lstrip("/")

    # Sanitize path: remove any .. components to prevent path traversal
    parts = relative_path.split("/")
    safe_parts = [p for p in parts if p and p != ".."]
    relative_path = "/".join(safe_parts)

    # Change the extension to .md
    md_filename = os.path.splitext(os.path.basename(relative_path))[0] + ".md"
    # Get the directory path without the filename
    dir_path = os.path.dirname(relative_path)

    # Create the output markdown path
    markdown_dir = os.path.join(workspace, "markdown", dir_path)
    markdown_path = os.path.join(markdown_dir, md_filename)

    return markdown_path


async def worker(args, work_queue: WorkQueue, worker_id):
    while True:

        work_item = await work_queue.get_work()

        if work_item is None:
            logger.info(f"Worker {worker_id} exiting due to empty queue")
            break

        logger.info(f"Worker {worker_id} processing work item {work_item.hash}")
        await tracker.clear_work(worker_id)

        try:
            async with asyncio.TaskGroup() as tg:
                dolma_tasks = []
                for path in work_item.work_paths:
                    if is_tarball_path(path):
                        # Tarball returns a list of docs, so we handle it specially
                        dolma_tasks.append(tg.create_task(process_tarball(args, worker_id, path)))
                    else:
                        dolma_tasks.append(tg.create_task(process_pdf(args, worker_id, path)))
                logger.info(f"Created all tasks for {work_item.hash}")

            logger.info(f"Finished TaskGroup for worker on {work_item.hash}")

            dolma_docs = []
            for task in dolma_tasks:
                try:
                    result = task.result()
                except:
                    # some dolma doc creations may have failed
                    result = None

                if result is None:
                    continue
                # process_tarball returns a list, process_pdf returns a single doc
                if isinstance(result, list):
                    dolma_docs.extend(result)
                else:
                    dolma_docs.append(result)

            logger.info(f"Got {len(dolma_docs)} docs for {work_item.hash}")

            # Write the Dolma documents to a local temporary file in JSONL format
            with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tf:
                for doc in dolma_docs:
                    tf.write(json.dumps(doc))
                    tf.write("\n")
                tf.flush()
                temp_path = tf.name

            try:
                # Define the output S3 path using the work_hash
                output_final_path = os.path.join(args.workspace, "results", f"output_{work_item.hash}.jsonl")

                if output_final_path.startswith("s3://"):
                    bucket, key = parse_s3_path(output_final_path)
                    workspace_s3.upload_file(temp_path, bucket, key)
                else:
                    # Ensure the results directory exists for local workspace
                    os.makedirs(os.path.dirname(output_final_path), exist_ok=True)
                    shutil.copyfile(temp_path, output_final_path)
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

            # If --markdown flag is set, also write the natural text to markdown files
            if args.markdown:
                logger.info(f"Writing {len(dolma_docs)} markdown files for {work_item.hash}")
                for doc in dolma_docs:
                    source_file = doc["metadata"]["Source-File"]
                    page_spans = doc.get("attributes", {}).get("pdf_page_numbers")
                    natural_text, page_spans = _qualify_markdown_image_refs_with_page_spans(doc["text"], page_spans)
                    detected_refs_by_page: dict[int, list[str]] | None = None

                    markdown_path = get_markdown_path(args.workspace, source_file)
                    markdown_dir = os.path.dirname(markdown_path)

                    # Create the directory structure if it doesn't exist
                    if markdown_path.startswith("s3://"):
                        # For S3 paths, we'll create a temporary file and upload it
                        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as md_tf:
                            md_tf.write(natural_text)
                            md_tf.flush()
                            md_temp_path = md_tf.name

                        try:
                            md_bucket, md_key = parse_s3_path(markdown_path)
                            workspace_s3.upload_file(md_temp_path, md_bucket, md_key)
                        finally:
                            # Make sure to clean up the temporary file even if upload fails
                            if os.path.exists(md_temp_path):
                                os.unlink(md_temp_path)
                    else:
                        if (
                            not source_file.startswith("s3://")
                            and not source_file.startswith("gs://")
                            and not source_file.startswith("weka://")
                            and "::" not in source_file
                        ):
                            canonical_refs_by_page = detect_page_figure_refs(
                                natural_text,
                                source_file,
                                page_spans=page_spans,
                                layout_model_name=args.figure_layout_model,
                                layout_model_device=args.figure_layout_device,
                                layout_model_score_threshold=args.figure_layout_score_threshold,
                            )
                            natural_text = _rewrite_markdown_with_detected_refs(natural_text, page_spans, canonical_refs_by_page)
                            detected_refs_by_page = {page_num: [ref.filename for ref in refs] for page_num, refs in canonical_refs_by_page.items()}

                        # For local paths, create the directory structure and write the file
                        os.makedirs(markdown_dir, exist_ok=True)
                        with open(markdown_path, "w") as md_f:
                            md_f.write(natural_text)

                        # Extract figure images when the source PDF is also local
                        if (
                            not source_file.startswith("s3://")
                            and not source_file.startswith("gs://")
                            and not source_file.startswith("weka://")
                            and "::" not in source_file
                        ):
                            try:
                                extract_page_images(
                                    natural_text,
                                    markdown_dir,
                                    source_file,
                                    page_spans=page_spans,
                                    layout_model_name=args.figure_layout_model,
                                    layout_model_device=args.figure_layout_device,
                                    layout_model_score_threshold=args.figure_layout_score_threshold,
                                    detected_refs_by_page=detected_refs_by_page,
                                )
                            except Exception as img_err:
                                logger.warning(f"Image extraction failed for {source_file}: {img_err}")

            # Update finished token counts from successful documents
            metrics.add_metrics(
                finished_input_tokens=sum(doc["metadata"]["total-input-tokens"] for doc in dolma_docs),
                finished_output_tokens=sum(doc["metadata"]["total-output-tokens"] for doc in dolma_docs),
            )

            await work_queue.mark_done(work_item)
        except Exception as e:
            logger.exception(f"Exception occurred while processing work_hash {work_item.hash}: {e}")


async def vllm_server_task(model_name_or_path, args, unknown_args=None):
    cmd = [
        "vllm",
        "serve",
        model_name_or_path,
        "--port",
        str(args.port),
        "--disable-log-requests",
        "--uvicorn-log-level",
        "warning",
        "--served-model-name",
        "olmocr",
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--data-parallel-size",
        str(args.data_parallel_size),
        "--limit-mm-per-prompt",
        '{"video": 0}',  # Disabling video encoder saves RAM that you can put towards the KV cache, thanks @charitarthchugh
    ]

    if args.gpu_memory_utilization is not None:
        cmd.extend(["--gpu-memory-utilization", str(args.gpu_memory_utilization)])

    if args.max_model_len is not None:
        cmd.extend(["--max-model-len", str(args.max_model_len)])

    if unknown_args:
        cmd.extend(unknown_args)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        # OMP_NUM_THREADS needs to be 1, otherwise you could have contention if you are running multiple copies of olmOCR on a machine with several GPUS
        env={**os.environ, "OMP_NUM_THREADS": "1"},
    )

    # Ensure the subprocess is terminated on exit
    def _kill_proc():
        try:
            proc.terminate()
        except:
            logger.info("VLLM Process already terminated")

    atexit.register(_kill_proc)

    # Shared variables between tasks
    last_running_req, peak_running_req, last_queue_req = 0, 0, 0
    server_printed_ready_message = False

    async def process_line(line):
        nonlocal last_running_req, last_queue_req, peak_running_req, server_printed_ready_message
        server_logger.info(line)

        if "Detected errors during sampling" in line:
            logger.error("Cannot continue, sampling errors detected, model is probably corrupt")
            sys.exit(1)

        if not server_printed_ready_message and ("The server is fired up and ready to roll!" in line or "Starting vLLM API server" in line):
            server_printed_ready_message = True

        if match := re.search(r"Running: (\d+)", line):
            current_running = int(match.group(1))
            # Track peak running requests
            if current_running > peak_running_req:
                peak_running_req = current_running
                logger.info(f"New peak running requests: {peak_running_req}")
            last_running_req = current_running

        if match := re.search(r"(?:Waiting|Pending):\s*(\d+)", line):
            global vllm_queued_requests
            last_queue_req = int(match.group(1))
            vllm_queued_requests = last_queue_req
            logger.info(f"vllm running req: {last_running_req} queue req: {last_queue_req}")

    async def read_stream(stream):
        while True:
            line = await stream.readline()
            if not line:
                break
            try:
                line = line.decode("utf-8").rstrip()
                await process_line(line)
            except Exception as ex:
                logger.warning(f"Got {ex} when reading log line from inference server, skipping")

    # Start tasks to read stdout, stderr, and handle timeout logic
    stdout_task = asyncio.create_task(read_stream(proc.stdout))
    stderr_task = asyncio.create_task(read_stream(proc.stderr))

    try:
        await proc.wait()
    except asyncio.CancelledError:
        logger.info("Got cancellation request for VLLM server")
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("VLLM server did not terminate within 10 seconds")
        raise

    await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)


async def vllm_server_host(model_name_or_path, args, unknown_args=None):
    MAX_RETRIES = 5
    retry = 0

    while retry < MAX_RETRIES:
        await vllm_server_task(model_name_or_path, args, unknown_args)
        logger.warning("VLLM server task ended")
        retry += 1

    if retry >= MAX_RETRIES:
        logger.error(f"Ended up starting the vllm server more than {retry} times, cancelling pipeline")
        logger.error("")
        logger.error(
            "Please make sure vllm is installed according to the latest instructions here: https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html"
        )
        sys.exit(1)


async def vllm_server_ready(args):
    max_attempts = args.max_server_ready_timeout
    delay_sec = 1
    url = f"{args.server.rstrip('/')}/models"

    for attempt in range(1, max_attempts + 1):
        try:
            headers = {}
            if args.server and hasattr(args, "api_key") and args.api_key:
                headers["Authorization"] = f"Bearer {args.api_key}"

            async with httpx.AsyncClient() as session:
                response = await session.get(url, headers=headers)

                if response.status_code == 200:
                    logger.info("vllm server is ready.")
                    return
                else:
                    logger.info(f"Attempt {attempt}: Unexpected status code {response.status_code}")
        except Exception:
            logger.warning(f"Attempt {attempt}: Please wait for vllm server to become ready...")

        await asyncio.sleep(delay_sec)

    raise Exception("vllm server did not become ready after waiting.")


async def download_model(model_name_or_path: str, max_retries: int = 5):
    for retry in range(max_retries):
        try:
            if model_name_or_path.startswith("s3://") or model_name_or_path.startswith("gs://") or model_name_or_path.startswith("weka://"):
                logger.info(f"Downloading model directory from '{model_name_or_path}'")
                model_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "olmocr", "model")
                # Delete existing model cache directory if it exists
                if os.path.exists(model_cache_dir):
                    shutil.rmtree(model_cache_dir)
                download_directory([model_name_or_path], model_cache_dir)
                return model_cache_dir
            elif os.path.isabs(model_name_or_path) and os.path.isdir(model_name_or_path):
                logger.info(f"Using local model path at '{model_name_or_path}'")
                return model_name_or_path
            else:
                logger.info(f"Downloading model with hugging face '{model_name_or_path}'")
                snapshot_download(repo_id=model_name_or_path)
                return model_name_or_path
        except Exception:
            if retry == max_retries - 1:
                raise  # Raise on final attempt and fail the job

            sleep_time = random.randrange(2, 20) * 2**retry
            logger.exception(f"Could not download model, sleeping for {sleep_time} seconds to retry ({retry + 1}/{max_retries})")
            await asyncio.sleep(random.randrange(10, 30) * 2**retry)


async def metrics_reporter(work_queue):
    while True:
        # Leading newlines preserve table formatting in logs
        logger.info(f"Queue remaining: {work_queue.size}")
        logger.info("\n" + str(metrics))
        logger.info("\n" + str(await tracker.get_status_table()))
        await asyncio.sleep(10)


def submit_beaker_job(args):
    from beaker import (  # type: ignore
        Beaker,
        BeakerConstraints,
        BeakerEnvVar,
        BeakerExperimentSpec,
        BeakerImageSource,
        BeakerJobPriority,
        BeakerResultSpec,
        BeakerRetrySpec,
        BeakerTaskContext,
        BeakerTaskResources,
        BeakerTaskSpec,
    )
    from beaker.exceptions import BeakerSecretNotFound

    Beaker.TIMEOUT = 60
    b = Beaker.from_env(default_workspace=args.beaker_workspace)
    owner = b.user_name
    beaker_image = f"jakep/olmocr-inference-{VERSION}"

    task_name = f"olmocr-{os.path.basename(args.workspace.rstrip('/'))}"

    # Take out --beaker flag so the workers will just run things
    args_list = [arg for arg in sys.argv[1:] if arg != "--beaker"]

    # Take out the --pdfs [arg] or --pdfs=[arg], since the queue is populated locally
    args_list = [arg for i, arg in enumerate(args_list) if not (arg.startswith("--pdfs") or (i > 0 and args_list[i - 1] == "--pdfs"))]

    try:
        b.secret.get(f"{owner}-WEKA_ACCESS_KEY_ID")
        b.secret.get(f"{owner}-WEKA_SECRET_ACCESS_KEY")
        b.secret.get(f"{owner}-AWS_CREDENTIALS_FILE")
    except BeakerSecretNotFound:
        print(
            f"Expected beaker secrets for accessing Weka and S3 are not found. Are you okay to write those to your beaker workspace {args.beaker_workspace}? [y/n]"
        )

        if input().strip().lower() != "y":
            print("Exiting...")
            sys.exit(1)

        b.secret.write(f"{owner}-WEKA_ACCESS_KEY_ID", os.environ.get("WEKA_ACCESS_KEY_ID", ""))
        b.secret.write(f"{owner}-WEKA_SECRET_ACCESS_KEY", os.environ.get("WEKA_SECRET_ACCESS_KEY", ""))
        b.secret.write(
            f"{owner}-AWS_CREDENTIALS_FILE",
            open(os.path.join(os.path.expanduser("~"), ".aws", "credentials")).read(),
        )

    env_var_secrets = [
        BeakerEnvVar(name="WEKA_ACCESS_KEY_ID", secret=f"{owner}-WEKA_ACCESS_KEY_ID"),
        BeakerEnvVar(name="WEKA_SECRET_ACCESS_KEY", secret=f"{owner}-WEKA_SECRET_ACCESS_KEY"),
        BeakerEnvVar(name="AWS_CREDENTIALS_FILE", secret=f"{owner}-AWS_CREDENTIALS_FILE"),
    ]

    try:
        b.secret.get("OLMOCR_PREVIEW_HF_TOKEN")
        env_var_secrets.append(BeakerEnvVar(name="HF_TOKEN", secret="OLMOCR_PREVIEW_HF_TOKEN"))
    except BeakerSecretNotFound:
        pass

    try:
        b.secret.get("OE_DATA_GCS_SA_KEY")
        env_var_secrets.append(BeakerEnvVar(name="GOOGLE_APPLICATION_CREDENTIALS_FILE", secret="OE_DATA_GCS_SA_KEY"))
    except BeakerSecretNotFound:
        print("Input the olmo-gcs SA key if you would like to load weights from gcs (end with a double newline):")
        lines = []
        prev_empty = False
        for line in iter(input, None):
            if not line and prev_empty:
                break
            prev_empty = not line
            lines.append(line)
        gcs_sa_key = "\n".join(lines[:-1]).strip()  # Remove the last empty line
        if gcs_sa_key:
            b.secret.write("OE_DATA_GCS_SA_KEY", gcs_sa_key)
            env_var_secrets.append(BeakerEnvVar(name="GOOGLE_APPLICATION_CREDENTIALS_FILE", secret="OE_DATA_GCS_SA_KEY"))

    # Create the experiment spec
    experiment_spec = BeakerExperimentSpec(
        budget="ai2/oe-base",
        description=task_name,
        tasks=[
            BeakerTaskSpec(
                name=task_name,
                propagate_failure=False,
                propagate_preemption=False,
                replicas=args.beaker_gpus,
                context=BeakerTaskContext(
                    priority=BeakerJobPriority[args.beaker_priority],
                    preemptible=True,
                ),
                image=BeakerImageSource(beaker=beaker_image),
                command=["python", "-m", "olmocr.pipeline"] + args_list,
                env_vars=[
                    BeakerEnvVar(name="BEAKER_JOB_NAME", value=task_name),
                    BeakerEnvVar(name="OWNER", value=owner),
                    BeakerEnvVar(name="HF_HUB_OFFLINE", value="1"),
                ]
                + env_var_secrets,
                resources=BeakerTaskResources(gpu_count=1, memory="125GB"),  # Have to set a memory limit, otherwise VLLM may use too much on its own
                constraints=BeakerConstraints(cluster=args.beaker_cluster if isinstance(args.beaker_cluster, list) else [args.beaker_cluster]),
                result=BeakerResultSpec(path="/noop-results"),
            )
        ],
        retry=BeakerRetrySpec(allowed_task_retries=10),
    )

    workload = b.experiment.create(spec=experiment_spec)

    print(f"Experiment URL: https://beaker.org/ex/{workload.experiment.id}")


def print_stats(args, root_work_queue):
    LONG_CONTEXT_THRESHOLD = 32768
    assert args.workspace.startswith("s3://"), "Printing stats functionality only works with s3 workspaces for now."

    done_work_items = expand_s3_glob(workspace_s3, os.path.join(args.workspace, "results", "*.jsonl"))
    work_queue_lines = download_zstd_csv(workspace_s3, os.path.join(args.workspace, "work_index_list.csv.zstd"))
    work_queue = {parts[0]: parts[1:] for line in work_queue_lines if line.strip() and (parts := root_work_queue._decode_csv_row(line.strip()))}

    total_items, completed_items = len(work_queue), len(done_work_items)

    def process_output_file(s3_path):
        try:
            stats = {
                "docs": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "pages": 0,
                "fallback_pages": 0,
                "long_docs": 0,
                "long_tokens": 0,
                "en_docs": 0,
                "en_tokens": 0,
            }
            paths = set()
            for line in get_s3_bytes(workspace_s3, s3_path).decode("utf-8").splitlines():
                if not line.strip():
                    continue
                doc = json.loads(line)
                meta, attrs = doc["metadata"], doc.get("attributes", {})
                out_tokens = meta.get("total-output-tokens", 0)
                stats["docs"] += 1
                stats["input_tokens"] += meta.get("total-input-tokens", 0)
                stats["output_tokens"] += out_tokens
                stats["pages"] += meta.get("pdf-total-pages", 0)
                stats["fallback_pages"] += meta.get("total-fallback-pages", 0)
                paths.add(meta["Source-File"])
                if out_tokens > LONG_CONTEXT_THRESHOLD:
                    stats["long_docs"] += 1
                    stats["long_tokens"] += out_tokens
                langs = attrs.get("primary_language", [])
                if langs and sum(1 for ln in langs if ln == "en") > len(langs) / 2:
                    stats["en_docs"] += 1
                    stats["en_tokens"] += out_tokens
            return stats, paths
        except Exception as e:
            logger.warning(f"Error processing {s3_path}: {e}")
            return {
                k: 0 for k in ["docs", "input_tokens", "output_tokens", "pages", "fallback_pages", "long_docs", "long_tokens", "en_docs", "en_tokens"]
            }, set()

    print(f"\nCompleted work items {completed_items:,} out of {total_items:,}: {completed_items/total_items*100:.2f}%")
    print("\nProcessing output files...")

    totals = {"docs": 0, "input_tokens": 0, "output_tokens": 0, "pages": 0, "fallback_pages": 0, "long_docs": 0, "long_tokens": 0, "en_docs": 0, "en_tokens": 0}
    all_processed, original_paths = set(), set()

    for item in done_work_items:
        if (match := re.search(r"output_(\w+).jsonl", item)) and match.group(1) in work_queue:
            original_paths.update(work_queue[match.group(1)])

    with ThreadPoolExecutor() as executor:
        for stats, paths in tqdm(executor.map(process_output_file, done_work_items), total=len(done_work_items)):
            for k in totals:
                totals[k] += stats[k]
            all_processed.update(paths)

    d, p, o, c = totals["docs"], totals["pages"], totals["output_tokens"], max(1, completed_items)
    print(
        f"""
Work Items Status:
Total work items: {total_items:,}
Completed items: {completed_items:,}
Remaining items: {total_items - completed_items:,}

Results:
Total documents processed: {d:,}
Total documents skipped: {len(original_paths - all_processed):,}
Total pages on fallback: {totals['fallback_pages']:,}
Total pages processed: {p:,}

Total output tokens: {o:,}
Projected output tokens: {round(o / c * total_items):,}

Average pages per doc: {p / max(1, d):,.1f}
Average output tokens per doc: {o / max(1, d):,.1f}
Average output tokens per page: {o / max(1, p):,.1f}

Long Context Documents (>{LONG_CONTEXT_THRESHOLD} tokens): {totals['long_docs']:,}
Total tokens in long context documents: {totals['long_tokens']:,}

English-only documents (>50% pages with 'en'): {totals['en_docs']:,}
Total output tokens in English-only documents: {totals['en_tokens']:,}
Projected English-only output tokens: {round(totals['en_tokens'] / c * total_items):,}"""
    )


async def main():
    parser = argparse.ArgumentParser(description="Manager for running millions of PDFs through a batch inference pipeline.")
    parser.add_argument(
        "workspace",
        help="The filesystem path where work will be stored, can be a local folder, or an s3 path if coordinating work with many workers, s3://bucket/prefix/ ",
    )
    parser.add_argument(
        "--pdfs",
        nargs="*",
        help="Path to add pdfs stored in s3 to the workspace, can be a glob path s3://bucket/prefix/*.pdf or path to file containing list of pdf paths",
        default=None,
    )
    parser.add_argument(
        "--model",
        help="Path where the model is located, allenai/olmOCR-2-7B-1025-FP8 is the default, can be local, s3, or hugging face.",
        default="allenai/olmOCR-2-7B-1025-FP8",
    )

    # More detailed config options, usually you shouldn't have to change these
    parser.add_argument("--workspace_profile", help="S3 configuration profile for accessing the workspace", default=None)
    parser.add_argument("--pdf_profile", help="S3 configuration profile for accessing the raw pdf documents", default=None)
    parser.add_argument("--pages_per_group", type=int, default=argparse.SUPPRESS, help="Aiming for this many pdf pages per work item group")
    parser.add_argument("--max_page_retries", type=int, default=8, help="Max number of times we will retry rendering a page")
    parser.add_argument("--max_page_error_rate", type=float, default=0.004, help="Rate of allowable failed pages in a document, 1/250 by default")
    parser.add_argument("--workers", type=int, default=20, help="Number of workers to run at a time")
    parser.add_argument("--max_concurrent_requests", type=int, default=1600, help="Max number of concurrent VLLM server requests at a time.")
    parser.add_argument("--max_server_ready_timeout", type=int, default=600, help="Number of seconds to wait for vllm to become ready before exiting.")
    parser.add_argument("--apply_filter", action="store_true", help="Apply basic filtering to English pdfs which are not forms, and not likely seo spam")
    parser.add_argument("--stats", action="store_true", help="Instead of running any job, reports some statistics about the current workspace")
    parser.add_argument("--markdown", action="store_true", help="Also write natural text to markdown files preserving the folder structure of the input pdfs")
    parser.add_argument(
        "--figure_layout_model",
        type=str,
        default="Aryn/deformable-detr-DocLayNet",
        help="Optional Hugging Face layout detection model used to localize figures on scanned markdown pages. Set to 'none' to disable.",
    )
    parser.add_argument(
        "--figure_layout_device",
        type=str,
        default="auto",
        help="Device for the figure layout detector during markdown image extraction. Use 'auto', 'cpu', or a torch device such as 'cuda' or 'cuda:0'.",
    )
    parser.add_argument(
        "--figure_layout_score_threshold",
        type=float,
        default=0.35,
        help="Minimum detection score for scanned-page figure layout boxes.",
    )
    parser.add_argument("--target_longest_image_dim", type=int, help="Dimension on longest side to use for rendering the pdf pages", default=1288)
    parser.add_argument("--target_anchor_text_len", type=int, help="Maximum amount of anchor text to use (characters), not used for new models", default=-1)
    parser.add_argument("--guided_decoding", action="store_true", help="Enable guided decoding for model YAML type outputs")
    parser.add_argument(
        "--disk_logging",
        type=str,
        nargs="?",
        const="olmocr-pipeline-debug.log",
        default=None,
        help="Enable writing logs to disk, optionally specify filename (default: olmocr-pipeline-debug.log)",
    )

    server_group = parser.add_argument_group("Server arguments, to specify where your VLLM inference engine is running")
    server_group.add_argument(
        "--server",
        type=str,
        help="URL of external vLLM (or other compatible provider) server (e.g., http://hostname:port/v1). If provided, skips spawning local vLLM instance",
    )
    server_group.add_argument("--api_key", type=str, default=None, help="API key for authenticated remote servers (e.g., DeepInfra)")

    vllm_group = parser.add_argument_group(
        "VLLM arguments", "These arguments are passed to vLLM. Any unrecognized arguments are also automatically forwarded to vLLM."
    )
    vllm_group.add_argument(
        "--gpu-memory-utilization", type=float, help="Fraction of VRAM vLLM may pre-allocate for KV-cache " "(passed through to vllm serve)."
    )
    vllm_group.add_argument("--max_model_len", type=int, default=16384, help="Upper bound (tokens) vLLM will allocate KV-cache for, lower if VLLM won't start")
    vllm_group.add_argument("--tensor-parallel-size", "-tp", type=int, default=1, help="Tensor parallel size for vLLM")
    vllm_group.add_argument("--data-parallel-size", "-dp", type=int, default=1, help="Data parallel size for vLLM")
    vllm_group.add_argument("--port", type=int, default=30024, help="Port to use for the VLLM server")

    # Beaker/job running stuff
    beaker_group = parser.add_argument_group("beaker/cluster execution")
    beaker_group.add_argument("--beaker", action="store_true", help="Submit this job to beaker instead of running locally")
    beaker_group.add_argument("--beaker_workspace", help="Beaker workspace to submit to", default="ai2/olmocr")
    beaker_group.add_argument(
        "--beaker_cluster",
        help="Beaker clusters you want to run on",
        default=["ai2/jupiter", "ai2/ceres", "ai2/neptune", "ai2/saturn"],
    )
    beaker_group.add_argument("--beaker_gpus", type=int, default=1, help="Number of gpu replicas to run")
    beaker_group.add_argument("--beaker_priority", type=str, default="normal", help="Beaker priority level for the job")

    args, unknown_args = parser.parse_known_args()

    # Set up file logging if enabled
    if args.disk_logging:
        file_handler = logging.FileHandler(args.disk_logging, mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)
        server_logger.addHandler(file_handler)

    logger.info(
        "If you run out of GPU memory during start-up or get 'KV cache is larger than available memory' errors, retry with lower values, e.g. --gpu_memory_utilization 0.80  --max_model_len 16384"
    )

    use_internal_server = not args.server
    global workspace_s3, pdf_s3, max_concurrent_requests_limit

    max_concurrent_requests_limit = asyncio.BoundedSemaphore(args.max_concurrent_requests)

    # setup the job to work in beaker environment, load secrets, adjust logging, etc.
    if "BEAKER_JOB_NAME" in os.environ:
        cred_path = os.path.join(os.path.expanduser("~"), ".aws", "credentials")
        os.makedirs(os.path.dirname(cred_path), exist_ok=True)
        with open(cred_path, "w") as f:
            f.write(os.environ.get("AWS_CREDENTIALS_FILE"))
        cred_path = os.path.join(os.path.expanduser("~"), ".gcs", "credentials")
        os.makedirs(os.path.dirname(cred_path), exist_ok=True)
        with open(cred_path, "w") as f:
            f.write(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_FILE"))
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path
        workspace_s3 = boto3.client("s3")
        pdf_s3 = boto3.client("s3")

        # Wait a little bit so that not all beaker jobs in a task start at the same time and download the model at the same time
        replica_count = int(os.environ.get("BEAKER_REPLICA_COUNT", "1"))
        interval = 10 if (replica_count - 1) * 10 <= 30 else 30 / max(1, replica_count - 1)
        sleep_time = int(os.environ.get("BEAKER_REPLICA_RANK", "0")) * interval
        logger.info(f"Beaker job sleeping for {sleep_time} seconds to stagger model downloads")
        await asyncio.sleep(sleep_time)

    # If you specify an API key, meaning you are on a remote provider, then lower the group size default, not to overwhelm such servers
    # and not to waste money if a group doesn't finish right away
    if not hasattr(args, "pages_per_group"):
        args.pages_per_group = 50 if args.api_key is not None else 500

    if args.workspace_profile:
        workspace_session = boto3.Session(profile_name=args.workspace_profile)
        workspace_s3 = workspace_session.client("s3")

    if args.pdf_profile:
        pdf_session = boto3.Session(profile_name=args.pdf_profile)
        pdf_s3 = pdf_session.client("s3")

    # We need poppler to load the initial pdfs, even if we are not processing them here
    check_poppler_version()

    # Create work queue
    if args.workspace.startswith("s3://"):
        work_queue = WorkQueue(S3Backend(workspace_s3, args.workspace))
    else:
        work_queue = WorkQueue(LocalBackend(args.workspace))

    if args.pdfs:
        logger.info("Got --pdfs argument, going to add to the work queue")
        pdf_work_paths = set()
        tarball_paths = set()

        for pdf_path in args.pdfs:
            # Expand s3 glob paths first, then categorize results
            if pdf_path.startswith("s3://"):
                logger.info(f"Expanding s3 glob at {pdf_path}")
                expanded_paths = set(expand_s3_glob(pdf_s3, pdf_path))
                tarball_paths.update(p for p in expanded_paths if is_tarball_path(p))
                pdf_work_paths.update(p for p in expanded_paths if not is_tarball_path(p))
            elif os.path.exists(pdf_path):
                # Check if this is a tar.gz file (local)
                if is_tarball_path(pdf_path):
                    tarball_paths.add(pdf_path)
                elif (
                    pdf_path.lower().endswith(".pdf")
                    or pdf_path.lower().endswith(".png")
                    or pdf_path.lower().endswith(".jpg")
                    or pdf_path.lower().endswith(".jpeg")
                ):
                    if open(pdf_path, "rb").read(4) == b"%PDF":
                        logger.info(f"Loading file at {pdf_path} as PDF document")
                        pdf_work_paths.add(pdf_path)
                    elif is_png(pdf_path) or is_jpeg(pdf_path):
                        logger.info(f"Loading file at {pdf_path} as image document")
                        pdf_work_paths.add(pdf_path)
                    else:
                        logger.warning(f"File at {pdf_path} is not a valid PDF")
                elif pdf_path.lower().endswith(".txt"):
                    logger.info(f"Loading file at {pdf_path} as list of paths")
                    with open(pdf_path, "r") as f:
                        lines = [line.strip() for line in f if line.strip()]
                    tarball_paths.update(p for p in lines if is_tarball_path(p))
                    pdf_work_paths.update(p for p in lines if not is_tarball_path(p))
                else:
                    raise ValueError(f"Unsupported file extension for {pdf_path}")
            else:
                raise ValueError("pdfs argument needs to be either a local path, an s3 path, or an s3 glob pattern...")

        logger.info(f"Found {len(pdf_work_paths):,} regular pdf paths and {len(tarball_paths):,} tarballs to add")

        # Process regular PDFs with calculated items_per_group
        if pdf_work_paths:
            # Estimate average pages per pdf
            sample_size = min(100, len(pdf_work_paths))
            sampled_pdfs = random.sample(list(pdf_work_paths), sample_size)
            page_counts = []

            for pdf in tqdm(sampled_pdfs, desc="Sampling PDFs to calculate optimal length"):
                try:
                    # Download the PDF to a temp file
                    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_file:
                        tmp_file.write(get_s3_bytes(pdf_s3, pdf))
                        tmp_file.flush()
                        if is_png(tmp_file.name) or is_jpeg(tmp_file.name):
                            page_counts.append(1)
                        else:
                            reader = PdfReader(tmp_file.name)
                            page_counts.append(len(reader.pages))
                except Exception as e:
                    logger.warning(f"Failed to read {pdf}: {e}")

            if page_counts:
                avg_pages_per_pdf = sum(page_counts) / len(page_counts)
            else:
                logger.warning("Could not read any PDFs to estimate average page count.")
                avg_pages_per_pdf = 10  # Default to 10 pages per PDF if sampling fails

            items_per_group = max(1, int(args.pages_per_group / avg_pages_per_pdf))
            logger.info(f"Calculated items_per_group: {items_per_group} based on average pages per PDF: {avg_pages_per_pdf:.2f}")

            # Now call populate_queue for regular PDFs
            await work_queue.populate_queue(list(pdf_work_paths), items_per_group)

        # Add tarballs to the queue - each tarball is one work item
        if tarball_paths:
            await work_queue.populate_queue(tarball_paths, 1)

    if args.stats:
        print_stats(args, work_queue)
        return

    if args.beaker:
        submit_beaker_job(args)
        return

    # If you get this far, then you are doing inference and need a GPU
    # check_sglang_version()
    if use_internal_server:
        check_torch_gpu_available()

    logger.info(f"Starting pipeline with PID {os.getpid()}")

    # Download the model before you do anything else
    if use_internal_server:
        model_name_or_path = await download_model(args.model)
        args.server = f"http://localhost:{args.port}/v1"
        args.model = "olmocr"  # Internal server always uses this name for the model, for supporting weird local model paths
        logger.info(f"Using internal server at {args.server}")
    else:
        logger.info(f"Using external server at {args.server}")
        model_name_or_path = None

    # Initialize the work queue
    qsize = await work_queue.initialize_queue()

    if qsize == 0:
        logger.info("No work to do, exiting")
        return

    # Start local vLLM instance if not using external one
    vllm_server = None
    if use_internal_server:
        vllm_server = asyncio.create_task(vllm_server_host(model_name_or_path, args, unknown_args))

    await vllm_server_ready(args)

    metrics_task = asyncio.create_task(metrics_reporter(work_queue))

    # Create worker tasks to process the queue concurrently.
    worker_tasks = []
    for i in range(args.workers):
        task = asyncio.create_task(worker(args, work_queue, worker_id=i))
        worker_tasks.append(task)

    # Wait for all worker tasks to finish
    await asyncio.gather(*worker_tasks)

    # Cancel vLLM server if it was started
    if vllm_server is not None:
        vllm_server.cancel()
    metrics_task.cancel()

    # Wait for cancelled tasks to complete
    tasks_to_wait = [metrics_task]
    if vllm_server is not None:
        tasks_to_wait.append(vllm_server)
    await asyncio.gather(*tasks_to_wait, return_exceptions=True)

    # Output final metrics summary
    metrics_summary = metrics.get_metrics_summary()
    logger.info("=" * 80)
    logger.info("FINAL METRICS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total elapsed time: {metrics_summary['elapsed_time_seconds']:.2f} seconds")

    # Output token counts and rates
    total_metrics = metrics_summary["total_metrics"]
    rates = metrics_summary["rates"]

    logger.info(f"Total Server Input tokens: {total_metrics.get('server_input_tokens', 0):,}")
    logger.info(f"Total Server Output tokens: {total_metrics.get('server_output_tokens', 0):,}")

    logger.info(f"Finished input tokens: {total_metrics.get('finished_input_tokens', 0):,}")
    logger.info(f"Finished output tokens: {total_metrics.get('finished_output_tokens', 0):,}")

    logger.info(f"Completed pages: {total_metrics.get('completed_pages', 0):,}")
    logger.info(f"Failed pages: {total_metrics.get('failed_pages', 0):,}")
    logger.info(
        f"Page Failure rate: {total_metrics.get('failed_pages', 0) / max(total_metrics.get('completed_pages', 0) + total_metrics.get('failed_pages', 0), 1) * 100:.2f}%"
    )

    # Output finished_on_attempt statistics
    logger.info("")
    logger.info("Pages finished by attempt number:")
    total_finished = sum(total_metrics.get(f"finished_on_attempt_{i}", 0) for i in range(args.max_page_retries))
    cumulative = 0

    for i in range(args.max_page_retries):
        if f"finished_on_attempt_{i}" in total_metrics:
            count = total_metrics[f"finished_on_attempt_{i}"]
            cumulative += count
            percentage = (count / total_finished * 100) if total_finished > 0 else 0
            cumulative_percentage = (cumulative / total_finished * 100) if total_finished > 0 else 0
            logger.info(f"  Attempt {i}: {count:,} pages ({percentage:.1f}%) - Cumulative: {cumulative:,} ({cumulative_percentage:.1f}%)")

    # Output rates
    if "server_input_tokens_per_sec" in rates:
        logger.info(f"Server Input tokens/sec rate: {rates['server_input_tokens_per_sec']:.2f}")
    if "server_output_tokens_per_sec" in rates:
        logger.info(f"Server Output tokens/sec rate: {rates['server_output_tokens_per_sec']:.2f}")
    if "finished_input_tokens_per_sec" in rates:
        logger.info(f"Finished Input tokens/sec rate: {rates['finished_input_tokens_per_sec']:.2f}")
    if "finished_output_tokens_per_sec" in rates:
        logger.info(f"Finished Output tokens/sec rate: {rates['finished_output_tokens_per_sec']:.2f}")

    logger.info("=" * 80)
    logger.info("Work done")


def cli_main():
    """Synchronous entry point for the CLI."""
    return asyncio.run(main())


if __name__ == "__main__":
    cli_main()
