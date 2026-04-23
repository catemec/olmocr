import base64
import json
import os
import warnings
from dataclasses import dataclass
from io import BytesIO
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from olmocr.pipeline import (
    DetectedFigureRef,
    LayoutDetection,
    PageResult,
    _prefix_markdown_image_refs,
    _strip_junk_figure_refs,
    _vlm_verify_is_figure,
    _load_layout_detector_model,
    _augment_markdown_with_detected_refs,
    _qualify_markdown_image_refs,
    _qualify_markdown_image_refs_with_page_spans,
    _rewrite_markdown_with_detected_refs,
    build_page_query,
    detect_page_figure_refs,
    detect_missing_figure_refs,
    extract_page_images,
    get_markdown_asset_dir,
    get_markdown_path,
    process_page,
)
from olmocr.prompts.anchor import BoundingBox, ImageElement, PageReport


def create_test_image(width=100, height=150):
    """Create a simple test image with distinct features to verify rotation."""
    img = Image.new("RGB", (width, height), color="white")
    pixels = img.load()

    # Draw a red square in top-left corner
    for x in range(10, 30):
        for y in range(10, 30):
            if pixels is not None:
                pixels[x, y] = (255, 0, 0)

    # Draw a blue rectangle in bottom-right corner
    for x in range(width - 40, width - 10):
        for y in range(height - 30, height - 10):
            if pixels is not None:
                pixels[x, y] = (0, 0, 255)

    # Draw a green line near the top
    for x in range(20, 80):
        if pixels is not None:
            pixels[x, 5] = (0, 255, 0)

    return img


def image_to_base64(img):
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def base64_to_image(base64_str):
    """Convert base64 string to PIL Image."""
    image_bytes = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_bytes))


class TestFigureLayoutDetectorLoading:
    def test_load_layout_detector_model_filters_known_meta_parameter_warning(self):
        class DummyLoader:
            @staticmethod
            def from_pretrained(model_name):
                warnings.warn_explicit(
                    "for conv1.weight: copying from a non-meta parameter in the checkpoint to a meta parameter in the current model, which is a no-op.",
                    UserWarning,
                    filename="ignored.py",
                    lineno=1,
                    module="torch.nn.modules.module",
                )
                return {"model_name": model_name}

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loaded = _load_layout_detector_model(DummyLoader, "dummy-layout-model")

        assert loaded == {"model_name": "dummy-layout-model"}
        assert caught == []


class TestImageRotation:
    @pytest.mark.asyncio
    async def test_no_rotation(self):
        """Test that image_rotation=0 returns the original image."""
        test_img = create_test_image()
        test_base64 = image_to_base64(test_img)

        with patch("olmocr.pipeline.render_pdf_to_base64png") as mock_render:
            mock_render.return_value = test_base64

            result = await build_page_query("fake_pdf.pdf", 1, 1000, image_rotation=0)

            # Extract the image from the result
            messages = result["messages"]
            content = messages[0]["content"]
            image_url = content[1]["image_url"]["url"]
            image_base64 = image_url.split(",")[1]
            result_img = base64_to_image(image_base64)

            # Should be the same size as original
            assert result_img.size == test_img.size

            # Check pixel at specific location (red square should be in top-left)
            assert result_img.getpixel((20, 20)) == (255, 0, 0)

    @pytest.mark.asyncio
    async def test_rotate_90_degrees(self):
        """Test that image_rotation=90 rotates the image 90 degrees counter-clockwise."""
        test_img = create_test_image(100, 150)
        test_base64 = image_to_base64(test_img)

        with patch("olmocr.pipeline.render_pdf_to_base64png") as mock_render:
            mock_render.return_value = test_base64

            result = await build_page_query("fake_pdf.pdf", 1, 1000, image_rotation=90)

            # Extract the image from the result
            messages = result["messages"]
            content = messages[0]["content"]
            image_url = content[1]["image_url"]["url"]
            image_base64 = image_url.split(",")[1]
            result_img = base64_to_image(image_base64)

            # After 90 degree counter-clockwise rotation, dimensions should be swapped
            assert result_img.size == (150, 100)

            # The red square that was at top-left should now be at bottom-left
            # Original (20, 20) -> After 90° CCW rotation -> (20, 80)
            assert result_img.getpixel((20, 80)) == (255, 0, 0)

    @pytest.mark.asyncio
    async def test_rotate_180_degrees(self):
        """Test that image_rotation=180 rotates the image 180 degrees."""
        test_img = create_test_image(100, 150)
        test_base64 = image_to_base64(test_img)

        with patch("olmocr.pipeline.render_pdf_to_base64png") as mock_render:
            mock_render.return_value = test_base64

            result = await build_page_query("fake_pdf.pdf", 1, 1000, image_rotation=180)

            # Extract the image from the result
            messages = result["messages"]
            content = messages[0]["content"]
            image_url = content[1]["image_url"]["url"]
            image_base64 = image_url.split(",")[1]
            result_img = base64_to_image(image_base64)

            # After 180 degree rotation, dimensions should be the same
            assert result_img.size == (100, 150)

            # The red square that was at top-left should now be at bottom-right
            # Original (20, 20) -> After 180° rotation -> (80, 130)
            assert result_img.getpixel((80, 130)) == (255, 0, 0)

    @pytest.mark.asyncio
    async def test_rotate_270_degrees(self):
        """Test that image_rotation=270 rotates the image 270 degrees counter-clockwise (90 clockwise)."""
        test_img = create_test_image(100, 150)
        test_base64 = image_to_base64(test_img)

        with patch("olmocr.pipeline.render_pdf_to_base64png") as mock_render:
            mock_render.return_value = test_base64

            result = await build_page_query("fake_pdf.pdf", 1, 1000, image_rotation=270)

            # Extract the image from the result
            messages = result["messages"]
            content = messages[0]["content"]
            image_url = content[1]["image_url"]["url"]
            image_base64 = image_url.split(",")[1]
            result_img = base64_to_image(image_base64)

            # After 270 degree counter-clockwise rotation, dimensions should be swapped
            assert result_img.size == (150, 100)

            # The red square that was at top-left should now be at top-right
            # Original (20, 20) -> After 270° CCW rotation -> (130, 20)
            assert result_img.getpixel((130, 20)) == (255, 0, 0)

    @pytest.mark.asyncio
    async def test_invalid_rotation_angle(self):
        """Test that invalid rotation angles raise an assertion error."""
        test_img = create_test_image()
        test_base64 = image_to_base64(test_img)

        with patch("olmocr.pipeline.render_pdf_to_base64png") as mock_render:
            mock_render.return_value = test_base64

            with pytest.raises(AssertionError, match="Invalid image rotation"):
                await build_page_query("fake_pdf.pdf", 1, 1000, image_rotation=45)

    @pytest.mark.asyncio
    async def test_rotation_preserves_image_quality(self):
        """Test that rotation preserves the image without distortion."""
        # Create a more complex test image
        test_img = create_test_image(200, 300)
        test_base64 = image_to_base64(test_img)

        with patch("olmocr.pipeline.render_pdf_to_base64png") as mock_render:
            mock_render.return_value = test_base64

            # Test all valid rotation angles
            for angle in [0, 90, 180, 270]:
                result = await build_page_query("fake_pdf.pdf", 1, 1000, image_rotation=angle)

                # Extract the image from the result
                messages = result["messages"]
                content = messages[0]["content"]
                image_url = content[1]["image_url"]["url"]
                image_base64 = image_url.split(",")[1]
                result_img = base64_to_image(image_base64)

                # Verify image format is preserved
                assert result_img.format == "PNG" or result_img.format is None
                assert result_img.mode == "RGB"


@dataclass
class MockArgs:
    max_page_retries: int = 8
    target_longest_image_dim: int = 1288
    guided_decoding: bool = False
    server: str = "http://localhost:30000/v1"
    model: str = "olmocr"


class TestRotationCorrection:
    @pytest.mark.asyncio
    async def test_process_page_with_rotation_correction(self):
        """Test that process_page correctly handles rotation correction from model response."""

        # Path to the test PDF that needs 90 degree rotation
        test_pdf_path = "tests/gnarly_pdfs/edgar-rotated90.pdf"

        # Mock arguments
        args = MockArgs()

        # Counter to track number of API calls
        call_count = 0

        async def mock_apost(url, json_data, api_key=None):
            nonlocal call_count
            call_count += 1

            # Check the rotation in the request
            messages = json_data.get("messages", [])
            if messages:
                content = messages[0].get("content", [])
                image_data = content[0].get("image_url", {}).get("url", "")

                # First call - model detects rotation is needed
                if call_count == 1:
                    response_content = """---
primary_language: en
is_rotation_valid: false
rotation_correction: 90
is_table: false
is_diagram: false
---

This document appears to be rotated and needs correction."""

                # Second call - after rotation, model says it's correct
                elif call_count == 2:
                    response_content = """---
primary_language: en
is_rotation_valid: true
rotation_correction: 0
is_table: false
is_diagram: false
---

UNITED STATES
SECURITIES AND EXCHANGE COMMISSION
Washington, D.C. 20549

This is the corrected text from the document."""

                else:
                    raise ValueError(f"Unexpected call count: {call_count}")

            # Mock response structure
            response_body = {
                "choices": [{"message": {"content": response_content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100},
            }

            return 200, json.dumps(response_body).encode()

        # Mock the worker tracker
        mock_tracker = AsyncMock()

        # Ensure the test PDF exists
        assert os.path.exists(test_pdf_path), f"Test PDF not found at {test_pdf_path}"

        # Track calls to build_page_query
        build_page_query_calls = []
        original_build_page_query = build_page_query

        async def mock_build_page_query(local_pdf_path, page, target_longest_image_dim, image_rotation=0, model_name="olmocr"):
            build_page_query_calls.append(image_rotation)
            return await original_build_page_query(local_pdf_path, page, target_longest_image_dim, image_rotation, model_name)

        with patch("olmocr.pipeline.apost", side_effect=mock_apost):
            with patch("olmocr.pipeline.tracker", mock_tracker):
                with patch("olmocr.pipeline.build_page_query", side_effect=mock_build_page_query):
                    result = await process_page(args=args, worker_id=0, pdf_orig_path="test-edgar-rotated90.pdf", pdf_local_path=test_pdf_path, page_num=1)

        # Verify the result
        assert isinstance(result, PageResult)
        assert result.page_num == 1
        assert result.is_fallback == False
        assert result.response.is_rotation_valid == True
        assert result.response.rotation_correction == 0
        assert result.response.natural_text is not None
        assert "SECURITIES AND EXCHANGE COMMISSION" in result.response.natural_text

        # Verify that exactly 2 API calls were made
        assert call_count == 2

        # Verify build_page_query was called with correct rotations
        assert len(build_page_query_calls) == 2
        assert build_page_query_calls[0] == 0  # First call with no rotation
        assert build_page_query_calls[1] == 90  # Second call with 90 degree rotation

        # Verify tracker was called correctly
        mock_tracker.track_work.assert_any_call(0, "test-edgar-rotated90.pdf-1", "started")
        mock_tracker.track_work.assert_any_call(0, "test-edgar-rotated90.pdf-1", "finished")

    @pytest.mark.asyncio
    async def test_process_page_with_cumulative_rotation(self):
        """Test that process_page correctly accumulates rotations across multiple attempts."""

        # Path to the test PDF (can use any test PDF)
        test_pdf_path = "tests/gnarly_pdfs/edgar-rotated90.pdf"

        # Mock arguments
        args = MockArgs()

        # Counter to track number of API calls
        call_count = 0

        async def mock_apost(url, json_data, api_key=None):
            nonlocal call_count
            call_count += 1

            # First call - model detects rotation is needed (90 degrees)
            if call_count == 1:
                response_content = """---
primary_language: en
is_rotation_valid: false
rotation_correction: 90
is_table: false
is_diagram: false
---

This document appears to be rotated and needs correction."""

            # Second call - model still detects rotation is needed (another 90 degrees)
            elif call_count == 2:
                response_content = """---
primary_language: en
is_rotation_valid: false
rotation_correction: 90
is_table: false
is_diagram: false
---

Document still needs rotation."""

            # Third call - after 180 total degrees of rotation, model says it's correct
            elif call_count == 3:
                response_content = """---
primary_language: en
is_rotation_valid: true
rotation_correction: 0
is_table: false
is_diagram: false
---

UNITED STATES
SECURITIES AND EXCHANGE COMMISSION
Washington, D.C. 20549

Document is now correctly oriented after 180 degree rotation."""

            else:
                raise ValueError(f"Unexpected call count: {call_count}")

            # Mock response structure
            response_body = {
                "choices": [{"message": {"content": response_content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100},
            }

            return 200, json.dumps(response_body).encode()

        # Mock the worker tracker
        mock_tracker = AsyncMock()

        # Ensure the test PDF exists
        assert os.path.exists(test_pdf_path), f"Test PDF not found at {test_pdf_path}"

        # Track calls to build_page_query
        build_page_query_calls = []
        original_build_page_query = build_page_query

        async def mock_build_page_query(local_pdf_path, page, target_longest_image_dim, image_rotation=0, model_name="olmocr"):
            build_page_query_calls.append(image_rotation)
            return await original_build_page_query(local_pdf_path, page, target_longest_image_dim, image_rotation, model_name)

        with patch("olmocr.pipeline.apost", side_effect=mock_apost):
            with patch("olmocr.pipeline.tracker", mock_tracker):
                with patch("olmocr.pipeline.build_page_query", side_effect=mock_build_page_query):
                    result = await process_page(args=args, worker_id=0, pdf_orig_path="test-cumulative-rotation.pdf", pdf_local_path=test_pdf_path, page_num=1)

        # Verify the result
        assert isinstance(result, PageResult)
        assert result.page_num == 1
        assert result.is_fallback == False
        assert result.response.is_rotation_valid == True
        assert result.response.rotation_correction == 0
        assert result.response.natural_text is not None
        assert "180 degree rotation" in result.response.natural_text

        # Verify that exactly 3 API calls were made
        assert call_count == 3

        # Verify build_page_query was called with correct cumulative rotations
        assert len(build_page_query_calls) == 3
        assert build_page_query_calls[0] == 0  # First call with no rotation
        assert build_page_query_calls[1] == 90  # Second call with 90 degree rotation
        assert build_page_query_calls[2] == 180  # Third call with cumulative 180 degree rotation

        # Verify tracker was called correctly
        mock_tracker.track_work.assert_any_call(0, "test-cumulative-rotation.pdf-1", "started")
        mock_tracker.track_work.assert_any_call(0, "test-cumulative-rotation.pdf-1", "finished")

    @pytest.mark.asyncio
    async def test_process_page_rotation_wraps_around(self):
        """Test that cumulative rotation correctly wraps around at 360 degrees."""

        # Path to the test PDF
        test_pdf_path = "tests/gnarly_pdfs/edgar-rotated90.pdf"

        # Mock arguments
        args = MockArgs()

        # Counter to track number of API calls
        call_count = 0

        async def mock_apost(url, json_data, api_key=None):
            nonlocal call_count
            call_count += 1

            # First call - model detects rotation is needed (270 degrees)
            if call_count == 1:
                response_content = """---
primary_language: en
is_rotation_valid: false
rotation_correction: 270
is_table: false
is_diagram: false
---

Document needs 270 degree rotation."""

            # Second call - model detects more rotation is needed (180 degrees)
            # Total would be 450, but should wrap to 90
            elif call_count == 2:
                response_content = """---
primary_language: en
is_rotation_valid: false
rotation_correction: 180
is_table: false
is_diagram: false
---

Document needs additional rotation."""

            # Third call - after wrapped rotation (90 degrees), model says it's correct
            elif call_count == 3:
                response_content = """---
primary_language: en
is_rotation_valid: true
rotation_correction: 0
is_table: false
is_diagram: false
---

Document correctly oriented at 90 degrees total rotation."""

            else:
                raise ValueError(f"Unexpected call count: {call_count}")

            # Mock response structure
            response_body = {
                "choices": [{"message": {"content": response_content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100},
            }

            return 200, json.dumps(response_body).encode()

        # Mock the worker tracker
        mock_tracker = AsyncMock()

        # Ensure the test PDF exists
        assert os.path.exists(test_pdf_path), f"Test PDF not found at {test_pdf_path}"

        # Track calls to build_page_query
        build_page_query_calls = []
        original_build_page_query = build_page_query

        async def mock_build_page_query(local_pdf_path, page, target_longest_image_dim, image_rotation=0, model_name="olmocr"):
            build_page_query_calls.append(image_rotation)
            return await original_build_page_query(local_pdf_path, page, target_longest_image_dim, image_rotation, model_name)

        with patch("olmocr.pipeline.apost", side_effect=mock_apost):
            with patch("olmocr.pipeline.tracker", mock_tracker):
                with patch("olmocr.pipeline.build_page_query", side_effect=mock_build_page_query):
                    result = await process_page(args=args, worker_id=0, pdf_orig_path="test-rotation-wrap.pdf", pdf_local_path=test_pdf_path, page_num=1)

        # Verify the result
        assert isinstance(result, PageResult)
        assert result.page_num == 1
        assert result.is_fallback == False
        assert result.response.is_rotation_valid == True

        # Verify that exactly 3 API calls were made
        assert call_count == 3

        # Verify build_page_query was called with correct cumulative rotations
        assert len(build_page_query_calls) == 3
        assert build_page_query_calls[0] == 0  # First call with no rotation
        assert build_page_query_calls[1] == 270  # Second call with 270 degree rotation
        assert build_page_query_calls[2] == 90  # Third call with wrapped rotation (270 + 180 = 450 % 360 = 90)


class TestMarkdownPathHandling:
    """Tests for the get_markdown_path function to ensure files stay within workspace."""

    def test_absolute_local_path_stays_in_workspace(self):
        """
        Test that absolute local paths produce markdown paths inside the workspace.

        This is a regression test for a bug where passing absolute paths like
        /home/user/documents/file.pdf would cause the markdown output to be written
        to /home/user/documents/file.md instead of workspace/markdown/.../file.md
        """
        workspace = "/tmp/test_workspace"
        source_file = "/home/ubuntu/test_documents/subfolder/test_document.pdf"

        markdown_path = get_markdown_path(workspace, source_file)

        # The markdown path should be inside the workspace
        assert markdown_path.startswith(workspace), (
            f"BUG: Markdown path '{markdown_path}' is outside workspace '{workspace}'. " f"Absolute source paths should not escape the workspace directory."
        )

    def test_s3_path_stays_in_workspace(self):
        """Test that S3 paths produce markdown paths inside the workspace."""
        workspace = "/tmp/test_workspace"
        source_file = "s3://my-bucket/documents/subfolder/test_document.pdf"

        markdown_path = get_markdown_path(workspace, source_file)

        assert markdown_path.startswith(workspace)
        assert markdown_path == "/tmp/test_workspace/markdown/documents/subfolder/test_document.md"

    def test_relative_local_path_stays_in_workspace(self):
        """Test that relative local paths produce markdown paths inside the workspace."""
        workspace = "/tmp/test_workspace"
        source_file = "documents/subfolder/test_document.pdf"

        markdown_path = get_markdown_path(workspace, source_file)

        assert markdown_path.startswith(workspace)
        assert markdown_path == "/tmp/test_workspace/markdown/documents/subfolder/test_document.md"

    def test_path_traversal_with_dotdot_stays_in_workspace(self):
        """Test that paths with ../ do not escape the workspace directory."""
        workspace = "/tmp/test_workspace"
        source_file = "documents/../../../etc/passwd.pdf"

        markdown_path = get_markdown_path(workspace, source_file)

        # Resolve the path to check if it actually stays in workspace
        resolved_path = os.path.normpath(markdown_path)
        resolved_workspace = os.path.normpath(workspace)

        assert resolved_path.startswith(resolved_workspace), (
            f"BUG: Path traversal attack! Markdown path '{resolved_path}' escapes " f"workspace '{resolved_workspace}'. Paths with ../ should be sanitized."
        )

    def test_markdown_asset_dir_is_scoped_per_document(self):
        workspace = "/tmp/test_workspace"

        optimizer_markdown_path = get_markdown_path(workspace, "/home/rsgarci1/myHDD/RkrishnanGehrke/Ch15_Optimizer.pdf")
        transaction_markdown_path = get_markdown_path(workspace, "/home/rsgarci1/myHDD/RkrishnanGehrke/Ch16_Transaction.pdf")

        assert get_markdown_asset_dir(optimizer_markdown_path) == "/tmp/test_workspace/markdown/home/rsgarci1/myHDD/RkrishnanGehrke/Ch15_Optimizer"
        assert get_markdown_asset_dir(transaction_markdown_path) == "/tmp/test_workspace/markdown/home/rsgarci1/myHDD/RkrishnanGehrke/Ch16_Transaction"
        assert get_markdown_asset_dir(optimizer_markdown_path) != get_markdown_asset_dir(transaction_markdown_path)


class TestMarkdownImageExtraction:
    def test_prefix_markdown_image_refs_targets_document_asset_dir(self):
        natural_text = "Text before ![Figure A](page_1_10_20_30_40.png) and ![Figure B](page_2_11_21_31_41.png)"

        rewritten = _prefix_markdown_image_refs(natural_text, "Ch16_Transaction")

        assert "![Figure A](Ch16_Transaction/page_1_10_20_30_40.png)" in rewritten
        assert "![Figure B](Ch16_Transaction/page_2_11_21_31_41.png)" in rewritten

    def test_qualify_markdown_image_refs_adds_page_numbers(self):
        natural_text = "Page one ![Figure A](page_10_20_30_40.png)\n\nPage two ![Figure B](page_10_20_30_40.png)"
        page_spans = [[0, 44, 1], [44, len(natural_text), 2]]

        qualified = _qualify_markdown_image_refs(natural_text, page_spans)

        assert "page_1_10_20_30_40.png" in qualified
        assert "page_2_10_20_30_40.png" in qualified
        assert qualified.count("page_10_20_30_40.png") == 0

    def test_qualify_markdown_image_refs_with_page_spans_rebuilds_offsets(self):
        natural_text = "Page one ![Figure A](page_10_20_30_40.png)\n\n" "Page two ![Figure B](page_10_20_30_40.png)"
        page_spans = [[0, 44, 1], [44, len(natural_text), 2]]

        qualified, qualified_page_spans = _qualify_markdown_image_refs_with_page_spans(natural_text, page_spans)

        assert qualified == ("Page one ![Figure A](page_1_10_20_30_40.png)\n\n" "Page two ![Figure B](page_2_10_20_30_40.png)")
        assert qualified_page_spans == [[0, 46, 1], [46, len(qualified), 2]]

    def test_rewrite_markdown_with_detected_refs_uses_rebuilt_page_spans_after_qualification(self):
        page_one_text = "Page one ![Figure A](page_10_20_30_40.png)\n\n" "The entity sets that participate in a relationship set need not be distinct. Since"
        page_two_text = "employees report to other employees."
        natural_text = page_one_text + page_two_text
        page_spans = [[0, len(page_one_text), 1], [len(page_one_text), len(natural_text), 2]]

        qualified_text, qualified_page_spans = _qualify_markdown_image_refs_with_page_spans(natural_text, page_spans)
        rewritten = _rewrite_markdown_with_detected_refs(
            qualified_text,
            qualified_page_spans,
            {
                1: [
                    DetectedFigureRef(
                        page_num=1,
                        box=(10, 20, 40, 60),
                        filename="page_1_10_20_30_40.png",
                        discovery_source="layout-detector",
                    )
                ],
                2: [
                    DetectedFigureRef(
                        page_num=2,
                        box=(10, 20, 40, 60),
                        filename="page_2_10_20_30_40.png",
                        discovery_source="layout-detector",
                    )
                ],
            },
        )

        assert "Since\n\n![Figure A](page_1_10_20_30_40.png)" in rewritten
        assert "employees report to other employees.\n\n![Figure](page_2_10_20_30_40.png)" in rewritten
        assert "Sin\n\n![Figure A](page_1_10_20_30_40.png)" not in rewritten
        assert rewritten.count("ce employees report to other employees.") == 0

    def test_rewrite_markdown_preserves_vlm_refs_on_pages_without_detections(self):
        page_one_text = "Page one ![Figure A](page_1_10_20_30_40.png)\n\n"
        page_two_text = "Page two ![Figure B](page_2_50_60_20_20.png)\n"
        natural_text = page_one_text + page_two_text
        page_spans = [[0, len(page_one_text), 1], [len(page_one_text), len(natural_text), 2]]

        rewritten = _rewrite_markdown_with_detected_refs(
            natural_text,
            page_spans,
            {
                1: [
                    DetectedFigureRef(
                        page_num=1,
                        box=(10, 20, 40, 60),
                        filename="page_1_10_20_30_40.png",
                        discovery_source="layout-detector",
                    )
                ],
                # Page 2 intentionally omitted: detector found nothing.
            },
        )

        # Page 1 detected ref still wins.
        assert "![Figure A](page_1_10_20_30_40.png)" in rewritten
        # Page 2 VLM ref must survive since we had no canonical detection there.
        assert "![Figure B](page_2_50_60_20_20.png)" in rewritten

    def test_extract_page_images_uses_anchor_bbox(self, tmp_path):
        img = Image.new("RGB", (100, 100), color="white")
        for x in range(5, 85):
            for y in range(5, 75):
                img.putpixel((x, y), (0, 0, 0))

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        report = PageReport(
            mediabox=BoundingBox(0, 0, 100, 100),
            text_elements=[],
            image_elements=[ImageElement("Im0", BoundingBox(5, 25, 85, 95))],
        )

        natural_text = "![Figure](page_1_10_10_10_10.png)"

        with patch("olmocr.pipeline.render_pdf_to_base64png", return_value=image_base64):
            with patch("olmocr.pipeline._pdf_report", return_value=report):
                extract_page_images(natural_text, str(tmp_path), "dummy.pdf")

        output_path = tmp_path / "page_1_10_10_10_10.png"
        assert output_path.exists()

        cropped = Image.open(output_path)
        assert cropped.size == (80, 70)

    def test_extract_page_images_anchor_bbox_expands_to_caption(self, tmp_path):
        img = Image.new("RGB", (140, 140), color="white")
        for x in range(18, 122):
            for y in range(16, 82):
                img.putpixel((x, y), (0, 0, 0))

        for x in range(34, 108):
            for y in range(94, 101):
                img.putpixel((x, y), (0, 0, 0))

        for x in range(12, 128):
            for y in range(116, 122):
                img.putpixel((x, y), (0, 0, 0))

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        report = PageReport(
            mediabox=BoundingBox(0, 0, 140, 140),
            text_elements=[],
            image_elements=[ImageElement("Im0", BoundingBox(18, 58, 122, 124))],
        )

        natural_text = "![Figure](page_1_30_20_30_30.png)"

        with patch("olmocr.pipeline.render_pdf_to_base64png", return_value=image_base64):
            with patch("olmocr.pipeline._pdf_report", return_value=report):
                extract_page_images(natural_text, str(tmp_path), "dummy.pdf")

        output_path = tmp_path / "page_1_30_20_30_30.png"
        assert output_path.exists()

        cropped = Image.open(output_path)
        assert cropped.size == (104, 85)

    def test_extract_page_images_prefers_layout_detector_on_scanned_pages(self, tmp_path):
        img = Image.new("RGB", (100, 100), color="white")
        for x in range(20, 50):
            for y in range(30, 60):
                img.putpixel((x, y), (0, 0, 0))

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        scanned_report = PageReport(
            mediabox=BoundingBox(0, 0, 100, 100),
            text_elements=[],
            image_elements=[ImageElement("Scan", BoundingBox(0, 0, 100, 100))],
        )

        class FakeDetector:
            def detect(self, image):
                return [LayoutDetection(label="picture", score=0.95, box=(20, 30, 50, 60))]

        natural_text = "![Figure](page_1_20_30_30_30.png)"

        with patch("olmocr.pipeline.render_pdf_to_base64png", return_value=image_base64):
            with patch("olmocr.pipeline._pdf_report", return_value=scanned_report):
                with patch("olmocr.pipeline.get_figure_layout_detector", return_value=FakeDetector()):
                    extract_page_images(natural_text, str(tmp_path), "dummy.pdf", layout_model_name="mock-layout")

        output_path = tmp_path / "page_1_20_30_30_30.png"
        assert output_path.exists()

        cropped = Image.open(output_path)
        assert cropped.size == (30, 30)

    def test_extract_page_images_refines_coarse_layout_detector_box(self, tmp_path):
        img = Image.new("RGB", (160, 140), color="white")

        # Caption/text line above the diagram.
        for x in range(10, 150):
            for y in range(18, 24):
                img.putpixel((x, y), (0, 0, 0))

        # A disconnected diagram cluster lower on the page.
        for x in range(38, 72):
            for y in range(62, 90):
                img.putpixel((x, y), (0, 0, 0))
        for x in range(94, 126):
            for y in range(62, 90):
                img.putpixel((x, y), (0, 0, 0))
        for x in range(70, 96):
            for y in range(74, 78):
                img.putpixel((x, y), (0, 0, 0))

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        scanned_report = PageReport(
            mediabox=BoundingBox(0, 0, 160, 140),
            text_elements=[],
            image_elements=[ImageElement("Scan", BoundingBox(0, 0, 160, 140))],
        )

        class FakeDetector:
            def detect(self, image):
                return [LayoutDetection(label="picture", score=0.9, box=(10, 10, 150, 96))]

        natural_text = "![Figure](page_1_46_60_76_34.png)"

        with patch("olmocr.pipeline.render_pdf_to_base64png", return_value=image_base64):
            with patch("olmocr.pipeline._pdf_report", return_value=scanned_report):
                with patch("olmocr.pipeline.get_figure_layout_detector", return_value=FakeDetector()):
                    extract_page_images(natural_text, str(tmp_path), "dummy.pdf", layout_model_name="mock-layout")

        output_path = tmp_path / "page_1_46_60_76_34.png"
        assert output_path.exists()

        cropped = Image.open(output_path)
        assert cropped.size[1] < 60
        assert cropped.size[1] > 20

    def test_extract_page_images_refines_coarse_layout_detector_box_with_caption_but_without_body_text(self, tmp_path):
        img = Image.new("RGB", (220, 220), color="white")

        # Small page header near the top-right.
        for x in range(150, 205):
            for y in range(8, 18):
                img.putpixel((x, y), (0, 0, 0))

        # Main diagram content.
        for x in range(40, 180):
            for y in range(48, 54):
                img.putpixel((x, y), (0, 0, 0))
        for x in range(55, 65):
            for y in range(55, 120):
                img.putpixel((x, y), (0, 0, 0))
        for x in range(150, 160):
            for y in range(55, 120):
                img.putpixel((x, y), (0, 0, 0))
        for x in range(75, 145):
            for y in range(78, 88):
                img.putpixel((x, y), (0, 0, 0))
        for x in range(82, 138):
            for y in range(98, 108):
                img.putpixel((x, y), (0, 0, 0))

        # Caption line below the figure.
        for x in range(62, 160):
            for y in range(132, 139):
                img.putpixel((x, y), (0, 0, 0))

        # Body text block lower on the page.
        for line_y in (156, 168, 180, 192):
            for x in range(18, 205):
                for y in range(line_y, line_y + 5):
                    img.putpixel((x, y), (0, 0, 0))

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        scanned_report = PageReport(
            mediabox=BoundingBox(0, 0, 220, 220),
            text_elements=[],
            image_elements=[ImageElement("Scan", BoundingBox(0, 0, 220, 220))],
        )

        class FakeDetector:
            def detect(self, image):
                return [LayoutDetection(label="picture", score=0.93, box=(10, 0, 210, 210))]

        natural_text = "![Figure](page_1_10_0_200_210.png)"

        with patch("olmocr.pipeline.render_pdf_to_base64png", return_value=image_base64):
            with patch("olmocr.pipeline._pdf_report", return_value=scanned_report):
                with patch("olmocr.pipeline.get_figure_layout_detector", return_value=FakeDetector()):
                    extract_page_images(natural_text, str(tmp_path), "dummy.pdf", layout_model_name="mock-layout")

        output_path = tmp_path / "page_1_10_0_200_210.png"
        assert output_path.exists()

        cropped = Image.open(output_path)
        assert cropped.size[0] < 170
        assert cropped.size[1] < 100
        assert cropped.size[1] > 80

    def test_extract_page_images_avoids_single_axis_collapse_for_multi_panel_figure(self, tmp_path):
        img = Image.new("RGB", (220, 160), color="white")

        for x in range(42, 82):
            for y in range(42, 104):
                img.putpixel((x, y), (0, 0, 0))

        for x in range(128, 168):
            for y in range(42, 104):
                img.putpixel((x, y), (0, 0, 0))

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        scanned_report = PageReport(
            mediabox=BoundingBox(0, 0, 220, 160),
            text_elements=[],
            image_elements=[ImageElement("Scan", BoundingBox(0, 0, 220, 160))],
        )

        class FakeDetector:
            def detect(self, image):
                return [LayoutDetection(label="picture", score=0.94, box=(30, 30, 180, 110))]

        natural_text = "![Figure](page_1_30_30_150_80.png)"

        with patch("olmocr.pipeline.render_pdf_to_base64png", return_value=image_base64):
            with patch("olmocr.pipeline._pdf_report", return_value=scanned_report):
                with patch("olmocr.pipeline.get_figure_layout_detector", return_value=FakeDetector()):
                    extract_page_images(natural_text, str(tmp_path), "dummy.pdf", layout_model_name="mock-layout")

        output_path = tmp_path / "page_1_30_30_150_80.png"
        assert output_path.exists()

        cropped = Image.open(output_path)
        # Both horizontally-separated panels (x 42-82 and x 128-168) must be included.
        # The union spans 168-42 = 126 px wide; allow a few pixels of rounding/margin.
        assert cropped.size[0] >= 120
        assert cropped.size[1] >= 55

    def test_extract_page_images_avoids_top_truncation_for_stacked_diagram(self, tmp_path):
        """Top label box separated from main diagram body (e.g. DBMS architecture figure)."""
        img = Image.new("RGB", (300, 280), color="white")

        # Top section: wide box at top of figure (Query Evaluation Engine)
        for x in range(40, 220):
            for y in range(30, 90):
                img.putpixel((x, y), (0, 0, 0))

        # Bottom section: wide box below a whitespace gap (DBMS body)
        for x in range(40, 220):
            for y in range(130, 230):
                img.putpixel((x, y), (0, 0, 0))

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        scanned_report = PageReport(
            mediabox=BoundingBox(0, 0, 300, 280),
            text_elements=[],
            image_elements=[ImageElement("Scan", BoundingBox(0, 0, 300, 280))],
        )

        # VLM correctly spans both sections; layout detector does the same
        class FakeDetector:
            def detect(self, image):
                return [LayoutDetection(label="picture", score=0.95, box=(30, 20, 230, 240))]

        natural_text = "![Figure](page_1_30_20_200_220.png)"

        with patch("olmocr.pipeline.render_pdf_to_base64png", return_value=image_base64):
            with patch("olmocr.pipeline._pdf_report", return_value=scanned_report):
                with patch("olmocr.pipeline.get_figure_layout_detector", return_value=FakeDetector()):
                    extract_page_images(natural_text, str(tmp_path), "dummy.pdf", layout_model_name="mock-layout")

        output_path = tmp_path / "page_1_30_20_200_220.png"
        assert output_path.exists()

        cropped = Image.open(output_path)
        # Must cover from top section (y≈30) to bottom section (y≈230): height ≥ 170 px
        assert cropped.size[1] >= 170, f"Top section cropped off; got height {cropped.size[1]}"

    def test_detect_missing_figure_refs_finds_layout_figure_without_vlm_ref(self):
        img = Image.new("RGB", (120, 120), color="white")
        for x in range(30, 90):
            for y in range(40, 95):
                img.putpixel((x, y), (0, 0, 0))

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        scanned_report = PageReport(
            mediabox=BoundingBox(0, 0, 120, 120),
            text_elements=[],
            image_elements=[ImageElement("Scan", BoundingBox(0, 0, 120, 120))],
        )

        class FakeDetector:
            def detect(self, image):
                return [LayoutDetection(label="picture", score=0.96, box=(30, 40, 90, 95))]

        with patch("olmocr.pipeline.render_pdf_to_base64png", return_value=image_base64):
            with patch("olmocr.pipeline._pdf_report", return_value=scanned_report):
                with patch("olmocr.pipeline.get_figure_layout_detector", return_value=FakeDetector()):
                    detected = detect_missing_figure_refs("plain page text", "dummy.pdf", page_spans=[[0, 15, 1]], layout_model_name="mock-layout")

        assert 1 in detected
        assert detected[1] == ["page_1_30_40_60_55.png"]

    def test_detect_page_figure_refs_uses_detector_output_as_canonical_list(self):
        img = Image.new("RGB", (120, 120), color="white")
        for x in range(32, 88):
            for y in range(42, 92):
                img.putpixel((x, y), (0, 0, 0))

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        scanned_report = PageReport(
            mediabox=BoundingBox(0, 0, 120, 120),
            text_elements=[],
            image_elements=[ImageElement("Scan", BoundingBox(0, 0, 120, 120))],
        )

        class FakeDetector:
            def detect(self, image):
                return [LayoutDetection(label="picture", score=0.97, box=(32, 42, 88, 92))]

        natural_text = "Body text ![Wrong Box](page_1_2_2_12_10.png)"

        with patch("olmocr.pipeline.render_pdf_to_base64png", return_value=image_base64):
            with patch("olmocr.pipeline._pdf_report", return_value=scanned_report):
                with patch("olmocr.pipeline.get_figure_layout_detector", return_value=FakeDetector()):
                    detected = detect_page_figure_refs(natural_text, "dummy.pdf", page_spans=[[0, len(natural_text), 1]], layout_model_name="mock-layout")

        assert 1 in detected
        assert [ref.filename for ref in detected[1]] == ["page_1_32_42_56_50.png"]
        assert detected[1][0].discovery_source == "layout-detector-refined"

    def test_detect_page_figure_refs_filters_text_line_fragments(self):
        img = Image.new("RGB", (160, 160), color="white")
        for x in range(18, 142):
            for y in range(28, 36):
                img.putpixel((x, y), (0, 0, 0))

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        scanned_report = PageReport(
            mediabox=BoundingBox(0, 0, 160, 160),
            text_elements=[],
            image_elements=[ImageElement("Scan", BoundingBox(0, 0, 160, 160))],
        )

        class FakeDetector:
            def detect(self, image):
                return [LayoutDetection(label="picture", score=0.9, box=(18, 28, 142, 36))]

        with patch("olmocr.pipeline.render_pdf_to_base64png", return_value=image_base64):
            with patch("olmocr.pipeline._pdf_report", return_value=scanned_report):
                with patch("olmocr.pipeline.get_figure_layout_detector", return_value=FakeDetector()):
                    detected = detect_page_figure_refs("plain page text", "dummy.pdf", page_spans=[[0, 15, 1]], layout_model_name="mock-layout")

        assert detected == {}

    def test_detect_page_figure_refs_falls_back_to_page_components_when_detector_missing(self):
        img = Image.new("RGB", (220, 260), color="white")

        # First figure near the top.
        for x in range(55, 165):
            for y in range(32, 38):
                img.putpixel((x, y), (0, 0, 0))
        for x in range(72, 82):
            for y in range(40, 98):
                img.putpixel((x, y), (0, 0, 0))
        for x in range(138, 148):
            for y in range(40, 98):
                img.putpixel((x, y), (0, 0, 0))
        for x in range(92, 128):
            for y in range(64, 74):
                img.putpixel((x, y), (0, 0, 0))

        # Caption under the first figure.
        for x in range(78, 150):
            for y in range(110, 116):
                img.putpixel((x, y), (0, 0, 0))

        # Second figure lower on the page.
        for x in range(44, 178):
            for y in range(150, 156):
                img.putpixel((x, y), (0, 0, 0))
        for x in range(58, 68):
            for y in range(158, 220):
                img.putpixel((x, y), (0, 0, 0))
        for x in range(154, 164):
            for y in range(158, 220):
                img.putpixel((x, y), (0, 0, 0))
        for x in range(86, 138):
            for y in range(182, 192):
                img.putpixel((x, y), (0, 0, 0))

        # Body text below both figures.
        for line_y in (232, 242):
            for x in range(15, 205):
                for y in range(line_y, line_y + 4):
                    img.putpixel((x, y), (0, 0, 0))

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        scanned_report = PageReport(
            mediabox=BoundingBox(0, 0, 220, 260),
            text_elements=[],
            image_elements=[ImageElement("Scan", BoundingBox(0, 0, 220, 260))],
        )

        with patch("olmocr.pipeline.render_pdf_to_base64png", return_value=image_base64):
            with patch("olmocr.pipeline._pdf_report", return_value=scanned_report):
                with patch("olmocr.pipeline.get_figure_layout_detector", return_value=None):
                    detected = detect_page_figure_refs(
                        "Figure 2.1  The Employees Entity Set\n\nFigure 2.2  The Works_In Relationship Set",
                        "dummy.pdf",
                        page_spans=[[0, 79, 1]],
                        layout_model_name="mock-layout",
                    )

        assert 1 in detected
        assert len(detected[1]) == 2
        assert all(ref.discovery_source == "page-components" for ref in detected[1])

    def test_detect_page_figure_refs_does_not_use_page_components_without_figure_mentions(self):
        img = Image.new("RGB", (220, 260), color="white")
        for x in range(55, 165):
            for y in range(32, 38):
                img.putpixel((x, y), (0, 0, 0))

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        scanned_report = PageReport(
            mediabox=BoundingBox(0, 0, 220, 260),
            text_elements=[],
            image_elements=[ImageElement("Scan", BoundingBox(0, 0, 220, 260))],
        )

        with patch("olmocr.pipeline.render_pdf_to_base64png", return_value=image_base64):
            with patch("olmocr.pipeline._pdf_report", return_value=scanned_report):
                with patch("olmocr.pipeline.get_figure_layout_detector", return_value=None):
                    detected = detect_page_figure_refs("plain page text", "dummy.pdf", page_spans=[[0, 15, 1]], layout_model_name="mock-layout")

        assert detected == {}

    def test_detect_page_figure_refs_counts_unique_figure_mentions_for_page_components(self):
        img = Image.new("RGB", (220, 260), color="white")

        for x in range(55, 165):
            for y in range(32, 38):
                img.putpixel((x, y), (0, 0, 0))
        for x in range(72, 82):
            for y in range(40, 98):
                img.putpixel((x, y), (0, 0, 0))
        for x in range(138, 148):
            for y in range(40, 98):
                img.putpixel((x, y), (0, 0, 0))

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        scanned_report = PageReport(
            mediabox=BoundingBox(0, 0, 220, 260),
            text_elements=[],
            image_elements=[ImageElement("Scan", BoundingBox(0, 0, 220, 260))],
        )

        repeated_mention_text = (
            "Figure 2.1 The Employees Entity Set\n\n"
            "As shown in Figure 2.1, the entity set has three attributes.\n\n"
            "Figure 2.1 appears again in the discussion."
        )

        with patch("olmocr.pipeline.render_pdf_to_base64png", return_value=image_base64):
            with patch("olmocr.pipeline._pdf_report", return_value=scanned_report):
                with patch("olmocr.pipeline.get_figure_layout_detector", return_value=None):
                    detected = detect_page_figure_refs(
                        repeated_mention_text,
                        "dummy.pdf",
                        page_spans=[[0, len(repeated_mention_text), 1]],
                        layout_model_name="mock-layout",
                    )

        assert 1 in detected
        assert len(detected[1]) == 1
        assert detected[1][0].discovery_source == "page-components"

    def test_detect_page_figure_refs_page_components_rejects_small_word_sized_candidates(self):
        img = Image.new("RGB", (220, 260), color="white")

        # A real figure candidate.
        for x in range(55, 165):
            for y in range(32, 38):
                img.putpixel((x, y), (0, 0, 0))
        for x in range(72, 82):
            for y in range(40, 98):
                img.putpixel((x, y), (0, 0, 0))
        for x in range(138, 148):
            for y in range(40, 98):
                img.putpixel((x, y), (0, 0, 0))

        # A word-sized text fragment lower on the page.
        for x in range(140, 190):
            for y in range(170, 178):
                img.putpixel((x, y), (0, 0, 0))
        for x in range(144, 148):
            for y in range(170, 190):
                img.putpixel((x, y), (0, 0, 0))
        for x in range(156, 160):
            for y in range(170, 188):
                img.putpixel((x, y), (0, 0, 0))
        for x in range(168, 172):
            for y in range(170, 190):
                img.putpixel((x, y), (0, 0, 0))

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        scanned_report = PageReport(
            mediabox=BoundingBox(0, 0, 220, 260),
            text_elements=[],
            image_elements=[ImageElement("Scan", BoundingBox(0, 0, 220, 260))],
        )

        with patch("olmocr.pipeline.render_pdf_to_base64png", return_value=image_base64):
            with patch("olmocr.pipeline._pdf_report", return_value=scanned_report):
                with patch("olmocr.pipeline.get_figure_layout_detector", return_value=None):
                    detected = detect_page_figure_refs(
                        "Figure 2.1 The Employees Entity Set\n\nFigure 2.2 Another Figure",
                        "dummy.pdf",
                        page_spans=[[0, 65, 1]],
                        layout_model_name="mock-layout",
                    )

        assert 1 in detected
        assert len(detected[1]) == 1
        assert detected[1][0].discovery_source == "page-components"

    def test_extract_page_images_saves_auto_detected_figure_without_vlm_ref(self, tmp_path):
        img = Image.new("RGB", (120, 120), color="white")
        for x in range(30, 90):
            for y in range(40, 95):
                img.putpixel((x, y), (0, 0, 0))

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        scanned_report = PageReport(
            mediabox=BoundingBox(0, 0, 120, 120),
            text_elements=[],
            image_elements=[ImageElement("Scan", BoundingBox(0, 0, 120, 120))],
        )

        class FakeDetector:
            def detect(self, image):
                return [LayoutDetection(label="picture", score=0.96, box=(30, 40, 90, 95))]

        with patch("olmocr.pipeline.render_pdf_to_base64png", return_value=image_base64):
            with patch("olmocr.pipeline._pdf_report", return_value=scanned_report):
                with patch("olmocr.pipeline.get_figure_layout_detector", return_value=FakeDetector()):
                    extract_page_images("plain page text", str(tmp_path), "dummy.pdf", page_spans=[[0, 15, 1]], layout_model_name="mock-layout")

        output_path = tmp_path / "page_1_30_40_60_55.png"
        assert output_path.exists()

    def test_augment_markdown_with_detected_refs_appends_missing_refs_by_page(self):
        text = "First page text\n\nSecond page text"
        page_spans = [[0, 15, 1], [15, len(text), 2]]
        augmented = _augment_markdown_with_detected_refs(text, page_spans, {2: ["page_2_10_20_30_40.png"]})

        assert "![Figure](page_2_10_20_30_40.png)" in augmented
        assert augmented.endswith("![Figure](page_2_10_20_30_40.png)")

    def test_rewrite_markdown_with_detected_refs_replaces_vlm_refs_with_canonical_refs(self):
        text = "Page text ![An Instance of Works_In3](page_1_2_2_12_10.png)"
        rewritten = _rewrite_markdown_with_detected_refs(
            text,
            [[0, len(text), 1]],
            {
                1: [
                    DetectedFigureRef(
                        page_num=1,
                        box=(2, 2, 14, 12),
                        filename="page_1_2_2_12_10.png",
                        discovery_source="layout-detector",
                    )
                ]
            },
        )

        assert "![An Instance of Works\\_In3](page_1_2_2_12_10.png)" in rewritten

    def test_rewrite_markdown_with_detected_refs_preserves_existing_alt_text_escapes(self):
        text = "Page text ![An Instance of Works\\_In3](page_1_2_2_12_10.png)"
        rewritten = _rewrite_markdown_with_detected_refs(
            text,
            [[0, len(text), 1]],
            {
                1: [
                    DetectedFigureRef(
                        page_num=1,
                        box=(2, 2, 14, 12),
                        filename="page_1_2_2_12_10.png",
                        discovery_source="layout-detector",
                    )
                ]
            },
        )

        assert "![An Instance of Works\\_In3](page_1_2_2_12_10.png)" in rewritten


class TestJunkFigureFiltering:
    def test_strip_junk_figure_refs_removes_tagged_refs(self):
        markdown = "Some text.\n\n![Figure 1](assets/page_1_0_0_100_200.png)\n\nMore text."
        cleaned = _strip_junk_figure_refs(markdown, {"page_1_0_0_100_200.png"})
        assert "page_1_0_0_100_200.png" not in cleaned
        assert "Some text." in cleaned
        assert "More text." in cleaned

    def test_strip_junk_figure_refs_leaves_non_junk_intact(self):
        markdown = "![Figure 1](assets/page_1_0_0_100_200.png)\n\n![Figure 2](assets/page_2_0_0_80_60.png)"
        cleaned = _strip_junk_figure_refs(markdown, {"page_1_0_0_100_200.png"})
        assert "page_1_0_0_100_200.png" not in cleaned
        assert "page_2_0_0_80_60.png" in cleaned

    def test_extract_page_images_skips_junk_when_vlm_says_no(self, tmp_path):
        """When the VLM verifier returns False, the crop is treated as junk."""
        img = Image.new("RGB", (400, 600), "white")
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        scanned_report = PageReport(
            mediabox=BoundingBox(0, 0, 400, 600),
            text_elements=[],
            image_elements=[ImageElement("Scan", BoundingBox(0, 0, 400, 600))],
        )

        natural_text = "![Figure](page_1_0_0_400_600.png)"

        with patch("olmocr.pipeline.render_pdf_to_base64png", return_value=image_base64):
            with patch("olmocr.pipeline._pdf_report", return_value=scanned_report):
                with patch("olmocr.pipeline.get_figure_layout_detector", return_value=None):
                    with patch("olmocr.pipeline._vlm_verify_is_figure", return_value=False):
                        junk = extract_page_images(
                            natural_text,
                            str(tmp_path),
                            "dummy.pdf",
                            vlm_verify_server="http://fake",
                        )

        assert "page_1_0_0_400_600.png" in junk, "Junk filename must be returned"
        assert not (tmp_path / "page_1_0_0_400_600.png").exists(), "Junk PNG must not be written to disk"

    def test_extract_page_images_keeps_crop_when_vlm_says_yes(self, tmp_path):
        """When the VLM verifier returns True, the crop is written to disk and not junk."""
        img = Image.new("RGB", (400, 600), "white")
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        scanned_report = PageReport(
            mediabox=BoundingBox(0, 0, 400, 600),
            text_elements=[],
            image_elements=[ImageElement("Scan", BoundingBox(0, 0, 400, 600))],
        )

        natural_text = "![Figure](page_1_0_0_400_600.png)"

        with patch("olmocr.pipeline.render_pdf_to_base64png", return_value=image_base64):
            with patch("olmocr.pipeline._pdf_report", return_value=scanned_report):
                with patch("olmocr.pipeline.get_figure_layout_detector", return_value=None):
                    with patch("olmocr.pipeline._vlm_verify_is_figure", return_value=True):
                        junk = extract_page_images(
                            natural_text,
                            str(tmp_path),
                            "dummy.pdf",
                            vlm_verify_server="http://fake",
                        )

        assert junk == set(), "No filenames should be reported as junk"
        assert (tmp_path / "page_1_0_0_400_600.png").exists(), "PNG must be written to disk"

    def test_vlm_verify_is_figure_no_server_returns_true(self):
        """With no server configured, the verifier must fail-open (keep the figure)."""
        img = Image.new("RGB", (100, 100), "white")
        assert _vlm_verify_is_figure(img, server=None, model="olmocr") is True

    def test_vlm_verify_is_figure_parses_no_response(self):
        """A response starting with 'no' must be treated as not-a-figure."""
        img = Image.new("RGB", (100, 100), "white")
        fake_resp = type(
            "R",
            (),
            {
                "raise_for_status": lambda self: None,
                "json": lambda self: {"choices": [{"message": {"content": "No, this is body text."}}]},
            },
        )()
        with patch("olmocr.pipeline.httpx.post", return_value=fake_resp):
            assert _vlm_verify_is_figure(img, server="http://fake", model="olmocr") is False

    def test_vlm_verify_is_figure_network_error_keeps_figure(self):
        """A network error must fail-open and keep the figure."""
        img = Image.new("RGB", (100, 100), "white")
        with patch("olmocr.pipeline.httpx.post", side_effect=Exception("boom")):
            assert _vlm_verify_is_figure(img, server="http://fake", model="olmocr") is True


class TestPromptContract:
    def test_extract_page_images_falls_back_locally_when_scanned_detector_unavailable(self, tmp_path):
        img = Image.new("RGB", (100, 100), color="white")
        for x in range(18, 52):
            for y in range(28, 62):
                img.putpixel((x, y), (0, 0, 0))

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        scanned_report = PageReport(
            mediabox=BoundingBox(0, 0, 100, 100),
            text_elements=[],
            image_elements=[ImageElement("Scan", BoundingBox(0, 0, 100, 100))],
        )

        natural_text = "![Figure](page_1_20_30_30_30.png)"

        with patch("olmocr.pipeline.render_pdf_to_base64png", return_value=image_base64):
            with patch("olmocr.pipeline._pdf_report", return_value=scanned_report):
                with patch("olmocr.pipeline.get_figure_layout_detector", return_value=None):
                    extract_page_images(natural_text, str(tmp_path), "dummy.pdf", layout_model_name="none")

        output_path = tmp_path / "page_1_20_30_30_30.png"
        assert output_path.exists()

        cropped = Image.open(output_path)
        assert cropped.size[0] < 100
        assert cropped.size[1] < 100
        assert cropped.size[0] >= 30
        assert cropped.size[1] >= 30
