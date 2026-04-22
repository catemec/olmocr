#!/usr/bin/env python3
"""
Extract figures referenced in olmocr markdown output as actual PNG files.

olmocr instructs the model to reference figures as:
  ![caption](page_startx_starty_width_height.png)

where coordinates are in the pixel space of the rendered page image.
This script renders each referenced PDF page, crops the region, and saves it
next to the markdown file so the image links resolve correctly.

Usage:
    python extract_markdown_images.py <markdown_file> <pdf_file> [--dim 2048]
    python extract_markdown_images.py --markdown-dir <dir> --pdf-dir <dir> [--dim 2048]
"""

import argparse
import base64
import io
import re
import sys
from pathlib import Path

from PIL import Image

from olmocr.data.renderpdf import render_pdf_to_base64png

IMAGE_REF_RE = re.compile(r"!\[([^\]]*)\]\((\d+_\d+_\d+_\d+_\d+\.png)\)")


def parse_image_filename(filename: str) -> tuple[int, int, int, int, int]:
    """Parse 'page_x_y_w_h.png' into (page, x, y, w, h)."""
    stem = Path(filename).stem
    parts = stem.split("_")
    page, x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
    return page, x, y, w, h


def render_page_image(pdf_path: str, page_num: int, dim: int) -> Image.Image:
    b64 = render_pdf_to_base64png(pdf_path, page_num, target_longest_image_dim=dim)
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def extract_images_from_markdown(md_path: Path, pdf_path: Path, dim: int) -> int:
    text = md_path.read_text()
    refs = IMAGE_REF_RE.findall(text)
    if not refs:
        return 0

    out_dir = md_path.parent
    page_cache: dict[int, Image.Image] = {}
    saved = 0

    for _alt, filename in refs:
        dest = out_dir / filename
        if dest.exists():
            continue

        page, x, y, w, h = parse_image_filename(filename)

        if page not in page_cache:
            print(f"  Rendering page {page} of {pdf_path.name}...")
            page_cache[page] = render_page_image(str(pdf_path), page, dim)

        img = page_cache[page]
        img_w, img_h = img.size

        # Clamp crop box to image bounds
        left = max(0, min(x, img_w))
        upper = max(0, min(y, img_h))
        right = max(0, min(x + w, img_w))
        lower = max(0, min(y + h, img_h))

        if right <= left or lower <= upper:
            print(f"  Warning: degenerate crop for {filename}, skipping")
            continue

        crop = img.crop((left, upper, right, lower))
        crop.save(dest, format="PNG")
        print(f"  Saved {dest.name}")
        saved += 1

    return saved


def find_pdf_for_markdown(md_path: Path, pdf_dir: Path) -> Path | None:
    """Find a PDF whose stem matches the markdown stem."""
    stem = md_path.stem
    for ext in (".pdf", ".PDF"):
        candidate = pdf_dir / (stem + ext)
        if candidate.exists():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description="Extract figure images from olmocr markdown output.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("markdown_file", nargs="?", help="Single markdown file to process")
    group.add_argument("--markdown-dir", type=Path, help="Directory of markdown files")

    parser.add_argument("pdf_file", nargs="?", help="PDF file (required when using markdown_file)")
    parser.add_argument("--pdf-dir", type=Path, help="Directory of PDF files (matched by stem to markdown files)")
    parser.add_argument("--dim", type=int, default=2048, help="Render dimension for PDF pages (default: 2048)")
    args = parser.parse_args()

    if args.markdown_file:
        md_path = Path(args.markdown_file)
        if not args.pdf_file:
            parser.error("pdf_file is required when processing a single markdown file")
        pdf_path = Path(args.pdf_file)
        if not md_path.exists():
            sys.exit(f"Markdown file not found: {md_path}")
        if not pdf_path.exists():
            sys.exit(f"PDF file not found: {pdf_path}")
        print(f"Processing {md_path.name}...")
        n = extract_images_from_markdown(md_path, pdf_path, args.dim)
        print(f"Done: {n} image(s) saved.")

    else:
        md_dir = args.markdown_dir
        pdf_dir = args.pdf_dir or md_dir
        md_files = sorted(md_dir.glob("**/*.md"))
        if not md_files:
            sys.exit(f"No markdown files found in {md_dir}")
        total = 0
        for md_path in md_files:
            pdf_path = find_pdf_for_markdown(md_path, pdf_dir)
            if pdf_path is None:
                print(f"No matching PDF for {md_path.name}, skipping")
                continue
            print(f"Processing {md_path.name}...")
            n = extract_images_from_markdown(md_path, pdf_path, args.dim)
            total += n
        print(f"Done: {total} image(s) saved across {len(md_files)} file(s).")


if __name__ == "__main__":
    main()
