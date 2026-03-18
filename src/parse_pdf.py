"""
PDF → text parsing for textbook ingestion.
Uses Mistral OCR 3 for fast, accurate extraction from textbook PDFs (multi-column, tables).
Requires `MISTRAL_API_KEY` in your environment (e.g. via `.env`).
"""

import os


def pdf_to_text(path, page_range=None, model=None):
    """
    Extract text from a PDF using Mistral OCR.

    Args:
        path: Path to the PDF file.
        page_range: Optional (start, end) 0-based page range, e.g. (0, 5) for first 6 pages.
                    None = all pages.
        model: Optional Mistral OCR model name. Defaults to env `MISTRAL_OCR_MODEL` or `mistral-ocr-latest`.

    Returns:
        A single string (markdown) with pages joined by newlines.
    """
    try:
        from mistralai import Mistral
    except ImportError as e:
        raise ImportError(
            "Mistral client is required for OCR parsing. Install with: pip install mistralai"
        ) from e

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Missing MISTRAL_API_KEY. Add it to your environment or `.env` before running OCR."
        )

    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PDF not found: {path}")

    if model is None:
        model = os.getenv("MISTRAL_OCR_MODEL", "mistral-ocr-latest")

    with Mistral(api_key=api_key) as mistral:
        uploaded = mistral.files.upload(
            file={"file_name": os.path.basename(path), "content": open(path, "rb")},
            purpose="ocr",
        )

        # OCR expects a Mistral file_id reference
        res = mistral.ocr.process(
            model=model,
            document={"file_id": uploaded.id},
        )

        pages = getattr(res, "pages", None) or res.get("pages", [])
        if not pages:
            return ""

        if page_range is not None:
            start, end = page_range
            start = max(0, start)
            end = max(start, end)
            pages = [p for p in pages if start <= (p.get("index", 0) - 1) < end]

        parts = []
        for p in pages:
            md = p.get("markdown", "")
            if md:
                parts.append(md)

        return "\n\n".join(parts).strip()
