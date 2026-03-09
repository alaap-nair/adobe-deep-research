"""
parse_pdf.py -- PDF text extraction with multiple backends.

Backends:
  - pymupdf (default): Local extraction via PyMuPDF. Fast, no API key needed.
  - llamaparse: Cloud extraction via LlamaParse. Better for complex layouts.
    Requires LLAMA_CLOUD_API_KEY in .env.
"""

import os


def parse_pdf(pdf_path: str, backend: str = "pymupdf") -> str:
    """Extract text from a PDF file.

    Args:
        pdf_path: Path to the PDF file.
        backend: "pymupdf" (default) or "llamaparse".

    Returns:
        Extracted text as a string.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if backend == "pymupdf":
        return parse_with_pymupdf(pdf_path)
    elif backend == "llamaparse":
        return parse_with_llamaparse(pdf_path)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'pymupdf' or 'llamaparse'.")


def parse_with_pymupdf(pdf_path: str) -> str:
    """Page-by-page text extraction via PyMuPDF."""
    import pymupdf

    doc = pymupdf.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if text.strip():
            pages.append(text)
    doc.close()
    return "\n\n".join(pages)


def parse_with_llamaparse(pdf_path: str) -> str:
    """Cloud API extraction via LlamaParse. Requires LLAMA_CLOUD_API_KEY in .env."""
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        raise ValueError("LLAMA_CLOUD_API_KEY not found in .env")

    from llama_parse import LlamaParse

    parser = LlamaParse(api_key=api_key, result_type="text")
    documents = parser.load_data(pdf_path)
    return "\n\n".join(doc.text for doc in documents)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/parse_pdf.py <path/to/file.pdf> [backend]")
        sys.exit(1)

    path = sys.argv[1]
    backend = sys.argv[2] if len(sys.argv) > 2 else "pymupdf"
    text = parse_pdf(path, backend)
    print(f"Extracted {len(text)} characters from {path} (backend: {backend})")
    print(text[:2000])
