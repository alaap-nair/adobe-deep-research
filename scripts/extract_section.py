"""
extract_section.py -- Extract a specific section from the OpenStax Biology 2e PDF.

Usage:
    python scripts/extract_section.py data/biology2e.pdf "7.2" -o data/biology_7_2.txt

Searches for section headings like "7.2 Glycolysis" and extracts text until the
next section heading (e.g. "7.3") or chapter boundary.
"""

import argparse
import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from parse_pdf import parse_with_pymupdf


def find_section(full_text: str, section_num: str) -> str:
    """Extract a section from full text by section number (e.g. '7.2').

    Looks for a heading like '7.2 Glycolysis' followed by body content
    (not a TOC entry) and captures text until the next section heading.
    """
    major, minor = section_num.split(".")
    # Match the section heading -- require it to be followed by body content
    # (e.g. "LEARNING OBJECTIVES" or a paragraph), not a page number (TOC entry)
    heading_pattern = re.compile(
        rf"^{re.escape(major)}\.{re.escape(minor)}\s+[A-Z]",
        re.MULTILINE,
    )

    # Next section pattern: same major number with next minor, or next major number
    next_minor = int(minor) + 1
    next_major = int(major) + 1
    end_pattern = re.compile(
        rf"^(?:{re.escape(major)}\.{next_minor}\s+[A-Z]|{next_major}\.\d+\s+[A-Z]|Chapter\s+{next_major}\b)",
        re.MULTILINE,
    )

    # Find the match followed by "LEARNING OBJECTIVES" (actual section body)
    start_match = None
    for m in heading_pattern.finditer(full_text):
        after = full_text[m.start():m.start() + 500]
        if "LEARNING OBJECTIVES" in after:
            start_match = m
            break

    if not start_match:
        # Fallback: find the match followed by substantial body content (not TOC)
        for m in heading_pattern.finditer(full_text):
            after = full_text[m.start():m.start() + 500]
            lines = after.split("\n")
            if len(lines) >= 5 and not re.match(r"^\d+$", lines[1].strip()):
                start_match = m
                break

    if not start_match:
        raise ValueError(f"Could not find section {section_num} body content in the document.")

    start_pos = start_match.start()

    # Find the end: next section heading with body content (skip TOC/summary mentions)
    end_pos = len(full_text)
    for m in end_pattern.finditer(full_text, start_pos + 100):
        after = full_text[m.start():m.start() + 500]
        lines = after.split("\n")
        if len(lines) >= 3 and not re.match(r"^\d+$", lines[1].strip()):
            end_pos = m.start()
            break

    section_text = full_text[start_pos:end_pos].strip()
    return section_text


def main():
    parser = argparse.ArgumentParser(description="Extract a section from a PDF textbook.")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("section", help="Section number to extract (e.g. '7.2')")
    parser.add_argument("-o", "--output", help="Output text file path", default=None)
    args = parser.parse_args()

    print(f"Parsing PDF: {args.pdf_path}")
    full_text = parse_with_pymupdf(args.pdf_path)
    print(f"Full document: {len(full_text)} characters")

    print(f"Searching for section {args.section}...")
    section_text = find_section(full_text, args.section)
    print(f"Extracted section {args.section}: {len(section_text)} characters")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(section_text)
        print(f"Saved to {args.output}")
    else:
        print("\n" + "=" * 60)
        print(section_text[:3000])
        if len(section_text) > 3000:
            print(f"\n... ({len(section_text) - 3000} more characters)")


if __name__ == "__main__":
    main()
