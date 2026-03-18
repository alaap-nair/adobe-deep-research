import re


def chunk_text(text: str, max_chars: int = 1500, overlap: int = 200) -> list[str]:
    """
    Simple chunker for textbook text.
    Splits on paragraph boundaries where possible, then packs into max_chars windows with overlap.
    """
    if not text:
        return []
    text = text.strip()
    if not text:
        return []

    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks: list[str] = []

    buf = ""
    for p in paras:
        if not buf:
            buf = p
            continue
        if len(buf) + 2 + len(p) <= max_chars:
            buf = f"{buf}\n\n{p}"
        else:
            chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)

    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    # character-level overlap at chunk boundaries
    out: list[str] = []
    prev_tail = ""
    for c in chunks:
        c2 = c
        if prev_tail:
            c2 = f"{prev_tail}{c}"
        out.append(c2[: max_chars].strip())
        prev_tail = c[-overlap:] if len(c) > overlap else c
    return out

