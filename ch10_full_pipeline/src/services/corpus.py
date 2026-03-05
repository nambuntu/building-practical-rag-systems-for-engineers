from collections.abc import Iterable


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> Iterable[str]:
    words = text.split()
    if not words:
        return

    step = max(1, chunk_size - chunk_overlap)
    for start in range(0, len(words), step):
        end = start + chunk_size
        piece = words[start:end]
        if not piece:
            break
        yield " ".join(piece)
        if end >= len(words):
            break
