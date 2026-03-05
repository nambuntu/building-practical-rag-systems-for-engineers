import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chunking import fixed_token_chunks, semantic_chunks, split_sentences  # noqa: E402


def test_fixed_chunker_non_empty_and_overlap_math():
    text = "one two three four five six seven eight nine ten eleven twelve"
    chunks = fixed_token_chunks(text=text, chunk_size=5, overlap=2)

    assert chunks
    for left, right in zip(chunks, chunks[1:]):
        assert left.token_end - right.token_start == 2


def test_semantic_chunker_keeps_sentences_intact():
    text = "Alpha starts here. Bravo continues there. Charlie ends now."
    chunks = semantic_chunks(text=text, target_size=2)
    sentences = split_sentences(text)

    assert len(chunks) == 3
    for sentence, chunk in zip(sentences, chunks):
        assert sentence == chunk.text
