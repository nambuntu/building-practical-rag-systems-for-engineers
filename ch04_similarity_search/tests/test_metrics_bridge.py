import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from metrics_bridge import recall_at_k, reciprocal_rank  # noqa: E402


def test_recall_at_k_hit_and_miss():
    assert recall_at_k("doc_1", ["doc_3", "doc_1"]) == 1.0
    assert recall_at_k("doc_1", ["doc_3", "doc_4"]) == 0.0


def test_reciprocal_rank_behaviour():
    assert reciprocal_rank("doc_1", ["doc_1", "doc_2"]) == 1.0
    assert reciprocal_rank("doc_1", ["doc_2", "doc_1", "doc_3"]) == 0.5
    assert reciprocal_rank("doc_1", ["doc_2", "doc_3"]) == 0.0
