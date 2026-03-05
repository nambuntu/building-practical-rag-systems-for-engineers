import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import similarity  # noqa: E402


def test_cosine_similarity_identical_vectors():
    value = similarity.cosine_similarity([1.0, 2.0], [1.0, 2.0])
    assert math.isclose(value, 1.0, rel_tol=1e-9)


def test_cosine_similarity_negative_relation():
    value = similarity.cosine_similarity([1.0, 0.0], [-1.0, 0.0])
    assert math.isclose(value, -1.0, rel_tol=1e-9)


def test_cosine_similarity_dimension_mismatch_raises():
    try:
        similarity.cosine_similarity([1.0, 2.0], [1.0])
    except ValueError as exc:
        assert "dimension mismatch" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError for mismatched dimensions")


def test_build_matrix_diagonal_near_one():
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    matrix = similarity.build_matrix(vectors)
    assert math.isclose(matrix[0][0], 1.0, rel_tol=1e-9)
    assert math.isclose(matrix[1][1], 1.0, rel_tol=1e-9)
