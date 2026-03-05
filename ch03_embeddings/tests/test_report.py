import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import report  # noqa: E402


def test_render_matrix_contains_headers_and_values():
    labels = ["a", "b"]
    matrix = [[1.0, 0.2], [0.2, 1.0]]
    rendered = report.render_matrix(labels, matrix)
    assert "Similarity Matrix" in rendered
    assert "a" in rendered and "b" in rendered
    assert "1.00" in rendered


def test_interpret_reports_highest_and_lowest_pairs():
    labels = ["db", "banana", "server"]
    matrix = [
        [1.0, 0.1, 0.8],
        [0.1, 1.0, 0.2],
        [0.8, 0.2, 1.0],
    ]
    text = report.interpret(labels, matrix)
    assert "Highest similarity pair" in text
    assert "Lowest similarity pair" in text
    assert "db" in text
