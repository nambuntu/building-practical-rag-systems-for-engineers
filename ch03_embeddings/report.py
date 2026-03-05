from __future__ import annotations


def render_matrix(labels: list[str], matrix: list[list[float]]) -> str:
    if len(labels) != len(matrix):
        raise ValueError("Labels and matrix size must match.")

    col_width = max(max(len(label) for label in labels), 10)
    header = " " * (col_width + 2) + " ".join(f"{label:>{col_width}}" for label in labels)
    lines = ["Similarity Matrix (cosine)", "", header]

    for label, row in zip(labels, matrix):
        values = " ".join(f"{value:>{col_width}.2f}" for value in row)
        lines.append(f"{label:>{col_width}}  {values}")

    return "\n".join(lines)


def interpret(labels: list[str], matrix: list[list[float]]) -> str:
    if len(labels) != len(matrix):
        raise ValueError("Labels and matrix size must match.")
    if len(labels) < 2:
        raise ValueError("Need at least two labels for interpretation.")

    highest = None
    lowest = None

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            value = matrix[i][j]
            pair = (labels[i], labels[j], value)
            if highest is None or value > highest[2]:
                highest = pair
            if lowest is None or value < lowest[2]:
                lowest = pair

    assert highest is not None and lowest is not None
    return (
        "Interpretation\n"
        f"- Highest similarity pair: {highest[0]} <-> {highest[1]} ({highest[2]:.2f})\n"
        f"- Lowest similarity pair: {lowest[0]} <-> {lowest[1]} ({lowest[2]:.2f})"
    )
