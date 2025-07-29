from src.landvaluetax.model.model import (
    neighbors_8,  # <-- replace with the actual module name
)


def test_neighbors_center_cell():
    # 3x3 grid, center index = 4 (row=1, col=1)
    G = 3
    idx = 4
    expected = [0, 1, 2, 3, 5, 6, 7, 8]  # all cells except 4 itself
    assert neighbors_8(idx, G) == expected


def test_neighbors_corner_cell():
    # 3x3 grid, top-left corner index = 0
    G = 3
    idx = 0
    expected = [1, 3, 4]  # right, below, diagonal-below-right
    assert neighbors_8(idx, G) == expected


def test_neighbors_edge_cell():
    # 3x3 grid, top edge but not corner: index = 1 (row=0, col=1)
    G = 3
    idx = 1
    expected = [0, 2, 3, 4, 5]  # left, right, and three below
    assert neighbors_8(idx, G) == expected
