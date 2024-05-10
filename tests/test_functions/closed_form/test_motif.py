import torch

from holo.test_functions.elemental import motif_search


def test_motif_search():
    cases = torch.tensor(
        [
            [0, 1, 2, 3, 4],
            [0, 1, 2, 4, 5],
            [0, 1, 4, 5, 6],
            [0, 4, 5, 6, 7],
            [4, 5, 6, 7, 8],
        ]
    )
    motif = cases[0, :-1]

    res = motif_search(
        solution=cases,
        motif=motif,
        spacing=None,
        mode="present",
        quantization=1,
    )
    assert torch.all(res == torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]))

    res = motif_search(
        solution=cases,
        motif=motif,
        spacing=None,
        mode="present",
        quantization=4,
    )
    assert torch.all(res == torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]))


def test_spaced_motif_search():
    cases = torch.tensor(
        [
            [0, 1, 2, 3, 4],
            [0, 1, 2, 4, 5],
            [0, 1, 4, 5, 6],
            [0, 4, 5, 6, 7],
            [4, 5, 6, 7, 8],
        ]
    )
    motif = cases[0, ::2]
    spacing = torch.tensor([2, 2])
    res = motif_search(
        solution=cases,
        motif=motif,
        spacing=spacing,
        mode="present",
        quantization=1,
    )
    assert torch.all(res == torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]))
