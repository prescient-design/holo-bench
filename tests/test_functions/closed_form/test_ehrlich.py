import pytest
import torch

from holo.test_functions.closed_form import (
    Ehrlich,
)
from holo.test_functions.elemental import (
    dmp_stationary_dist,
    sample_dmp,
)


def test_ehrlich_single_motif():
    num_states = 32
    num_steps = 256
    motif_length = 8
    noise_std = 0.0
    negate = False
    random_seed = 0
    ehrlich = Ehrlich(
        num_states=num_states,
        dim=num_steps,
        motif_length=motif_length,
        noise_std=noise_std,
        negate=negate,
        random_seed=random_seed,
    )
    num_samples = 4
    stationary_dist = dmp_stationary_dist(ehrlich.transition_matrix)
    generator = torch.Generator().manual_seed(random_seed)
    dmp_samples = sample_dmp(
        initial_dist=stationary_dist.squeeze(),
        transition_matrix=ehrlich.transition_matrix,
        num_steps=ehrlich.dim,
        num_samples=num_samples,
        generator=generator,
    )
    f = ehrlich.evaluate_true(dmp_samples)
    assert torch.all(f >= 0.0)
    assert torch.all(f <= 1.0)

    unif_samples = torch.randint(0, num_states, (num_samples, num_steps))
    f = ehrlich.evaluate_true(unif_samples)
    assert torch.allclose(f, torch.zeros_like(f))


def test_ehrlich_multi_motif():
    num_states = 32
    num_steps = 256
    motif_length = 8
    random_seed = 0
    ehrlich = Ehrlich(
        num_states=num_states,
        dim=num_steps,
        num_motifs=4,
        motif_length=motif_length,
        random_seed=random_seed,
    )
    num_samples = 4
    stationary_dist = dmp_stationary_dist(ehrlich.transition_matrix)
    generator = torch.Generator().manual_seed(random_seed)
    dmp_samples = sample_dmp(
        initial_dist=stationary_dist.squeeze(),
        transition_matrix=ehrlich.transition_matrix,
        num_steps=ehrlich.dim,
        num_samples=num_samples,
        generator=generator,
    )
    f = ehrlich.evaluate_true(dmp_samples)
    assert torch.all(f >= 0.0)
    assert torch.all(f <= 1.0)

    unif_samples = torch.randint(0, num_states, (num_samples, num_steps))
    f = ehrlich.evaluate_true(unif_samples)
    assert torch.allclose(f, torch.zeros_like(f))


def test_invalid_ehrlich():
    with pytest.raises(ValueError):
        Ehrlich(
            num_states=32,
            dim=2,
            num_motifs=1,
            motif_length=8,
        )

    with pytest.raises(ValueError):
        Ehrlich(
            num_states=32,
            dim=16,
            num_motifs=4,
            motif_length=8,
            random_seed=0,
        )
