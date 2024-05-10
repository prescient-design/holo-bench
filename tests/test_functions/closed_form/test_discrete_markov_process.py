import torch

from holo.test_functions.elemental import (
    banded_square_matrix,
    dmp_sample_log_likelihood,
    dmp_stationary_dist,
    sample_dmp,
    sample_sparse_ergodic_transition_matrix,
)


def test_banded_square_matrix():
    ndim = 5
    bandwidth = 1
    matrix = banded_square_matrix(ndim, bandwidth)
    expected_matrix = torch.tensor(
        [
            [1, 1, 0, 0, 1],
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [1, 0, 0, 1, 1],
        ]
    ).float()
    assert torch.allclose(matrix, expected_matrix)


def test_dmp_sample_log_likelihood():
    ndim = 5
    bandwidth = 1
    num_steps = 11
    num_samples = 7
    random_seed = 0
    initial_dist = torch.ones(ndim) / ndim
    generator = torch.Generator().manual_seed(random_seed)
    transition_matrix = sample_sparse_ergodic_transition_matrix(ndim, bandwidth, generator=generator)
    samples = sample_dmp(initial_dist, transition_matrix, num_steps, num_samples, generator=generator)
    log_likelihood = dmp_sample_log_likelihood(samples, initial_dist, transition_matrix)
    assert log_likelihood.shape == (num_samples,)
    assert torch.all(log_likelihood > -float("inf"))


def test_dmp_stationary_dist():
    ndim = 5
    bandwidth = 1
    random_seed = 1
    generator = torch.Generator().manual_seed(random_seed)
    transition_matrix = sample_sparse_ergodic_transition_matrix(ndim, bandwidth, generator=generator)
    stationary_dist = dmp_stationary_dist(transition_matrix)
    total_prob = stationary_dist.sum()
    assert stationary_dist.size() == (ndim, 1)
    assert torch.allclose(total_prob, torch.tensor(1.0))
    apply_transition = stationary_dist.t() @ transition_matrix
    assert torch.allclose(apply_transition, stationary_dist.t())


def test_sample_sparse_ergodic_transition_matrix():
    ndim = 5
    bandwidth = 1
    random_seed = 2
    generator = torch.Generator().manual_seed(random_seed)
    transition_matrix = sample_sparse_ergodic_transition_matrix(ndim, bandwidth, generator=generator)
    assert transition_matrix.size() == (ndim, ndim)
    assert torch.allclose(transition_matrix.sum(-1), torch.ones(ndim))


def test_sample_dmp():
    ndim = 5
    bandwidth = 1
    num_steps = 11
    num_samples = 7
    random_seed = 3
    generator = torch.Generator().manual_seed(random_seed)
    initial_dist = torch.ones(ndim) / ndim
    transition_matrix = sample_sparse_ergodic_transition_matrix(ndim, bandwidth, generator=generator)
    samples = sample_dmp(initial_dist, transition_matrix, num_steps, num_samples, generator=generator)

    assert samples.size() == (num_samples, num_steps)
    assert torch.all(samples.lt(ndim))
