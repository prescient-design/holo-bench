from typing import Optional

import torch


def banded_square_matrix(
    ndim: int,
    bandwidth: int,
):
    """
    Generate a banded square matrix with entries 0 or 1.
    The rows wrap around, so the number of non-zero entries in each row is 2 * bandwidth + 1.

    Args:
    ndim: An integer representing the dimension of the square matrix.
    bandwidth: An integer representing the bandwidth of the matrix.

    Returns:
    A tensor of shape (ndim, ndim) representing the banded square matrix.
    """
    matrix = torch.zeros(ndim, ndim)
    for i in range(ndim):
        for j in range(i - bandwidth, i + bandwidth + 1):
            matrix[i, j % ndim] = 1
    return matrix


def dmp_marginal_dist(
    initial_dist: torch.Tensor,
    transition_matrix: torch.Tensor,
    num_steps: int,
):
    """Compute the marginal distribution of a discrete Markov process.

    Args:
    initial_dist: A tensor of shape (n, 1) representing the initial distribution.
    transition_matrix: A tensor of shape (n, n) representing the transition matrix.
    num_steps: An integer representing the number of steps to take.

    Returns:
    A tensor of shape (n, 1) representing the marginal distribution after num_steps steps.
    """
    marginal_dist = initial_dist
    for _ in range(num_steps):
        marginal_dist = torch.matmul(transition_matrix, marginal_dist)
    return marginal_dist


def dmp_stationary_dist(
    transition_matrix: torch.Tensor,
    tolerance: float = 1e-5,
):
    """Compute the stationary distribution of a discrete Markov process.

    Args:
    transition_matrix: A tensor of shape (n, n) representing the transition matrix.

    Returns:
    A tensor of shape (n, 1) representing the stationary distribution.
    """
    n = transition_matrix.shape[0]
    A = (torch.eye(n) - transition_matrix).T
    A = torch.cat([A, torch.ones(n).reshape(1, -1)], dim=0)

    b = torch.zeros(n + 1, 1)
    b[-1] = 1

    try:
        # get QR decomposition
        Q, R = torch.linalg.qr(A)
        soln = torch.linalg.solve(R, Q.T @ b)
        resid = torch.norm(A @ soln - b) / torch.norm(b)
        if resid > tolerance:
            raise torch._C._LinAlgError
    except torch._C._LinAlgError as err:
        msg = "Stationary distribution solve failed. " "Are you sure the stationary distribution exists?"
        raise RuntimeError(msg) from err

    return soln


def sample_sparse_ergodic_transition_matrix(
    num_states: int,
    bandwidth: int,
    softmax_temp: float = 1.0,
    generator: Optional[torch.Generator] = None,
    repeats_always_possible: bool = False,
):
    """
    Sample a transition matrix $P$ that satisfies the ergodicity conditions
    - irreducibility
    - aperiodicity
    - positive recurrence
    Additionally, some entries of the transition matrix must be zero.
    """

    if softmax_temp <= 0:
        msg = "Softmax temperature must be greater than 0."
        raise ValueError(msg)

    # set generator
    if generator is None:
        generator = torch.Generator()

    randn_matrix = torch.randn(num_states, num_states, generator=generator)
    dense_transition_matrix = (randn_matrix / softmax_temp).softmax(dim=-1)

    # construct mask as banded matrix
    mask = banded_square_matrix(num_states, bandwidth).bool()
    # shuffle the mask
    mask = mask[torch.randperm(num_states, generator=generator)]

    if repeats_always_possible:
        # set diagonal entries of mask to True
        mask = mask | torch.eye(num_states, dtype=torch.bool)

    transition_matrix = torch.where(mask, dense_transition_matrix, torch.zeros_like(dense_transition_matrix))
    transition_matrix = transition_matrix / transition_matrix.sum(dim=-1, keepdim=True)

    # check Perron-Frobenius theorem
    m = (num_states - 1) ** 2 + 1

    # compute P^m
    transition_matrix_m = transition_matrix
    for _ in range(m - 1):
        transition_matrix_m = transition_matrix_m @ transition_matrix

    if not torch.all(transition_matrix_m > 0):
        msg = "Perron-Frobenius theorem not satisfied."
        raise RuntimeError(msg)

    return transition_matrix


def sample_dmp(
    initial_dist: torch.Tensor,
    transition_matrix: torch.Tensor,
    num_steps: int,
    num_samples: int = 1,
    generator: Optional[torch.Generator] = None,
):
    samples = torch.zeros(num_samples, num_steps, dtype=torch.int64)
    if generator is None:
        generator = torch.Generator(device=initial_dist.device)
    samples[:, 0] = torch.multinomial(
        initial_dist,
        num_samples,
        replacement=True,
        generator=generator,
    )
    for t in range(1, num_steps):
        # import pdb; pdb.set_trace()
        samples[:, t] = torch.multinomial(
            transition_matrix[samples[:, t - 1]],
            1,
            generator=generator,
        ).squeeze(-1)
    return samples.to(initial_dist.device)


def dmp_sample_log_likelihood(
    samples: torch.Tensor,
    initial_dist: torch.Tensor,
    transition_matrix: torch.Tensor,
):
    samples = samples.long()  # may need to cast from float
    log_likelihood = 0.0
    log_likelihood += torch.log(initial_dist[samples[:, 0]])
    for t in range(1, samples.size(1)):
        log_likelihood += torch.log(transition_matrix[samples[:, t - 1], samples[:, t]])
    return log_likelihood
