import torch

from holo.optim import DiscreteEvolution
from holo.test_functions.closed_form import Ehrlich


def test_discrete_evolution_single_param():
    seq_len = 3
    vocab_size = 5
    num_steps = 20
    mutation_prob = 0.2
    recombine_prob = 0.5
    params = [
        torch.nn.Parameter(
            torch.tensor([vocab_size - 1] * seq_len).float(),
        )
    ]
    vocab = list(range(vocab_size))
    optimizer = DiscreteEvolution(params, vocab, mutation_prob=mutation_prob, recombine_prob=recombine_prob)

    def closure(param_list):
        return param_list[0].sum(-1)

    for _ in range(num_steps):
        loss = optimizer.step(closure)

    assert torch.allclose(loss, torch.zeros_like(loss))
    assert torch.allclose(params[0].data, torch.zeros_like(params[0].data))


def test_discrete_evolution_param_groups():
    seq_lens = [1, 3]
    vocab_size = 5
    num_steps = 20

    param_groups = [
        {
            "params": [
                torch.nn.Parameter(
                    torch.tensor([vocab_size - 1] * seq_lens[0]).float(),
                )
            ],
            "mutation_prob": 0.9,
            "recombine_prob": 0.1,
        },
        {
            "params": [
                torch.nn.Parameter(
                    torch.tensor([vocab_size - 1] * seq_lens[1]).float(),
                )
            ],
            "mutation_prob": 0.1,
            "recombine_prob": 0.9,
        },
    ]

    vocab = list(range(vocab_size))
    optimizer = DiscreteEvolution(param_groups, vocab)

    def closure(param_list):
        return sum(p.sum(-1) for p in param_list)

    for _ in range(num_steps):
        loss = optimizer.step(closure)

    assert torch.allclose(loss, torch.zeros_like(loss))
    param_list = [p.data for group in param_groups for p in group["params"]]
    cat_param = torch.cat(param_list, dim=-1)
    assert torch.allclose(cat_param, torch.zeros_like(cat_param))


def test_ehrlich_optimization():
    num_states = 32
    num_steps = 128
    motif_length = 4
    noise_std = 0.0
    random_seed = 0
    ehrlich = Ehrlich(
        num_states=num_states,
        dim=num_steps,
        num_motifs=1,
        motif_length=motif_length,
        noise_std=noise_std,
        negate=True,
        random_seed=random_seed,
    )

    # initialization
    initial_solution = ehrlich.initial_solution()

    def closure(param_list):
        return ehrlich(param_list[0])

    params = [torch.nn.Parameter(initial_solution.float())]

    mutation_prob = 1 / num_steps
    recombine_prob = 1 / num_steps
    survival_quantile = 0.01
    num_particles = 2048
    optimizer = DiscreteEvolution(
        params,
        vocab=list(range(num_states)),
        mutation_prob=mutation_prob,
        recombine_prob=recombine_prob,
        survival_quantile=survival_quantile,
        num_particles=num_particles,
    )
    num_opt_steps = 256

    best = float("inf")
    for t in range(num_opt_steps):
        loss = optimizer.step(closure)
        if loss < best:
            best = loss.item()

    assert best == -1.0
