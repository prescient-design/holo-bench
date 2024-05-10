from typing import Iterable

import torch
from torch.optim import Optimizer


class DiscreteEvolution(Optimizer):
    r"""Implements discrete evolution optimizer."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        vocab: Iterable[list[int]],
        mutation_prob: float = 0.2,
        recombine_prob: float = 0.5,
        num_particles: int = 128,
        survival_quantile: float = 0.5,
    ):
        """
        Args:
            params: Iterable of parameters to optimize or dicts defining parameter groups.
            vocab: Iterable of lists of integers representing the vocabulary for each parameter group.
            mutation_prob: Probability of mutating each element of the parameters.
            recombine_prob: Probability of recombining elements of the parameters.
            num_particles: Number of particles to maintain for each parameter.
            survival_quantile: Fraction of particles to keep at each step.
        """
        if mutation_prob <= 0.0 or mutation_prob >= 1.0:
            msg = "Mutation probability must be in (0, 1)."
            raise ValueError(msg)

        if recombine_prob < 0.0 or recombine_prob >= 1.0:
            msg = "Recombination probability must be in [0, 1)."
            raise ValueError(msg)

        if num_particles < 1:
            msg = "Number of particles must be positive."
            raise ValueError(msg)

        if survival_quantile <= 0.0 or survival_quantile >= 1.0:
            msg = "Survival quantile must be in (0, 1)."
            raise ValueError(msg)

        if survival_quantile * num_particles < 2:
            msg = "Survival quantile must be large enough to keep at least two particles."
            raise ValueError(msg)

        defaults = dict(
            mutation_prob=mutation_prob,
            recombine_prob=recombine_prob,
            vocab=vocab,
        )

        super(DiscreteEvolution, self).__init__(params, defaults)
        self._num_particles = num_particles
        self._survival_quantile = survival_quantile
        self.particle_loss = None

    def _init_group(self, group, particle_buffer_list):
        for p in group["params"]:
            state = self.state[p]
            if "particles" not in state:
                repeat_args = [1] * len(p.data.shape)
                particles = p.data.clone().repeat(self._num_particles, *repeat_args)
                particles.requires_grad_(False)
                state["particles"] = particles
                particle_buffer_list.append(particles)
            else:
                particle_buffer_list.append(state["particles"])

    def step(self, closure: callable):
        particle_buffer_list = []
        for group in self.param_groups:
            self._init_group(group, particle_buffer_list)

        self.particle_loss = closure(particle_buffer_list)  # (num_particles,)
        particle_arg_min = torch.argmin(self.particle_loss)
        soln_loss = self.particle_loss[particle_arg_min]

        # update parameters to current best
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                particles = state["particles"]
                p.data = particles[particle_arg_min].clone()

        # update particles
        # particle survival threshold determined by quantile
        threshold = torch.quantile(self.particle_loss, self._survival_quantile)
        if threshold < float("inf"):
            surviving = self.particle_loss <= threshold
        else:
            surviving = torch.ones_like(self.particle_loss, dtype=torch.bool)
        for group in self.param_groups:
            num_params = len(group["params"])
            count = 0
            while count < num_params:
                particles = particle_buffer_list.pop(0)
                count += 1

                surviving_particles = particles[surviving]
                num_surviving = surviving_particles.size(0)
                num_missing = self._num_particles - num_surviving

                # repopulate missing particles by recombining survivors
                parent_1 = torch.randint(0, num_surviving, (num_missing,))
                parent_2 = torch.randint(0, num_surviving, (num_missing,))
                recombine_prob = group["recombine_prob"]
                recombine_mask = torch.rand_like(surviving_particles[parent_1]) < recombine_prob
                replace_particles = torch.where(
                    recombine_mask,
                    surviving_particles[parent_1],
                    surviving_particles[parent_2],
                )
                particles[~surviving] = replace_particles

                # mutate particles
                mutation_mask = torch.rand_like(particles) < group["mutation_prob"]
                vocab_tensor = torch.tensor(group["vocab"], device=particles.device)
                replace_elements = torch.randint(
                    len(vocab_tensor),
                    particles.shape,
                    device=particles.device,
                    dtype=particles.dtype,
                )
                new_particles = torch.where(mutation_mask, replace_elements, particles)
                particles.copy_(new_particles)

        return soln_loss
