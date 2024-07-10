import torch
from botorch.test_functions import SyntheticTestFunction

from holo.test_functions.elemental import (
    dmp_sample_log_likelihood,
    dmp_stationary_dist,
    motif_search,
    sample_dmp,
    sample_sparse_ergodic_transition_matrix,
)


class Ehrlich(SyntheticTestFunction):
    _optimal_value = 1.0
    num_objectives = 1

    def __init__(
        self,
        num_states: int = 5,
        dim: int = 7,
        num_motifs: int = 1,
        motif_length: int = 3,
        quantization: int | None = None,
        epistasis_factor: float = 0.0,
        noise_std: float = 0.0,
        negate: bool = False,
        random_seed: int = 0,
    ) -> None:
        bounds = [(0.0, float(num_states - 1)) for _ in range(dim)]
        self.num_states = num_states
        self.dim = dim
        self._random_seed = random_seed
        self._motif_length = motif_length
        self._quantization = quantization
        super(Ehrlich, self).__init__(
            noise_std=noise_std,
            negate=negate,
            bounds=bounds,
        )
        self._generator = torch.Generator().manual_seed(random_seed)
        self._epistasis_factor = epistasis_factor
        self.initial_dist = torch.ones(num_states) / num_states
        bandwidth = int(num_states * 0.4)
        self.transition_matrix = sample_sparse_ergodic_transition_matrix(
            num_states, bandwidth, softmax_temp=0.5, generator=self._generator, repeats_always_possible=True
        )
        self.stationary_dist = dmp_stationary_dist(self.transition_matrix)

        slack_positions = dim - num_motifs * motif_length
        element_gaps = num_motifs * (motif_length - 1)
        max_spacing = 1 + slack_positions // element_gaps
        if max_spacing < 1:
            raise ValueError("cannot guarantee a solution satisfying all motifs exists.")

        # draw motifs as single sequence from DMP and chunk to ensure feasible soln exists
        all_motifs = sample_dmp(
            initial_dist=self.stationary_dist.squeeze(-1),
            transition_matrix=self.transition_matrix,
            num_steps=num_motifs * motif_length,
            num_samples=1,
            generator=self._generator,
        ).squeeze(0)
        self.motifs = torch.chunk(all_motifs, num_motifs, dim=0)

        self.spacings = []
        for _ in range(num_motifs):
            # draw random spacing
            # spacing = torch.randint(1, max_spacing + 1, (motif_length - 1,), generator=self._generator)
            # random draw from (motif_length - 1) simplex
            weights = torch.rand(motif_length - 1, generator=self._generator)
            weights /= weights.sum()
            spacing = (slack_positions // num_motifs) * weights
            # round down
            spacing = 1 + spacing.floor().to(torch.int64)
            self.spacings.append(spacing)

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        motif_contrib = []
        for motif, spacing in zip(self.motifs, self.spacings):
            motif_present = motif_search(
                solution=X,
                motif=motif,
                spacing=spacing,
                mode="present",
                quantization=self._quantization,
            )
            response = _cubic_response(motif_present, self._epistasis_factor)
            motif_contrib.append(response)

        all_motifs_contrib = torch.stack(motif_contrib).prod(dim=0)
        log_likelihood = dmp_sample_log_likelihood(
            samples=X,
            initial_dist=self.initial_dist,
            transition_matrix=self.transition_matrix,
        )
        is_feasible = log_likelihood > -float("inf")
        return torch.where(is_feasible, all_motifs_contrib, -float("inf"))

    def initial_solution(self, n: int = 1):
        # reset generator seed so initial solution is always the same
        self._generator = self._generator.manual_seed(self._random_seed)
        dmp_samples = sample_dmp(
            initial_dist=self.stationary_dist.squeeze(-1),
            transition_matrix=self.transition_matrix,
            num_steps=self.dim,
            num_samples=n,
            generator=self._generator,
        )
        if n == 1:
            return dmp_samples.squeeze(0)
        return dmp_samples

    def random_solution(self, n: int = 1):
        unif_samples = torch.randint(self.num_states, (n, self.dim), device=self.initial_dist.device)
        if n == 1:
            return unif_samples.squeeze(0)
        return unif_samples

    def optimal_solution(self):
        # sample random sequence from DMP
        soln = torch.zeros(self.dim, dtype=torch.int64, device=self.initial_dist.device)
        # fill in spaced motifs with repeats
        position = 0
        for motif, spacing in zip(self.motifs, self.spacings):
            spacing = torch.cat(
                [
                    torch.tensor([0], device=self.initial_dist.device),
                    spacing,
                ]
            )
            index = spacing.cumsum(0).tolist()
            print(index)
            motif = motif.tolist()
            for idx in range(index[-1] + 1):
                if idx in index:
                    next_state = motif.pop(0)
                soln[position] = next_state
                # print(position)
                position += 1
        # fill in remaining states with last state of last motif
        soln[position:] = soln[position - 1]

        # check optimal value
        optimal_value = self.evaluate_true(soln.unsqueeze(0))
        if not optimal_value == self._optimal_value:
            print(soln)
            print(optimal_value)
            raise RuntimeError("optimal value not achieved by optimal solution.")
        return soln

    def to(self, device, dtype):
        self.transition_matrix = self.transition_matrix.to(device, dtype)
        self.initial_dist = self.initial_dist.to(device, dtype)
        self.stationary_dist = self.stationary_dist.to(device, dtype)
        self.motifs = [motif.to(device) for motif in self.motifs]
        self.spacings = [spacing.to(device) for spacing in self.spacings]
        self._generator = torch.Generator(device=device).manual_seed(self._random_seed)
        return self

    def __repr__(self):
        motif_list = [f"motif_{i}: {motif.tolist()}" for i, motif in enumerate(self.motifs)]
        spacing_list = [f"spacing_{i}: {spacing.tolist()}" for i, spacing in enumerate(self.spacings)]
        return (
            f"Ehrlich("
            f"num_states={self.num_states}, "
            f"dim={self.dim}, "
            f"num_motifs={len(self.motifs)}, "
            f"motifs=[{', '.join(motif_list)}], "
            f"spacings=[{', '.join(spacing_list)}], "
            f"quantization={self._quantization}, "
            f"noise_std={self.noise_std}, "
            f"negate={self.negate}, "
            f"random_seed={self._random_seed})"
        )


def _cubic_response(X: torch.Tensor, epistasis_factor: float):
    coeff = epistasis_factor * X * (X - 1.0) + 1.0
    return coeff * X
