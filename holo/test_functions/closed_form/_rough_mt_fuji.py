import torch
from botorch.test_functions import SyntheticTestFunction

from holo.test_functions.elemental import hamming_dist


class RoughMtFuji(SyntheticTestFunction):
    """
    Implementation of the Rough Mt. Fuji fitness landscape
    as described in Neidhart et al. (2014)
    'Adaptation in tunably rugged fitness landscapes:
    The Rough Mount Fuji model'
    https://arxiv.org/abs/1402.3065
    """

    num_states = 2
    num_centroids = 1

    def __init__(
        self,
        dim: int = 2,
        additive_term: float = 0.25,
        random_term_std: float = 1.0,
        noise_std: float = 0.0,
        negate: bool = False,
        random_seed: int = 0,
    ):
        self.dim = dim
        super().__init__(
            noise_std=noise_std,
            negate=negate,
            bounds=[(0.0, 1.0) for _ in range(dim)],
        )
        self._random_seed = random_seed
        self._generator = torch.Generator().manual_seed(random_seed)
        self.centroids = torch.randint(0, 2, (1, dim), generator=self._generator)
        self._additive_term = additive_term
        self._random_term_std = random_term_std
        self._random_term = torch.randn(1, dim, generator=self._generator) * random_term_std

    def _evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        dist = hamming_dist(X, self.centroids, dim=-1)
        return -self._additive_term * dist + (self._random_term * X).sum(dim=-1)

    # syntactic sugar for botorch<0.14
    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        return self._evaluate_true(X)

    def forward(self, X: torch.Tensor, noise: bool = True) -> torch.Tensor:
        # cast X to float to ensure you can get noisy observations
        return super().forward(X.float(), noise)

    def initial_solution(self, n: int = 1):
        # reset generator seed so initial solution is always the same
        self._generator = self._generator.manual_seed(self._random_seed)
        return self.random_solution(n)

    def random_solution(self, n: int = 1):
        unif_samples = torch.randint(self.num_states, (n, self.dim), device=self.centroids.device)
        if n == 1:
            return unif_samples.squeeze(0)
        return unif_samples

    @property
    def _optimal_value(self):
        return self.evaluate_true(self.optimal_solution()).squeeze()

    def optimal_solution(self):
        soln = self.centroids.clone()
        mask = self._random_term - self._additive_term > 0
        mask = mask.to(device=soln.device)
        soln = torch.where(mask, torch.ones_like(soln), soln)
        mask = self._random_term + self._additive_term < 0
        mask = mask.to(device=soln.device)
        soln = torch.where(mask, torch.zeros_like(soln), soln)

        return soln

    def to(self, device, dtype):
        self.centroids = self.centroids.to(device, dtype)
        self._generator = torch.Generator(device=device).manual_seed(self._random_seed)
        self._random_term = self._random_term.to(device)
        return self

    def __repr__(self):
        return (
            f"RoughMtFuji("
            f"num_states={self.num_states}, "
            f"dim={self.dim}, "
            f"additive_term={self._additive_term}, "
            f"random_term={self._random_term}, "
            f"random_term_std={self._random_term_std}, "
            f"centroids={self.centroids}, "
            f"noise_std={self.noise_std}, "
            f"negate={self.negate}, "
            f"random_seed={self._random_seed})"
        )
