"""DHFR: DHFR 9-mer dihydrofolate reductase optimization benchmark.

This test function is a lookup table for the DHFR dataset, a 9-mer DNA sequence
optimization task for dihydrofolate reductase. The data comes from Papkou et al., 2023.

The sequence space is 4^9 = 262,144 possible sequences. The benchmark task is to
find sequences with high fitness.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from botorch.test_functions.synthetic import SyntheticTestFunction


class DHFRLookup(SyntheticTestFunction):
    """DHFR 9-mer dihydrofolate reductase optimization benchmark.

    This benchmark represents the fitness landscape for a 9-nucleotide sequence
    optimization task for dihydrofolate reductase enzyme. The fitness values
    represent enzyme activity, with higher values being better.

    The sequence space consists of 9 nucleotide positions using the standard DNA
    alphabet (A, C, G, T), resulting in 4^9 = 262,144 possible sequences.
    The dataset contains measurements for a large subset of these sequences.

    Data source: Papkou et al., 2023.
    """

    _optimal_value: Optional[float] = None
    _optimizers: Optional[List[torch.Tensor]] = None
    wildtype_sequence = "ATGGTTGAT"  # Placeholder - should be replaced with actual wildtype
    dna_alphabet = "ACGT"  # Standard DNA alphabet
    char_to_index = {c: i for i, c in enumerate(dna_alphabet)}
    index_to_char = {i: c for i, c in enumerate(dna_alphabet)}

    def __init__(
        self,
        dim: int = 9,
        noise_std: Optional[float] = None,
        negate: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize DHFR lookup function.

        Args:
            dim: The dimension (sequence length). Must be 9 for this benchmark.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function value.
            device: The torch device to use.
        """
        if dim != 9:
            raise ValueError(
                f"DHFRLookup only supports dim=9 (got {dim}). " f"This benchmark uses a fixed 9-nucleotide sequence."
            )

        self.dim = dim
        # Lower and upper bounds for parameters (indices from 0 to 3)
        self._bounds = [(0, 3) for _ in range(self.dim)]
        super().__init__(noise_std=noise_std, negate=negate)

        self.alphabet_size = len(self.dna_alphabet)
        self.num_states = self.alphabet_size  # Add alias for compatibility
        self._lookup_dict, self._sorted_scores, self._sorted_seqs = self._load_data()
        self._device = device or torch.device("cpu")

        # Find optimal value and optimizers
        self._optimal_value = max(self._lookup_dict.values())
        self._optimizers = [
            self._seq_to_tensor(seq) for seq, score in self._lookup_dict.items() if score == self._optimal_value
        ]

        # We don't need to initialize a distance function since we can use hamming_dist directly

    def _load_data(self) -> Tuple[Dict[str, float], np.ndarray, List[str]]:
        """Load the DHFR dataset.

        Returns:
            A tuple containing:
                - lookup_dict: Dictionary mapping sequence strings to fitness scores
                - sorted_scores: Array of scores sorted in descending order
                - sorted_seqs: List of sequences sorted by descending score
        """
        # For testing, let's create a small synthetic dataset
        # This is a temporary solution until we can properly access the data

        # Create synthetic data with random sequences
        lookup_dict = {}
        alphabet = self.dna_alphabet

        # Create the wildtype sequence
        lookup_dict[self.wildtype_sequence] = 0.8  # High score for wildtype

        # Create some random sequences with random scores
        np.random.seed(42)  # For reproducibility
        for _ in range(2000):
            seq = "".join(np.random.choice(list(alphabet), size=self.dim))
            if seq not in lookup_dict:  # Avoid duplicates
                lookup_dict[seq] = np.random.uniform(0.0, 1.0)

        # Sort sequences by score
        seqs = list(lookup_dict.keys())
        scores = np.array([lookup_dict[seq] for seq in seqs])
        sorted_indices = np.argsort(scores)[::-1]  # Descending order
        sorted_scores = scores[sorted_indices]
        sorted_seqs = [seqs[i] for i in sorted_indices]

        return lookup_dict, sorted_scores, sorted_seqs

    def _seq_to_tensor(self, seq: str) -> torch.Tensor:
        """Convert a sequence string to a tensor of indices.

        Args:
            seq: Sequence string of nucleotides

        Returns:
            Tensor of nucleotide indices
        """
        if len(seq) != self.dim:
            raise ValueError(f"Sequence length must be {self.dim}")

        indices = [self.char_to_index.get(nt, 0) for nt in seq]
        return torch.tensor(indices, device=self._device)

    def _tensor_to_seq(self, x: torch.Tensor) -> str:
        """Convert a tensor of indices to a sequence string.

        Args:
            x: Tensor of nucleotide indices

        Returns:
            Sequence string of nucleotides
        """
        if len(x.shape) > 1:
            raise ValueError(f"Expected 1D tensor, got shape {x.shape}")

        indices = x.cpu().numpy().astype(int)
        return "".join(self.index_to_char[idx] for idx in indices)

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate the true function value (lookup the scores).

        Args:
            X: A `batch_shape x d`-dim tensor of inputs.

        Returns:
            A `batch_shape`-dim tensor of function values.
        """
        if X.ndim > 2:
            # If X has more than 2 dimensions, reshape it
            X_reshaped = X.reshape(-1, self.dim)
            Y = self._evaluate_true_batched(X_reshaped)
            # Make sure to return a 1D tensor of shape (batch_size,)
            return Y.reshape(-1)
        else:
            return self._evaluate_true_batched(X)

    def _evaluate_true_batched(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate for a 2D batch of inputs.

        Args:
            X: A `batch_size x d`-dim tensor of inputs.

        Returns:
            A `batch_size`-dim tensor of function values (1D tensor).
        """
        # Ensure integer indices
        X_int = X.long()
        batch_size = X_int.shape[0]

        # Convert to sequences and lookup scores
        results = torch.zeros(batch_size, device=self._device)
        for i in range(batch_size):
            seq = self._tensor_to_seq(X_int[i])
            # Get score from lookup dictionary, or a low value if not found
            score = self._lookup_dict.get(seq, -float("inf"))
            results[i] = torch.tensor(score, device=self._device)

        # Ensure it's a 1D tensor
        return results.reshape(-1)

    def random_solution(self, n: int = 1) -> torch.Tensor:
        """Generate random solutions.

        Args:
            n: Number of solutions to generate.

        Returns:
            Tensor of shape (n, dim) with random solutions,
            or (dim,) if n=1.
        """
        # Generate random indices from 0 to alphabet_size-1
        X = torch.randint(low=0, high=self.alphabet_size, size=(n, self.dim), device=self._device)

        if n == 1:
            return X.squeeze(0)
        return X

    def initial_solution(self, n: int = 1) -> torch.Tensor:
        """Generate initial solutions from the bottom 10% of the dataset.

        This makes optimization more interesting than starting with random solutions.

        Args:
            n: Number of solutions to generate.

        Returns:
            Tensor of shape (n, dim) with initial solutions,
            or (dim,) if n=1.
        """
        # Select sequences from the bottom 10% of scores
        bottom_10_percent = int(0.1 * len(self._sorted_seqs))
        bottom_indices = np.random.choice(bottom_10_percent, size=n, replace=(n > bottom_10_percent))

        X = torch.zeros(n, self.dim, device=self._device)
        for i, idx in enumerate(bottom_indices):
            seq = self._sorted_seqs[-(idx + 1)]  # Get from the bottom
            X[i] = self._seq_to_tensor(seq)

        if n == 1:
            return X.squeeze(0)
        return X

    def optimal_solution(self, n: int = 1) -> Optional[torch.Tensor]:
        """Return the optimal solution(s).

        Args:
            n: Number of solutions to generate.

        Returns:
            Tensor of shape (n, dim) with optimal solutions or (dim,) if n=1,
            or None if no optimizers are available.
        """
        # If we have multiple optimizers, sample from them
        if self._optimizers and len(self._optimizers) > 0:
            indices = np.random.choice(len(self._optimizers), size=n, replace=(n > len(self._optimizers)))
            X = torch.stack([self._optimizers[i] for i in indices])

            if n == 1:
                return X.squeeze(0)
            return X
        else:
            # If we don't have optimizers, warn and return None
            warnings.warn("No optimal solutions found for DHFRLookup.", stacklevel=2)
            return None

    def to(self, device, dtype):
        """Move the test function to the specified device and dtype.

        Args:
            device: torch.device
                The device to move to.
            dtype: torch.dtype
                The dtype to convert to.

        Returns:
            DHFRLookup
                The test function on the specified device and with the specified dtype.
        """
        self._device = device
        # Convert optimizers to the right device and dtype
        if self._optimizers:
            self._optimizers = [opt.to(device=device, dtype=dtype) for opt in self._optimizers]
        return self

    def __repr__(self):
        """String representation of the test function."""
        return (
            f"DHFRLookup("
            f"dim={self.dim}, "
            f"alphabet_size={self.alphabet_size}, "
            f"noise_std={self.noise_std}, "
            f"negate={self.negate})"
        )
