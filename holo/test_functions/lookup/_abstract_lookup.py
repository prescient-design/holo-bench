"""Abstract base class for lookup-based benchmark functions."""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from botorch.test_functions.synthetic import SyntheticTestFunction


class AbstractLookup(SyntheticTestFunction):
    """
    Abstract base class for lookup-based benchmark functions.

    This class provides a template and common functionality for creating lookup-based
    test functions. Subclasses need to implement the _load_data method and define
    their specific properties (alphabet, dimension, etc.).

    Attributes:
        _optimal_value: Optional[float] = None  # Will be set after loading the data
        _optimizers: Optional[List[torch.Tensor]] = None  # Optimal solutions
        num_objectives: int = 1  # Single objective optimization
        wildtype_sequence: str  # The wild-type sequence if known
        dim: int  # The sequence length
        num_states: int  # Number of possible states for each position (alphabet size)
        alphabet_size: int  # Alias for num_states, used in tests
    """

    _optimal_value: Optional[float] = None
    _optimizers: Optional[List[torch.Tensor]] = None
    num_objectives = 1

    def __init__(
        self,
        dim: int,
        alphabet: Union[str, List[str]],
        wildtype_sequence: Optional[str] = None,
        noise_std: Optional[float] = None,
        negate: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize the abstract lookup function.

        Args:
            dim: The dimension (sequence length).
            alphabet: The alphabet as a string or list of characters.
            wildtype_sequence: The wild-type sequence if known.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function value.
            device: The torch device to use.
        """
        self.dim = dim
        self.categorical_inds = list(range(dim))

        # Set up alphabet and mappings
        if isinstance(alphabet, str):
            self.alphabet = list(alphabet)
        else:
            self.alphabet = alphabet

        self.num_states = len(self.alphabet)
        self.alphabet_size = self.num_states  # For test compatibility

        # Create character to index mappings
        self.char_to_index = {c: i for i, c in enumerate(self.alphabet)}
        self.index_to_char = {i: c for i, c in enumerate(self.alphabet)}

        # Set wildtype sequence if provided
        if wildtype_sequence:
            self.wildtype_sequence = wildtype_sequence

        # Define bounds for parameters
        bounds = [(0.0, float(self.num_states - 1)) for _ in range(self.dim)]

        super().__init__(
            noise_std=noise_std,
            negate=negate,
            bounds=bounds,
        )

        # Set device
        self._device = device or torch.device("cpu")

        # Load dataset
        self._lookup_dict, self._sorted_scores, self._sorted_seqs = self._load_data()

        # Find optimal value and optimizers
        self._optimal_value = float(max(self._lookup_dict.values()))
        self._optimizers = [
            self._seq_to_tensor(seq) for seq, score in self._lookup_dict.items() if score == self._optimal_value
        ]

    def _load_data(self) -> Tuple[Dict[str, float], np.ndarray, List[str]]:
        """Load the dataset.

        This method must be implemented by subclasses to load their specific datasets.

        Returns:
            A tuple containing:
                - lookup_dict: Dictionary mapping sequence strings to fitness scores
                - sorted_scores: Array of scores sorted in descending order
                - sorted_seqs: List of sequences sorted by descending score
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _seq_to_tensor(self, seq: str) -> torch.Tensor:
        """Convert a sequence string to a tensor of indices.

        Args:
            seq: Sequence string using the defined alphabet

        Returns:
            Tensor of alphabet indices
        """
        if len(seq) != self.dim:
            raise ValueError(f"Sequence length must be {self.dim}")

        indices = [self.char_to_index.get(char, 0) for char in seq]
        return torch.tensor(indices, device=self._device)

    def _tensor_to_seq(self, x: torch.Tensor) -> str:
        """Convert a tensor of indices to a sequence string.

        Args:
            x: Tensor of alphabet indices

        Returns:
            Sequence string using the defined alphabet
        """
        if len(x.shape) > 1:
            raise ValueError(f"Expected 1D tensor, got shape {x.shape}")

        indices = x.cpu().numpy().astype(int)
        return "".join(self.index_to_char[idx] for idx in indices)

    def _evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
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

    # syntactic sugar for botorch<0.14
    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        return self._evaluate_true(X)

    def forward(self, X: torch.Tensor, noise: bool = True) -> torch.Tensor:
        # cast X to float to ensure you can get noisy observations
        return super().forward(X.float(), noise)

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
        X = torch.randint(
            low=0,
            high=self.num_states,
            size=(n, self.dim),
            device=self._device,
        ).float()

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
            warnings.warn(f"No optimal solutions found for {self.__class__.__name__}.", stacklevel=2)
            return None

    def to(self, device, dtype=None):
        """Move the test function to the specified device and dtype.

        Args:
            device: torch.device
                The device to move to.
            dtype: torch.dtype
                The dtype to convert to.

        Returns:
            The test function on the specified device and with the specified dtype.
        """
        self._device = device
        # Convert optimizers to the right device and dtype
        if self._optimizers:
            if dtype is None:
                self._optimizers = [opt.to(device=device) for opt in self._optimizers]
            else:
                self._optimizers = [opt.to(device=device, dtype=dtype) for opt in self._optimizers]
        return self

    def __repr__(self):
        """String representation of the test function."""
        return (
            f"{self.__class__.__name__}("
            f"dim={self.dim}, "
            f"alphabet_size={self.alphabet_size}, "
            f"noise_std={self.noise_std}, "
            f"negate={self.negate})"
        )
