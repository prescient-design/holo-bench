import os
import tempfile

import numpy as np
import pooch
import torch
from botorch.test_functions import SyntheticTestFunction


class TFBIND8Lookup(SyntheticTestFunction):
    """
    TFBIND8 Lookup function.

    This is a lookup-based function for the 8-mer DNA transcription factor binding
    dataset. The dataset contains 65,792 sequences of 8 nucleotides and their
    corresponding binding affinity scores.

    The DNA alphabet is {A, C, G, T} encoded as integers {0, 1, 2, 3}.

    Args:
        noise_std: float, default=0.0
            Standard deviation of Gaussian noise added to the output.
        negate: bool, default=False
            If True, negate the function values. Default is maximizing the binding affinity.
    """

    _optimal_value = None  # Will be set after loading the data
    num_objectives = 1
    # Sequence length is fixed at 8 for TFBIND8
    dim = 8

    def __init__(
        self,
        noise_std: float = 0.0,
        negate: bool = False,
        dim: int = 8,  # Added for config compatibility, but can't be changed
    ) -> None:
        if dim != 8:
            raise ValueError("TFBIND8Lookup has a fixed sequence length of 8 and cannot be changed.")

        self.alphabet = ["A", "C", "G", "T"]
        self.num_states = len(self.alphabet)
        self._device = torch.device("cpu")  # Default device

        # Define bounds for the design space (each dimension can be 0, 1, 2, or 3)
        bounds = [(0.0, float(self.num_states - 1)) for _ in range(self.dim)]

        super(TFBIND8Lookup, self).__init__(
            noise_std=noise_std,
            negate=negate,
            bounds=bounds,
        )

        # Load data
        self.x_data, self.y_data = self._load_data()

        # Create lookup dictionary - map from tuples of sequence indices to score
        self.sequence_to_score = {}
        for i in range(len(self.x_data)):
            seq_tuple = tuple(self.x_data[i].tolist())
            self.sequence_to_score[seq_tuple] = self.y_data[i][0]

        # Set optimal value
        max_idx = np.argmax(self.y_data)
        self._optimal_value = float(self.y_data[max_idx][0])
        self._optimizers = [torch.tensor(self.x_data[max_idx].astype(float))]

    def _load_data(self):
        """
        Load the TFBIND8 dataset using pooch.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The sequence data and their scores.
        """
        # URLs for the data files (from GitHub raw content)
        base_url = "https://github.com/csiro-funml/variationalsearch/raw/main/data/TFBIND8"

        # Paths for data
        x_path = "tf_bind_8-x-0.npy"
        y_path = "tf_bind_8-y-0.npy"

        # Use pooch to fetch data
        x_file = pooch.retrieve(
            url=f"{base_url}/{x_path}",
            known_hash="8957a728704d2e6c5dbba921dd9e16cbd76797489b081c5dc7f39ca23582d95d",
            fname=os.path.basename(x_path),
            path=tempfile.gettempdir(),  # Store in temp dir
            progressbar=True,
        )

        y_file = pooch.retrieve(
            url=f"{base_url}/{y_path}",
            known_hash="a61dd97796a4173d3a2a3d016db44e5d42e1f1d6a39d9f1ad2465d2c6b94fcb7",
            fname=os.path.basename(y_path),
            path=tempfile.gettempdir(),  # Store in temp dir
            progressbar=True,
        )

        # Load the data
        x_data = np.load(x_file)
        y_data = np.load(y_file)

        return x_data, y_data

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the function at the given points.

        Args:
            X: torch.Tensor
                Input tensor of shape (batch_size, 1, dim) or (batch_size, dim)
                containing integer indices representing DNA sequences.

        Returns:
            torch.Tensor
                Output tensor of shape (batch_size,) containing the binding
                affinity scores for each sequence.
        """
        # Ensure X has the right shape
        if X.dim() == 3:
            X = X.squeeze(1)  # (batch_size, 1, dim) -> (batch_size, dim)

        batch_size = X.shape[0]
        results = torch.empty(batch_size, dtype=torch.float, device=X.device)

        # Convert to numpy for processing
        X_np = X.detach().cpu().numpy().astype(int)

        # Lookup each sequence's score
        for i in range(batch_size):
            seq_tuple = tuple(X_np[i].tolist())

            if seq_tuple in self.sequence_to_score:
                results[i] = torch.tensor(float(self.sequence_to_score[seq_tuple]))
            else:
                # If not found, return a very low score
                results[i] = torch.tensor(float("-inf"))

        return results

    def random_solution(self, n: int = 1):
        """
        Generate random solutions from the design space.

        Args:
            n: int, default=1
                Number of random solutions to generate.

        Returns:
            torch.Tensor
                Random solution(s) of shape (n, dim) or (dim,) if n=1.
        """
        solutions = torch.randint(
            low=0,
            high=self.num_states,
            size=(n, self.dim),
            device=self._device,
        )

        if n == 1:
            return solutions.squeeze(0)
        return solutions

    def initial_solution(self, n: int = 1):
        """
        Generate initial solutions based on the dataset.

        Args:
            n: int, default=1
                Number of initial solutions to generate.

        Returns:
            torch.Tensor
                Initial solution(s) from the dataset.
        """
        # Sample from bottom 10% of dataset (worst performers)
        bottom_indices = np.argsort(self.y_data.flatten())[: int(0.1 * len(self.y_data))]
        selected_indices = np.random.choice(bottom_indices, size=n, replace=n > len(bottom_indices))

        solutions = torch.tensor(
            self.x_data[selected_indices].astype(float),
            device=self._device,
        )

        if n == 1:
            return solutions.squeeze(0)
        return solutions

    def optimal_solution(self):
        """
        Return the optimal solution (sequence with highest binding affinity).

        Returns:
            torch.Tensor
                The sequence with the highest binding affinity.
        """
        return self._optimizers[0]

    def to(self, device, dtype):
        """
        Move the test function to the specified device and dtype.

        Args:
            device: torch.device
                The device to move to.
            dtype: torch.dtype
                The dtype to convert to.

        Returns:
            TFBIND8Lookup
                The test function on the specified device and with the specified dtype.
        """
        self._device = device
        # Convert optimizers to the right device and dtype
        self._optimizers = [opt.to(device=device, dtype=dtype) for opt in self._optimizers]
        return self

    def __repr__(self):
        """String representation of the test function."""
        return (
            f"TFBIND8Lookup("
            f"dim={self.dim}, "
            f"num_states={self.num_states}, "
            f"noise_std={self.noise_std}, "
            f"negate={self.negate})"
        )
