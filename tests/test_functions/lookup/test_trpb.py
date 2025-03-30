import os
from typing import Dict, List, Tuple

import numpy as np
import pytest
import torch

from holo.test_functions.lookup import TRPBLookup


# Mock implementation that doesn't download data
class MockTRPBLookup(TRPBLookup):
    def _load_data(self) -> Tuple[Dict[str, float], np.ndarray, List[str]]:
        """Create synthetic data for testing instead of downloading."""
        np.random.seed(42)  # For reproducibility
        lookup_dict = {}

        # Add wildtype with high score
        lookup_dict[self.wildtype_sequence] = 0.9

        # Add some random sequences
        for _ in range(100):
            seq = "".join(np.random.choice(self.alphabet, size=self.dim))
            if seq not in lookup_dict:
                lookup_dict[seq] = np.random.uniform(0.0, 1.0)

        # Sort sequences by score
        seqs = list(lookup_dict.keys())
        scores = np.array([lookup_dict[seq] for seq in seqs])
        sorted_indices = np.argsort(scores)[::-1]
        sorted_scores = scores[sorted_indices]
        sorted_seqs = [seqs[i] for i in sorted_indices]

        return lookup_dict, sorted_scores, sorted_seqs


class TestTRPBLookup:
    @pytest.fixture
    def trpb_function(self):
        """Create a TRPBLookup instance for testing."""
        # Use mock implementation in CI
        is_ci = os.environ.get("CI", "false").lower() == "true"
        if is_ci:
            return MockTRPBLookup()
        # Use real implementation locally
        return TRPBLookup()

    def test_initialization(self, trpb_function):
        """Test that the function initializes properly."""
        assert trpb_function.dim == 4
        assert trpb_function.alphabet_size == 20
        assert "".join(trpb_function.alphabet) == "ARNDCQEGHILKMFPSTWYV"
        assert len(trpb_function._lookup_dict) > 0
        assert trpb_function._optimal_value is not None
        assert len(trpb_function._optimizers) > 0

    def test_evaluate_true(self, trpb_function):
        """Test function evaluation."""
        # Create a batch of random inputs
        batch_size = 5
        X = trpb_function.random_solution(batch_size)

        # Evaluate
        y = trpb_function.evaluate_true(X)

        # Check output shape
        assert y.shape == (batch_size,)

        # Check output type
        assert y.dtype == torch.float

        # Test with 3D input
        X_3d = X.unsqueeze(1)  # (batch_size, dim) -> (batch_size, 1, dim)
        y_3d = trpb_function.evaluate_true(X_3d)
        assert y_3d.shape == (batch_size,)

    def test_random_solution(self, trpb_function):
        """Test random solution generation."""
        # Single solution
        x_single = trpb_function.random_solution()
        assert x_single.shape == (trpb_function.dim,)
        assert torch.all((x_single >= 0) & (x_single < trpb_function.alphabet_size))

        # Multiple solutions
        n = 10
        x_multiple = trpb_function.random_solution(n)
        assert x_multiple.shape == (n, trpb_function.dim)
        assert torch.all((x_multiple >= 0) & (x_multiple < trpb_function.alphabet_size))

    def test_initial_solution(self, trpb_function):
        """Test initial solution generation."""
        # Single solution
        x_single = trpb_function.initial_solution()
        assert x_single.shape == (trpb_function.dim,)
        assert torch.all((x_single >= 0) & (x_single < trpb_function.alphabet_size))

        # Multiple solutions
        n = 10
        x_multiple = trpb_function.initial_solution(n)
        assert x_multiple.shape == (n, trpb_function.dim)
        assert torch.all((x_multiple >= 0) & (x_multiple < trpb_function.alphabet_size))

    def test_optimal_solution(self, trpb_function):
        """Test optimal solution retrieval."""
        x_opt = trpb_function.optimal_solution()
        assert x_opt is not None
        assert x_opt.shape == (trpb_function.dim,)

        # The optimal solution should achieve the optimal value
        y_opt = trpb_function.evaluate_true(x_opt.unsqueeze(0))
        assert torch.isclose(y_opt[0], torch.tensor(trpb_function._optimal_value))

    def test_seq_to_tensor_conversion(self, trpb_function):
        """Test conversion between sequence and tensor representations."""
        # Test wildtype sequence conversion
        seq = trpb_function.wildtype_sequence
        tensor = trpb_function._seq_to_tensor(seq)
        assert tensor.shape == (trpb_function.dim,)

        # Convert back to sequence
        seq_back = trpb_function._tensor_to_seq(tensor)
        assert seq_back == seq

    def test_out_of_vocabulary_sequence(self, trpb_function):
        """Test handling of sequences not in the lookup table."""
        # Create a batch with a sequence containing indices that, when converted
        # to a sequence, will likely not be in the lookup table
        X = torch.zeros(1, trpb_function.dim, device=trpb_function._device)
        # Evaluate (should return -inf for unknown sequences)
        y = trpb_function.evaluate_true(X)
        # Either the sequence is in the lookup table (unlikely) or it returns -inf
        if trpb_function._tensor_to_seq(X[0]) not in trpb_function._lookup_dict:
            assert y[0] == -float("inf")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
