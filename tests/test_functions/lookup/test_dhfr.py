import pytest
import torch

from holo.test_functions.lookup import DHFRLookup


class TestDHFRLookup:
    @pytest.fixture
    def dhfr_function(self):
        """Create a DHFRLookup instance for testing."""
        # Skip download if needed to speed up the test for CI
        return DHFRLookup()

    def test_initialization(self, dhfr_function):
        """Test that the function initializes properly."""
        assert dhfr_function.dim == 9
        assert dhfr_function.alphabet_size == 4
        assert dhfr_function.dna_alphabet == "ACGT"
        assert len(dhfr_function._lookup_dict) > 0
        assert dhfr_function._optimal_value is not None
        assert len(dhfr_function._optimizers) > 0

    def test_evaluate_true(self, dhfr_function):
        """Test function evaluation."""
        # Create a batch of random inputs
        batch_size = 5
        X = dhfr_function.random_solution(batch_size)

        # Evaluate
        y = dhfr_function.evaluate_true(X)

        # Check output shape
        assert y.shape == (batch_size,)

        # Check output type
        assert y.dtype == torch.float

        # Test with 3D input
        X_3d = X.unsqueeze(1)  # (batch_size, dim) -> (batch_size, 1, dim)
        y_3d = dhfr_function.evaluate_true(X_3d)
        assert y_3d.shape == (batch_size,)

    def test_random_solution(self, dhfr_function):
        """Test random solution generation."""
        # Single solution
        x_single = dhfr_function.random_solution()
        assert x_single.shape == (dhfr_function.dim,)
        assert torch.all((x_single >= 0) & (x_single < dhfr_function.alphabet_size))

        # Multiple solutions
        n = 10
        x_multiple = dhfr_function.random_solution(n)
        assert x_multiple.shape == (n, dhfr_function.dim)
        assert torch.all((x_multiple >= 0) & (x_multiple < dhfr_function.alphabet_size))

    def test_initial_solution(self, dhfr_function):
        """Test initial solution generation."""
        # Single solution
        x_single = dhfr_function.initial_solution()
        assert x_single.shape == (dhfr_function.dim,)
        assert torch.all((x_single >= 0) & (x_single < dhfr_function.alphabet_size))

        # Multiple solutions
        n = 10
        x_multiple = dhfr_function.initial_solution(n)
        assert x_multiple.shape == (n, dhfr_function.dim)
        assert torch.all((x_multiple >= 0) & (x_multiple < dhfr_function.alphabet_size))

    def test_optimal_solution(self, dhfr_function):
        """Test optimal solution retrieval."""
        x_opt = dhfr_function.optimal_solution()
        assert x_opt is not None
        assert x_opt.shape == (dhfr_function.dim,)

        # The optimal solution should achieve the optimal value
        y_opt = dhfr_function.evaluate_true(x_opt.unsqueeze(0))
        assert torch.isclose(y_opt[0], torch.tensor(dhfr_function._optimal_value))

    def test_seq_to_tensor_conversion(self, dhfr_function):
        """Test conversion between sequence and tensor representations."""
        # Test wildtype sequence conversion
        seq = dhfr_function.wildtype_sequence
        tensor = dhfr_function._seq_to_tensor(seq)
        assert tensor.shape == (dhfr_function.dim,)

        # Convert back to sequence
        seq_back = dhfr_function._tensor_to_seq(tensor)
        assert seq_back == seq

    def test_out_of_vocabulary_sequence(self, dhfr_function):
        """Test handling of sequences not in the lookup table."""
        # Create a batch with a sequence containing indices that, when converted
        # to a sequence, will likely not be in the lookup table
        X = torch.zeros(1, dhfr_function.dim, device=dhfr_function._device)
        # Evaluate (should return -inf for unknown sequences)
        y = dhfr_function.evaluate_true(X)
        # Either the sequence is in the lookup table (unlikely) or it returns -inf
        if dhfr_function._tensor_to_seq(X[0]) not in dhfr_function._lookup_dict:
            assert y[0] == -float("inf")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
