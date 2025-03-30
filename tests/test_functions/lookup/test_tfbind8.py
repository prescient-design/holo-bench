import pytest
import torch

from holo.test_functions.lookup import TFBIND8Lookup


class TestTFBIND8Lookup:
    @pytest.fixture
    def tfbind8_function(self):
        """Create a TFBIND8Lookup instance for testing."""
        # Skip download if needed to speed up the test for CI
        return TFBIND8Lookup()

    def test_initialization(self, tfbind8_function):
        """Test that the function initializes properly."""
        assert tfbind8_function.dim == 8
        assert tfbind8_function.num_states == 4
        assert tfbind8_function.alphabet == ["A", "C", "G", "T"]
        assert len(tfbind8_function._lookup_dict) > 0
        assert tfbind8_function._optimal_value is not None
        assert len(tfbind8_function._optimizers) >= 1

    def test_evaluate_true(self, tfbind8_function):
        """Test function evaluation."""
        # Create a batch of random inputs
        batch_size = 5
        X = tfbind8_function.random_solution(batch_size)

        # Evaluate
        y = tfbind8_function.evaluate_true(X)

        # Check output shape
        assert y.shape == (batch_size,)

        # Check output type
        assert y.dtype == torch.float

        # Test with 3D input
        X_3d = X.unsqueeze(1)  # (batch_size, dim) -> (batch_size, 1, dim)
        y_3d = tfbind8_function.evaluate_true(X_3d)
        assert y_3d.shape == (batch_size,)

    def test_random_solution(self, tfbind8_function):
        """Test random solution generation."""
        # Single solution
        x_single = tfbind8_function.random_solution()
        assert x_single.shape == (tfbind8_function.dim,)
        assert torch.all((x_single >= 0) & (x_single < tfbind8_function.num_states))

        # Multiple solutions
        n = 10
        x_multiple = tfbind8_function.random_solution(n)
        assert x_multiple.shape == (n, tfbind8_function.dim)
        assert torch.all((x_multiple >= 0) & (x_multiple < tfbind8_function.num_states))

    def test_initial_solution(self, tfbind8_function):
        """Test initial solution generation."""
        # Single solution
        x_single = tfbind8_function.initial_solution()
        assert x_single.shape == (tfbind8_function.dim,)
        assert torch.all((x_single >= 0) & (x_single < tfbind8_function.num_states))

        # Multiple solutions
        n = 10
        x_multiple = tfbind8_function.initial_solution(n)
        assert x_multiple.shape == (n, tfbind8_function.dim)
        assert torch.all((x_multiple >= 0) & (x_multiple < tfbind8_function.num_states))

    def test_optimal_solution(self, tfbind8_function):
        """Test optimal solution retrieval."""
        x_opt = tfbind8_function.optimal_solution()
        assert x_opt.shape == (tfbind8_function.dim,)

        # The optimal solution should achieve the optimal value
        y_opt = tfbind8_function.evaluate_true(x_opt.unsqueeze(0))
        assert torch.isclose(y_opt[0], torch.tensor(tfbind8_function._optimal_value))

    def test_to_device(self, tfbind8_function):
        """Test moving the function to a device."""
        # This test doesn't actually move to CUDA since it may not be available
        # but it tests the to() method functionality
        func_cpu = tfbind8_function.to(torch.device("cpu"), torch.float32)
        assert func_cpu._device == torch.device("cpu")

        # Generate a solution and check device
        x = func_cpu.random_solution()
        assert x.device == torch.device("cpu")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
