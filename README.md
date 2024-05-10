## HoloBench
### Benchmarking Discrete Sequence Optimization Algorithms

### Installation

```bash
python -m pip install -r requirements.in
pip install -e .
```

### Usage

```bash
python scripts/benchmark_optimizer.py
```


### Example
```python
import torch
from holo.test_functions.closed_form import Ehrlich
from holo.optim import DiscreteEvolution

test_fn = Ehrlich(negate=True)
print(f"Desired motif: {test_fn.motifs[0]}")
print(f"Desired motif spacing: {test_fn.spacings[0]}")
print(f"Optimal value: {test_fn.optimal_value}")
initial_solution = test_fn.initial_solution()
vocab = list(range(test_fn.num_states))

params = [
    torch.nn.Parameter(
        initial_solution.float(),
    )
]

optimizer = DiscreteEvolution(
    params,
    vocab,
    mutation_prob=1/test_fn.dim,
    recombine_prob=1/test_fn.dim,
    num_particles=1024,
    survival_quantile=0.01
)

print(f"\nInitial solution: {params[0].data}")
print("\nOptimizing...")
for _ in range(4):
    loss = optimizer.step(lambda x: test_fn(x[0]))
    print(f"loss: {loss}")
print(f"\nFinal solution: {params[0].data}")
```

### Unit Tests

```bash
python -m pytest tests/
```
