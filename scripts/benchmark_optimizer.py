import random

import hydra
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf

from holo.logging import wandb_setup


@hydra.main(config_path="../config/hydra", config_name="benchmark_optimizer")
def main(cfg):
    if cfg.optimizer.mutation_prob is None:
        cfg.optimizer.mutation_prob = 1.1 / cfg.test_function.dim
    if cfg.optimizer.recombine_prob is None:
        cfg.optimizer.recombine_prob = 1.1 / cfg.test_function.dim

    wandb_setup(cfg)
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    print(OmegaConf.to_yaml(cfg, resolve=True))

    dtype = torch.double if cfg.dtype == "float64" else torch.float
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_function = hydra.utils.instantiate(cfg.test_function)
    test_function = test_function.to(device, dtype)

    initial_solution = test_function.initial_solution().to(dtype)
    params = [torch.nn.Parameter(initial_solution)]

    print(f"Test function: {test_function}")
    print(f"Known optimal solution: {test_function.optimal_solution()}")

    def closure(param_list):
        return test_function(param_list[0])

    vocab = list(range(test_function.num_states))
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=params, vocab=vocab)

    print(f"Searching for solution with optimal value {test_function.optimal_value}...")
    cumulative_regret = 0.0
    best_loss = float("inf")
    for t_idx in range(cfg.num_opt_steps):
        loss = optimizer.step(closure)
        if loss < best_loss:
            best_loss = loss.item()
            best_params = [p.data.clone() for p in params]

        simple_regret_best = best_loss - test_function.optimal_value
        simple_regret_last = loss - test_function.optimal_value
        cumulative_regret += best_loss - test_function.optimal_value
        # frac_particles_feasible = optimizer.particle_loss.gt(-float("inf")).float().mean().item()
        frac_particles_feasible = optimizer.particle_loss.lt(float("inf")).float().mean().item()

        metrics = {
            "simple_regret_best": simple_regret_best,
            "simple_regret_last": simple_regret_last,
            "cumulative_regret": cumulative_regret,
            "frac_particles_feasible": frac_particles_feasible,
            "timestep": t_idx,
        }

        stop = simple_regret_best == 0
        if t_idx % cfg.log_interval == 0 or stop:
            wandb.log(metrics)
            print(f"Step {t_idx}: Loss {loss}")

        if stop:
            break
    print(f"Best solution: {best_params[0].long()}")


if __name__ == "__main__":
    main()
