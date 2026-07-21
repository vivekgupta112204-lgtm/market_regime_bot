"""Example injecting custom hyperparameter constraints into Bayesian loops."""

def sweep_optuna(target_metric: str = "sharpe"):
    """Demonstrates narrowing hyperparameter search spaces dynamically."""
    search_space = {
        "n_hidden_states": [2, 3, 4],
        "covariance": ["full", "diag"]
    }
    return search_space
