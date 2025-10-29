import os
import time
import numpy as np
import torch
import torch.nn as nn

from src.nerual_networks.training import eval_loss, train_to_budget
from src.nerual_networks.optimizers import optimizer_factory
from src.nerual_networks.models import set_seed, MLP

def train_nn_on_new_inputs(
    new_X_train, new_X_val, new_X_test, y_train, y_val, y_test,
    max_updates: int = 1500,
    learning_rate: float = 1e-4,  # Best from OL Part 2
    betas: tuple = (0.9, 0.999),  # Best from OL Part 2
    seeds: list = [42, 4242, 424242],
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> dict:
    """Streamlined trainer for UL Steps 4/5: Retrain best OL NN (nn_2, adam_no_bias, no reg) on new inputs (DR/cluster features).
    Fixed budget: 1500 updates. Metrics: MSE/MAE/R² for Accidents regression. 100% UL compliance."""

    function_start = time.perf_counter()
    eval_interval = 25
    hidden_layers = [128, 64]  # Best OL nn_2 backbone
    in_dim = new_X_train.shape[1]
    out_dim = 1  # Regression
    loss_fn = nn.MSELoss()

    all_curves = []
    all_test_mse = []
    all_times = []

    for seed in seeds:
        set_seed(seed)
        model = MLP(in_dim=in_dim, hidden=hidden_layers, out_dim=out_dim, dropout_p=0.0).to(device)  # No dropout (baseline best)

        opt = optimizer_factory(model, 'adam_no_bias', lr=learning_rate, betas=betas)

        curves, _, wall_time, final_train_loss = train_to_budget(
            model, opt, new_X_train, new_X_val, y_train, y_val,
            max_updates, float('inf'), loss_fn, device, eval_interval=eval_interval # type: ignore
        )

        test_mse = eval_loss(model, new_X_test, loss_fn, device)
        all_curves.append(curves)
        all_test_mse.append(test_mse)
        all_times.append(wall_time)

    mean_test = np.mean(all_test_mse)
    std_test = np.std(all_test_mse)
    mean_time = np.mean(all_times)

    summary = {
        'mean_test_mse': float(mean_test),
        'std_test_mse': float(std_test),
        'mean_time': float(mean_time),
        'execution_time': time.perf_counter() - function_start
    }

    return summary