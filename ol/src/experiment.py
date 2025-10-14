# -*- coding: utf-8 -*-
import os
from typing import Any, Dict, List, Tuple
import warnings

import numpy as np
import torch
import torch.nn as nn
from joblib import Parallel, delayed

from src.training import eval_loss, train_to_budget
from src.optimizers import optimizer_factory
from src.utils.logger import MLLogger
from src.data_processing import load_or_process_data, wrap_into_loaders
from src.utils.plotter import plot_epoch_curve, plot_heatmaps
from src.models import set_seed, MLP
from src.ro import rhc, sa, ga, validation_objective, get_trainable_params


# Suppress warnings
warnings.filterwarnings("ignore", message="resource_tracker")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.neural_network")


class Experiment:
    def __init__(self, id: int, dataset: str, target: str, method: str, subsample: float, best_params: Dict[str, Any]|None=None):
        """Initialize experiment with dataset config and setup logging/system info."""
        self.experiment_id = id
        self.dataset = dataset
        self.target = target
        self.method = method
        self.subsample = subsample
        self.seed = set_seed()
        self.best_params = best_params or {}
        self.save_path = f"figures/{self.dataset}"  # Base path, subfolders added per part

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.ml_logger = MLLogger()
        self.ml_logger.current_logs = []
        self.update_logs()


    def get_data(self) -> Tuple:
        """Load/process data and wrap into PyTorch DataLoaders with logging."""
        with self.ml_logger.log_step("Load Data") as step_info:
            X_train, X_val, X_test, y_train, y_val, y_test, data_info = load_or_process_data(
                self.dataset, self.target, self.method, self.subsample, self.seed
            )
            step_info.update(data_info)
            train_loader, val_loader, test_loader = wrap_into_loaders(
                self.method, X_train, X_val, X_test, y_train, y_val, y_test
            )
        return train_loader, val_loader, test_loader


    def run_part1_ro(self, max_param: int = 50000, max_evals: int = 11000, batch_size: int = 512, plateau_threshold: int = 250):
        """
        Run Part 1 RO: freeze model, run rhc/sa/ga, log history for analysis, generate report,
        and plot curves including best-so-far objective vs. evals.
        """
        print(f"\n\nExecuting {self.method} neural network on {self.dataset}...".upper())
        part1_path = f"{self.save_path}/part1"
        os.makedirs(part1_path, exist_ok=True)

        train_loader, val_loader, test_loader = self.get_data()

        # Model initialization
        in_dim = train_loader.dataset.tensors[0].shape[1]  # get feature count from first batch
        out_dim = len(torch.unique(train_loader.dataset.tensors[1])) if self.method == 'classification' else 1
        model = MLP(
            in_dim=in_dim,
            hidden=self.best_params.get('hidden_layer_sizes', [128, 64]),
            out_dim=out_dim,
            activation=self.best_params.get('activation', 'relu')
        ).to(self.device)

        # Dynamic layer freezing
        max_k = 3
        selected_k = 1
        for try_k in range(max_k, 0, -1):
            if model.compute_trainable_for_k(try_k) <= max_param:
                selected_k = try_k
                break
        model.freeze_all_but_last_k(k=selected_k, limit=max_param)

        # Loss function setup
        loss_fn = nn.CrossEntropyLoss() if self.method == 'classification' else nn.MSELoss()

        # Run RO algorithms
        test_losses = {}
        for algo in [rhc, sa, ga]:
            optimized_model, history = algo(
                model=model,
                val_loader=val_loader,
                loss_fn=loss_fn,
                device=self.device,
                max_evals=max_evals,
                plateau_threshold=plateau_threshold,
                logger=self.ml_logger
            )
            self.ml_logger.log_step(f"{algo.__name__} RO")

            # Process history
            evals = [e for e, _ in history]
            losses = [l for _, l in history]
            if len(evals) != len(losses) or len(evals) < 2:
                print(f"Skipping plot for {algo.__name__}: insufficient data (evals: {len(evals)}, losses: {len(losses)})")
                continue
            
            best_so_far = [min(losses[:i+1]) for i in range(len(losses))]

            # Log and plot curves
            self.ml_logger.log_learning_curve({
                'algo': algo.__name__, 'evals': evals, 'raw_losses': losses, 'best_so_far': best_so_far
            })
            plot_epoch_curve(
                evals, losses,
                title=f"{algo.__name__} Optimization Curve",
                save_path=f"{part1_path}/{algo.__name__}_optimization.png"
            )
            plot_epoch_curve(
                evals, best_so_far,
                title=f"{algo.__name__} Best-So-Far Objective vs. Evals",
                save_path=f"{part1_path}/{algo.__name__}_best_so_far.png",
                log_x=True
            )

            # Test evaluation
            with self.ml_logger.log_step(f"{algo.__name__} Test Evaluation") as step_info:
                test_loss = validation_objective(
                    get_trainable_params(optimized_model), optimized_model, test_loader, loss_fn, self.device
                )
                step_info.update({'test_loss': test_loss})
                self.ml_logger.log_metric('test_loss', test_loss)
                test_losses[algo.__name__] = test_loss

        # Plot comparison
        if test_losses:
            x_indices = np.arange(len(test_losses))
            self.ml_logger.log_learning_curve({
                'comparison': 'test_losses', 'algorithms': list(test_losses.keys()), 'values': list(test_losses.values())
            })
            plot_epoch_curve(
                x_indices, list(test_losses.values()), # type: ignore
                title="Test Loss Comparison Across RO Algorithms",
                save_path=f"{part1_path}/test_loss_comparison.png",
                log_x=False
            )

        self.ml_logger.generate_log_report(output_file=f"{part1_path}/part1_report.txt")


    def run_part2_ablations(
        self,
        max_updates: int = 10000,  # maximum gradient evaluations per training run
        learning_threshold: float = 0.5,  # target validation loss threshold to stop training early
        learning_rate: float = 1e-3,  # learning rate for all optimizers, fixed
        seeds: List[int] = [4242, 42, 24],  # random seeds for multiple runs
    ):
        """
        Run Part 2 Adam ablations: loop over 7 opts on full network, train to budget,
        analyze speed/stability/gen per LaGrow PDF.
        Multiple seeds for stability bands; logs curves, norms, metrics for tables/plots/heatmaps.
        """
        print(f"\n\nExecuting Part 2 Adam Ablations on {self.dataset}...".upper())
        part2_path = f"{self.save_path}/part2"
        os.makedirs(part2_path, exist_ok=True)

        train_loader, val_loader, test_loader = self.get_data()

        # Setup
        in_dim = train_loader.dataset.tensors[0].shape[1]  # type: ignore
        unique_classes = torch.unique(train_loader.dataset.tensors[1])  # type: ignore
        out_dim = 2 if len(unique_classes) == 2 else len(unique_classes)
        loss_fn = nn.CrossEntropyLoss() if self.method == 'classification' else nn.MSELoss()
        if self.method == 'regression':
            learning_threshold /= 5.0
        log_interval = 100

        kinds = ['sgd', 'sgd_momentum', 'nesterov', 'adam', 'adam_no_bias', 'rmsprop_like', 'adamw']

        all_data: Dict[str, Dict[str, List]] = {
            kind: {'curves': [], 'grad_norms': [], 'steps_to_l': [], 'wall_times': [], 'test_losses': []}
            for kind in kinds
        }

        # Seed loop
        print(f"Starting ablation runs with {len(seeds)} seeds: {seeds}")
        for seed_idx, seed in enumerate(seeds):
            set_seed(seed)
            print(f"Processing seed {seed} ({seed_idx + 1}/{len(seeds)})")

            def run_ablation(kind: str):
                print(f"Running {kind} optimizer")
                model = MLP(
                    in_dim=in_dim,
                    hidden=self.best_params.get('hidden_layer_sizes', [128, 64]),
                    out_dim=out_dim,
                    activation=self.best_params.get('activation', 'relu')
                ).to(self.device)

                kwargs: Dict[str, Any] = {}
                if kind == 'adamw':kwargs['weight_decay'] = 0.01  # decoupled wd for adamw

                opt = optimizer_factory(model, kind, lr=learning_rate, **kwargs)
                if opt is None: raise ValueError(f"Optimizer factory returned None for kind '{kind}'.")

                with self.ml_logger.log_step(f"{kind} Ablation (seed={seed})") as step_info:
                    curves, grad_norms, steps_to_l, wall_time = train_to_budget(
                        model, opt, train_loader, val_loader, max_updates, learning_threshold,
                        loss_fn, self.device, log_interval, optimizer_name=kind
                    )
                    test_loss = eval_loss(model, test_loader, loss_fn, self.device)

                    step_info.update({
                        'steps_to_l': steps_to_l,
                        'wall_time': wall_time,
                        'test_loss': test_loss,
                        'curves': curves,
                        'grad_norms': grad_norms
                    })
                    print(f"Completed {kind} with seed {seed}: steps_to_l={steps_to_l}, test_loss={test_loss:.4f}")

                return kind, curves, grad_norms, steps_to_l, wall_time, test_loss

            # Run sequentially to avoid CUDA issues
            results = []
            for kind in kinds:
                result = run_ablation(kind)
                results.append(result)                
            # results = Parallel(n_jobs=-1, backend='threading')(delayed(run_ablation)(kind) for kind in kinds) # run in parallel using joblib

            # Collect results
            for kind, curves, grad_norms, steps_to_l, wall_time, test_loss in results: # type:ignore
                all_data[kind]['curves'].append(curves)
                all_data[kind]['grad_norms'].append(grad_norms)
                all_data[kind]['steps_to_l'].append(steps_to_l)
                all_data[kind]['wall_times'].append(wall_time)
                all_data[kind]['test_losses'].append(test_loss)

        # Analysis
        print(f"Analyzing results for {len(kinds)} optimizers...")
        summary_table = []
        for kind_idx, kind in enumerate(kinds):
            print(f"Analyzing {kind} ({kind_idx + 1}/{len(kinds)})")
            data = all_data[kind]

            # Speed metric
            avg_steps = float(np.mean(data['steps_to_l']))
            self.ml_logger.log_metric(f"{kind}_avg_steps_to_l", avg_steps)

            # Stability metric
            max_len = max(len(c) for c in data['curves'])
            padded_curves = [c + [c[-1]] * (max_len - len(c)) for c in data['curves']]
            mean_curve = np.mean(padded_curves, axis=0)
            std_curve = list[int | float](np.std(padded_curves, axis=0))
            updates = list[int | float](np.arange(log_interval, (max_len + 1) * log_interval, log_interval))

            # Plot stability
            plot_epoch_curve(
                updates, mean_curve, title=f"{kind} Loss Curve (mean ± std)",
                save_path=f"{part2_path}/{kind}_loss.png", std=std_curve
            )

            # Generalization metric
            avg_test = float(np.mean(data['test_losses']))
            self.ml_logger.log_metric(f"{kind}_avg_test_loss", avg_test)

            # add avg wall time for table
            avg_time = float(np.mean(data['wall_times']))

            # append to table for summary
            summary_table.append((kind, avg_steps, avg_test, avg_time))

            # Heatmap
            layers = list(data['grad_norms'][0][0].keys())
            num_time = len(data['grad_norms'][0])
            norm_matrix = np.zeros((len(seeds), len(layers), num_time))
            for s in range(len(seeds)):
                for t in range(num_time):
                    for l, layer in enumerate(layers):
                        norm_matrix[s, l, t] = data['grad_norms'][s][t].get(layer, 0.0)
            mean_norms = np.mean(norm_matrix, axis=0)
            scaled_mean_norms = np.log10(np.maximum(mean_norms, 1e-10))  # log scale for visibility

            plot_heatmaps(
                scaled_mean_norms, layers, updates, title=f"{kind} Log Grad Norms Heatmap",
                save_path=f"{part2_path}/{kind}_heatmap.png"
            )

            # Calculate dynamic column widths
        max_opt_name = max(len(kind) for kind in kinds)
        max_steps = max(len(f"{steps:>6.0f}") for _, steps, _, _ in summary_table)
        max_test = max(len(f"{test:>7.4f}") for _, _, test, _ in summary_table)
        max_time = max(len(f"{time:>6.1f}") for _, _, _, time in summary_table)
        
        table_str = f"| {'Optimizer':<{max_opt_name}} | {'Speed-to-l':>{max_steps}} | {'% Budget':>{max_steps}} | {'Test Loss':>{max_test}} | {'Avg Time':>{max_time}} |\n"
        table_str += f"|{'-' * (max_opt_name + 2)}|{'-' * (max_steps + 2)}|{'-' * (max_steps + 2)}|{'-' * (max_test + 2)}|{'-' * (max_time + 2)}|\n"
        for kind, steps, test_loss, wall_time in summary_table:
            budget_pct = steps / max_updates * 100
            table_str += f"| {kind:<{max_opt_name}} | {steps:>{max_steps}.0f} | {budget_pct:>{max_steps}.1f}% | {test_loss:>{max_test}.4f} | {wall_time:>{max_time}.1f}s |\n"
        print(table_str)
        
        self.ml_logger.log_metric('part2_table', table_str)
        self.ml_logger.generate_log_report(output_file=f"{part2_path}/part2_report.txt")


    def update_logs(self):
        """Update experiment context in logger for traceability."""
        self.ml_logger.set_experiment_context(
            dataset=self.dataset,
            target=self.target,
            method=self.method,
            subsample=self.subsample,
            seed=self.seed
        )