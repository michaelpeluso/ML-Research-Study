import os
from typing import Any
import warnings

import numpy as np

from src.training import eval_loss, train_to_budget
warnings.filterwarnings("ignore", message="resource_tracker")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.neural_network")  # Suppress NN batch_size/convergence warnings

from src.optimizers import optimizer_factory
from src.utils.logger import MLLogger
from src.data_processing import load_or_process_data, wrap_into_loaders
from src.utils.plotter import plot_complexity_curve, plot_epoch_curve, plot_feature_importance, plot_model_evaluation, plot_pruning_path
from src.models import set_seed, MLP
from src.ro import rhc, sa, ga, validation_objective, get_trainable_params
import torch
import torch.nn as nn

class Experiment:
    def __init__(self, id, dataset, target, method, subsample, best_params={}):
        self.experiment_id = id
        self.dataset = dataset
        self.target = target
        self.method = method
        self.subsample = subsample
        self.seed = set_seed()
        self.best_params = best_params
        self.save_path = f"figures/{self.dataset}"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.ml_logger = MLLogger()
        self.ml_logger.current_logs = []
        self.update_logs()

    # Load or process data
    def get_data(self):
        with self.ml_logger.log_step("Load Data") as step_info:
            X_train, X_val, X_test, y_train, y_val, y_test, data_info = load_or_process_data(self.dataset, self.target, self.method, self.subsample, self.seed)
            step_info.update(data_info)  # type: ignore
            # convert to PyTorch tensors and wrap on DataLoader
            train_loader, val_loader, test_loader = wrap_into_loaders(self.method, X_train, X_val, X_test, y_train, y_val, y_test)
        return train_loader, val_loader, test_loader

    def run_part1_ro(self, max_param=50000, max_evals=11000, plateau_threshold=250):
        '''
        Run Part 1 RO: freeze model, run rhc/sa/ga, log history for analysis, generate report, and plot curves including best-so-far objective vs. evals.
        '''
        print(f"\n\nExecuting {self.method} neural network on {self.dataset}...".upper())

        train_loader, val_loader, test_loader = self.get_data()

        # initialize model with dynamic in_dim
        in_dim = train_loader.dataset.tensors[0].shape[1]  # type: ignore # get feature count from first batch
        out_dim = len(torch.unique(train_loader.dataset.tensors[1])) if self.method == 'classification' else 1 # type: ignore
        model = MLP(
            in_dim=in_dim, 
            hidden=self.best_params.get('hidden_layer_sizes', [128, 64]), 
            out_dim=out_dim,
            activation=self.best_params.get('activation', 'relu')
        ).to(self.device)

        # dynamically select max k (1-3) s.t. trainable <=50k
        max_k = 3
        selected_k = 1  # default min
        for try_k in range(max_k, 0, -1):
            if model.compute_trainable_for_k(try_k) <= max_param:
                selected_k = try_k
                break
        # freeze with selected k (<=50k params)
        model.freeze_all_but_last_k(k=selected_k, limit=max_param)  # now asserts ok

        # loss fn per task
        loss_fn = nn.CrossEntropyLoss() if self.method == 'classification' else nn.MSELoss()

        # run algorithms, log history, and plot curves
        test_losses = {}  # to store test_loss per algorithm
        #for algo in [rhc, sa, ga]:
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
            self.ml_logger.log_step(f"{algo.__name__} RO")  # log for plots

            # extract evals and losses, skip if short
            evals = [e for e, _ in history]
            losses = [l for _, l in history]
            if len(evals) != len(losses) or len(evals) < 2:
                print(f"Skipping plot for {algo.__name__}: insufficient data (evals: {len(evals)}, losses: {len(losses)})")
                continue

            # compute best-so-far objective
            best_so_far = [min(losses[:i+1]) for i in range(len(losses))]  # best-so-far objective

            # log curve data for report/analysis (e.g., tails/minima)
            self.ml_logger.log_learning_curve({'algo': algo.__name__, 'evals': evals, 'raw_losses': losses, 'best_so_far': best_so_far})

            # plot optimization curve (raw losses)
            os.makedirs(self.save_path, exist_ok=True)
            plot_epoch_curve(evals, losses, title=f"{algo.__name__} Optimization Curve", save_path=f"{self.save_path}/{algo.__name__}_curve.png")

            # plot best-so-far objective vs. evals with log-x
            plot_epoch_curve(evals, best_so_far, title=f"{algo.__name__} Best-So-Far Objective vs. Evals", 
                           save_path=f"{self.save_path}/{algo.__name__}_best_so_far.png", log_x=True)

            # evaluate on test for generalization
            with self.ml_logger.log_step(f"{algo.__name__} Test Evaluation") as step_info:
                test_loss = validation_objective(get_trainable_params(optimized_model), optimized_model, test_loader, loss_fn, self.device)
                step_info.update({'test_loss': test_loss})
                self.ml_logger.log_metric('test_loss', test_loss)
                test_losses[algo.__name__] = test_loss

        # plot test loss comparison as a curve
        if test_losses:
            os.makedirs(self.save_path, exist_ok=True)
            x_indices = np.arange(len(test_losses))  # Dummy x-axis for each algorithm
            self.ml_logger.log_learning_curve({'comparison': 'test_losses', 'algorithms': list(test_losses.keys()), 'values': list(test_losses.values())})
            plot_epoch_curve(x_indices, list(test_losses.values()), title="Test Loss Comparison Across RO Algorithms",  # type: ignore
                           save_path=f"{self.save_path}/test_loss_comparison.png", log_x=False)

        # generate execution report after all RO steps
        self.ml_logger.generate_log_report(output_file=f"{self.save_path}/part1_report.txt")
        

    # run additional optimizers
    def run_part2_ablations(self, 
        max_updates: int = 10000,  # maximum gradient evaluations per training run
        learning_threshold: float = 0.5,  # target validation loss threshold to stop training early
        learning_rate: float = 1e-3,  # learning rate for all optimizers, fixed
        seeds: list[int] = [4242, 42, 24],  # random seeds for multiple runs
    ):
        '''
        run part 2 adam ablations: loop over 7 opts on full network, train to budget, analyze speed/stability/gen per lagrow pdf.
        multiple seeds for stability bands; logs curves, norms, metrics for tables/plots/heatmaps.
        '''
        print(f"\n\nExecuting Part 2 Adam Ablations on {self.dataset}...".upper())

        train_loader, val_loader, test_loader = self.get_data()

        in_dim = train_loader.dataset.tensors[0].shape[1]  # type: ignore
        out_dim = len(torch.unique(train_loader.dataset.tensors[1])) if self.method == 'classification' else 1  # type: ignore
        loss_fn = nn.CrossEntropyLoss() if self.method == 'classification' else nn.MSELoss()
        if self.method == 'regression': learning_threshold /= 5.0
        log_interval = 100

        kinds = ['sgd', 'sgd_momentum', 'nesterov', 'adam', 'adam_no_bias', 'rmsprop_like', 'adamw']

        all_data: dict[str, dict[str, list]] = {kind: {'curves': [], 'grad_norms': [], 'steps_to_l': [], 'wall_times': [], 'test_losses': []} for kind in kinds}

        # log progress at start of seed loop
        print(f"Starting ablation runs with {len(seeds)} seeds: {seeds}")
        for seed_idx, seed in enumerate(seeds):
            set_seed(seed)
            print(f"Processing seed {seed} ({seed_idx + 1}/{len(seeds)})")

            # define inner function for parallel
            def run_ablation(kind: str):
                print(f"Running {kind} optimizer in parallel process")
                model = MLP(
                    in_dim=in_dim,
                    hidden=self.best_params.get('hidden_layer_sizes', [128, 64]),
                    out_dim=out_dim,
                    activation=self.best_params.get('activation', 'relu')
                ).to(self.device)

                kwargs: dict[str, Any] = {}
                if kind == 'adamw':
                    kwargs['weight_decay'] = 0.01  # decoupled wd for adamw per lagrow

                opt = optimizer_factory(model, kind, lr=learning_rate, **kwargs)

                with self.ml_logger.log_step(f"{kind} Ablation (seed={seed})") as step_info:
                    curves, grad_norms, steps_to_l, wall_time = train_to_budget(
                        model, opt, train_loader, val_loader, max_updates, learning_threshold, loss_fn, self.device, log_interval, optimizer_name=kind #type: ignore
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

            # run in parallel using joblib
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=-1)(delayed(run_ablation)(kind) for kind in kinds)

            # collect results
            for kind, curves, grad_norms, steps_to_l, wall_time, test_loss in results: # type:ignore
                all_data[kind]['curves'].append(curves)
                all_data[kind]['grad_norms'].append(grad_norms)
                all_data[kind]['steps_to_l'].append(steps_to_l)
                all_data[kind]['wall_times'].append(wall_time)
                all_data[kind]['test_losses'].append(test_loss)

        # log progress at start of analysis
        print(f"Analyzing results for {len(kinds)} optimizers...")
        # analyze: average over seeds for metrics/tables and plot
        os.makedirs(self.save_path, exist_ok=True)
        for kind_idx, kind in enumerate(kinds):
            print(f"Analyzing {kind} ({kind_idx + 1}/{len(kinds)})")
            data = all_data[kind]

            # speed: avg steps_to_l
            avg_steps = float(np.mean(data['steps_to_l']))
            self.ml_logger.log_metric(f"{kind}_avg_steps_to_l", avg_steps)

            # stability: curves mean/std for bands
            max_len = max(len(c) for c in data['curves'])
            padded_curves = [c + [c[-1]] * (max_len - len(c)) for c in data['curves']]  # pad with last
            mean_curve = np.mean(padded_curves, axis=0)
            std_curve = list[int|float](np.std(padded_curves, axis=0))
            updates = list[int|float](np.arange(log_interval, (max_len + 1) * log_interval, log_interval))

            # plot mean curve with stability bands
            plot_epoch_curve(updates, mean_curve, title=f"{kind} Loss Curve (mean ± std)", 
                        save_path=f"{self.save_path}/{kind}_curve.png", std=std_curve)

            # generalization: avg test_loss
            avg_test = float(np.mean(data['test_losses']))
            self.ml_logger.log_metric(f"{kind}_avg_test_loss", avg_test)

            # heatmaps: avg grad norms over seeds/time
            layers = list(data['grad_norms'][0][0].keys())  # assume same
            num_time = len(data['grad_norms'][0])
            norm_matrix = np.zeros((len(seeds), len(layers), num_time))
            for s in range(len(seeds)):
                for t in range(num_time):
                    for l, layer in enumerate(layers):
                        norm_matrix[s, l, t] = data['grad_norms'][s][t].get(layer, 0.0)
            mean_norms = np.mean(norm_matrix, axis=0)  # layers x time

            # plot heatmap (assumes plotter.py has plot_heatmaps)
            from src.utils.plotter import plot_heatmaps
            plot_heatmaps(mean_norms, layers, updates, title=f"{kind} Grad Norms Heatmap", 
                        save_path=f"{self.save_path}/{kind}_heatmap.png")

            # tables: generate table data
            table_str = "\n| Optimizer | Speed-to-ℓ | Test Metric |\n|-----------|------------|-------------|\n"
            for kind in kinds:
                table_str += f"| {kind} | {all_data[kind]['steps_to_l']:>10.2f} | {all_data[kind]['test_losses']:>11.4f} |\n"
            self.ml_logger.log_metric('part2_table', table_str) #type:ignore

            self.ml_logger.generate_log_report(output_file=f"{self.save_path}/part2_report.txt")

        self.ml_logger.generate_log_report(output_file=f"{self.save_path}/part2_report.txt")

    def update_logs(self):
        self.ml_logger.set_experiment_context(
            dataset=self.dataset,
            target=self.target,
            method=self.method,
            subsample=self.subsample,
            seed=self.seed
        )