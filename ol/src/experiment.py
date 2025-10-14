# -*- coding: utf-8 -*-
import os
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
import torch
import torch.nn as nn

from src.training import eval_loss, train_to_budget
from src.optimizers import optimizer_factory
from src.utils.logger import MLLogger
from src.data_processing import load_or_process_data, wrap_into_loaders
from src.utils.plotter import plot_curve, plot_heatmaps, stitch_images
from src.models import set_seed, MLP
from src.ro import rhc, sa, ga, validation_objective, get_trainable_params


# Suppress warnings
warnings.filterwarnings("ignore", message="resource_tracker")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.neural_network")

print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

class Experiment:
    def __init__(self, id: int, dataset: str, target: str, method: str, subsample: float, batch_size: int, best_params: Dict[str, Any]|None=None):
        """Initialize experiment with dataset config and setup logging/system info."""
        self.experiment_id = id
        self.dataset = dataset
        self.target = target
        self.method = method
        self.subsample = subsample
        self.batch_size = batch_size
        self.seed = set_seed()
        self.best_params = best_params or {}
        self.save_path = f"figures/{self.dataset}"  # Base path, subfolders added per part

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                self.method, X_train, X_val, X_test, y_train, y_val, y_test, self.batch_size
            )
        return train_loader, val_loader, test_loader


    def run_part1_ro(self, max_param: int = 50000, max_evals: int = 11000, plateau_threshold: int = 250):
        '''
        Run Part 1 RO: freeze model, run rhc/sa/ga, log history for analysis, generate report,
        and plot curves including best-so-far objective vs. evals.
        '''
        print(f"\n\nExecuting Randomized optimization...\nMethod: {self.method}\nDataset: {self.dataset}\nNetwork: {self.best_params.get('hidden_layer_sizes', [128, 64])}\n".upper())
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
        all_evals = []
        all_raw_losses = []  # collect raw losses per algo for comparison
        all_best_so_far = []  # collect best-so-far per algo for comparison plot
        algo_names = []
        colors_list = ['blue', 'green', 'red']  # per-algo colors for correlation
        for i, algo in enumerate([rhc, sa, ga]):  # run all three RO algos per report
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
            
            best_so_far = np.minimum.accumulate(np.array(losses)).tolist()  # efficient cumulative min

            # Log learning curve
            self.ml_logger.log_learning_curve({
            'algo': algo.__name__, 'evals': evals, 'raw_losses': losses, 'best_so_far': best_so_far
        })

            # Per-algo condensed plot: raw (solid) + best-so-far (dotted) in one graph
            plot_curve(
                x=evals,
                y_list=[losses, best_so_far],
                labels=["Raw Loss", "Best-so-Far"],
                linestyles=['-', '--'],  # changed: dotted for best-so-far
                xlabel="Function Evaluations",
                ylabel="Validation Loss",
                title=f"{algo.__name__} Condensed Curves",
                save_path=f"{part1_path}/{algo.__name__}_condensed.png",
                colors=[colors_list[i], colors_list[i]]  # same color, different style
            )

            # Collect for multi-algo comparisons
            all_evals.append(evals)
            all_raw_losses.append(losses)
            all_best_so_far.append(best_so_far)
            algo_names.append(algo.__name__)

            # Evaluate on test set
            test_loss = validation_objective(get_trainable_params(optimized_model), optimized_model, test_loader, loss_fn, self.device)
            test_losses[algo.__name__] = test_loss
            self.ml_logger.log_metric(f"{algo.__name__}_test_loss", test_loss)

        # combined plot: raw (solid) and best-so-far (dotted) for all algos
        x_combined = []
        y_combined = []
        labels_combined = []
        linestyles_combined = []
        colors_combined = []
        colors_list = ['blue', 'green', 'red']  # per algo
        for i in range(len(algo_names)):
            x_combined.append(all_evals[i])  # same x for raw
            y_combined.append(all_raw_losses[i])
            labels_combined.append(f"{algo_names[i]} Raw")
            linestyles_combined.append('-')  # solid for raw
            colors_combined.append(colors_list[i])

            x_combined.append(all_evals[i])  # same x for best
            y_combined.append(all_best_so_far[i])
            labels_combined.append(f"{algo_names[i]} Best")
            linestyles_combined.append('--')  # dotted for best
            colors_combined.append(colors_list[i])

        plot_curve(
            x_combined, y_combined, labels=labels_combined,
            xlabel="Function Evaluations", ylabel="Validation Loss",
            title="Raw and Best-so-far Objective vs. Evals",
            save_path=f"{part1_path}/combined_ro_curves.png",
            colors=colors_combined, linestyles=linestyles_combined
        )

        # test losses
        for algo_name, test_loss in test_losses.items():
            print(f"{algo_name} Test Loss: {test_loss}")
            self.ml_logger.log_metric(f"{algo_name}_test_loss", test_loss)

        # generate report
        self.ml_logger.generate_log_report(output_file=f"{part1_path}/part1_report.txt", part=1)


    def run_part2_ablations(
        self,
        max_updates: int = 10000,
        learning_threshold: float = 0.5,
        learning_rate: float = 0.01,
        seeds: List[int] = [42, 4242, 424242]
    ):
        '''run optimizer ablations with sensitivity analysis.'''  # existing comment preserved
        print(f"\n\nExecuting Optimizer Ablations...\nMethod: {self.method}\nDataset: {self.dataset}\nNetwork: {self.best_params.get('hidden_layer_sizes', [128, 64])}\n".upper())
        part2_path = f"{self.save_path}/part2"
        os.makedirs(part2_path, exist_ok=True)

        train_loader, val_loader, test_loader = self.get_data()

        # model setup (full network, no freezing)
        in_dim = train_loader.dataset.tensors[0].shape[1]
        out_dim = len(torch.unique(train_loader.dataset.tensors[1])) if self.method == 'classification' else 1
        model = MLP(
            in_dim=in_dim,
            hidden=self.best_params.get('hidden_layer_sizes', [128, 64]),
            out_dim=out_dim,
            activation=self.best_params.get('activation', 'relu')
        ).to(self.device)

        loss_fn = nn.CrossEntropyLoss() if self.method == 'classification' else nn.MSELoss()

        # ablation kinds per pdf: sgd, momentum, nesterov, adam baseline, adam w/o bias, β1=0, adamw
        kinds = ['sgd', 'sgd_momentum', 'nesterov', 'adam', 'adam_no_bias', 'rmsprop_like', 'adamw']

        all_data: Dict[str, Dict[str, Any]] = {kind: {'curves': [], 'test_metrics': [], 'times_to_l': [], 'grad_evals': [], 'func_evals': [], 'updates': []} for kind in kinds}
        summary_table: List[Tuple[str, float, float, float, float, float, float, float]] = []  # (kind, avg_steps, std_steps, avg_test, std_test, gen_gap, avg_time, std_time)

        # local run_seed func (unchanged)
        def run_seed(kind: str, seed: int, **opt_kwargs) -> Optional[Tuple[List[float], float, int, float]]:
            set_seed(seed)
            local_model = MLP(
                in_dim=in_dim,
                hidden=self.best_params.get('hidden_layer_sizes', [128, 64]),
                out_dim=out_dim,
                activation=self.best_params.get('activation', 'relu')
            ).to(self.device)
            opt = optimizer_factory(local_model, kind, **opt_kwargs)
            if opt is None:
                return None
            curves, _, steps_to_l, wall_time, final_train_loss = train_to_budget(
                local_model, opt, train_loader, val_loader, max_updates, learning_threshold, loss_fn, self.device, optimizer_name=kind
            )
            test_metric = eval_loss(local_model, test_loader, loss_fn, self.device)  # test loss as metric
            return curves, test_metric, steps_to_l, wall_time

        for kind in kinds:
            print(f"Running ablation for {kind}")
            # changed: replace parallel with serial list comp for sequential execution
            results = [run_seed(kind, seed, lr=learning_rate, betas=(0.9, 0.999) if 'adam' in kind else None) for seed in seeds]
            # filter valid results
            valid_results = [res for res in results if res]
            if not valid_results:
                continue

            curves_list = [res[0] for res in valid_results]
            test_metrics = [res[1] for res in valid_results]
            times_to_l = [res[3] for res in valid_results]
            steps_to_l = [res[2] for res in valid_results]

            # store for analysis
            all_data[kind]['curves'] = curves_list
            all_data[kind]['test_metrics'] = test_metrics
            all_data[kind]['times_to_l'] = times_to_l
            all_data[kind]['updates'] = steps_to_l  # approx updates as steps_to_l

            # log metrics (unchanged)
            avg_steps = np.mean(steps_to_l)
            std_steps = np.std(steps_to_l)
            avg_test = np.mean(test_metrics)
            std_test = np.std(test_metrics)
            gen_gap = np.mean([c[-1] - eval_loss(model, train_loader, loss_fn, self.device) for c in curves_list])  # val - train gap
            avg_time = np.mean(times_to_l)
            std_time = np.std(times_to_l)
            summary_table.append((kind, avg_steps, std_steps, avg_test, std_test, gen_gap, avg_time, std_time)) # type: ignore
            self.ml_logger.log_metric(f'avg_steps_to_l_{kind}', avg_steps)
            self.ml_logger.log_metric(f'avg_test_metric_{kind}', avg_test)

            # plot mean curve with std band (unchanged except for x alignment)
            mean_curves = np.mean(all_data[kind]['curves'], axis=0)
            std_curves = np.std(all_data[kind]['curves'], axis=0)
            log_interval = self.ml_logger.log_interval  # assuming log_interval is stored here; adjust if needed
            all_data[kind]['updates'] = np.arange(1, len(mean_curves) + 1) * log_interval

            plot_curve(
                x=all_data[kind]['updates'],
                y_list=[mean_curves],
                std=std_curves,
                xlabel="Updates",
                ylabel="Validation Loss",
                title=f"Ablation: {kind}",
                save_path=f"{part2_path}/curve_{kind}.png",
                band_label="± Std Dev"
            )

            '''
            # sensitivity analysis (lr vs hyperparam) with serial execution
            lr_grid = np.logspace(-4, -1, 5)  # example grid
            if kind in ['sgd_momentum', 'nesterov']:
                momentum_grid = np.linspace(0.5, 0.99, 5)
                # changed: replace parallel with serial list comp for sequential execution
                sens_lr_mom = np.zeros((len(lr_grid), len(momentum_grid)))
                for i, lr in enumerate(lr_grid):
                    for j, mom in enumerate(momentum_grid):
                        results = [run_seed(kind, seed, lr=lr, momentum=mom) for seed in seeds]
                        steps_list = [res[2] for res in results if res]
                        sens_lr_mom[i, j] = np.mean(steps_list)
                plot_heatmaps(sens_lr_mom, momentum_grid, lr_grid, title=f"{kind} Sensitivity (LR vs Momentum)", save_path=f"{part2_path}/{kind}_lr_mom_heatmap.png")
                self.ml_logger.log_metric(f"{kind}_sens_lr_mom", sens_lr_mom.tolist())

            elif 'adam' in kind:
                beta1_grid = np.linspace(0.8, 0.99, 5)
                beta2_grid = np.linspace(0.9, 0.999, 5)

                # lr vs beta1 (fix beta2=0.999)
                # changed: replace parallel with serial list comp for sequential execution
                sens_lr_b1 = np.zeros((len(lr_grid), len(beta1_grid)))
                for i, lr in enumerate(lr_grid):
                    for j, b1 in enumerate(beta1_grid):
                        results = [run_seed(kind, seed, lr=lr, betas=(b1, 0.999)) for seed in seeds]
                        steps_list = [res[2] for res in results if res]
                        sens_lr_b1[i, j] = np.mean(steps_list)
                plot_heatmaps(sens_lr_b1, beta1_grid, lr_grid, title=f"{kind} Sensitivity (LR vs Beta1)", save_path=f"{part2_path}/{kind}_lr_b1_heatmap.png")
                self.ml_logger.log_metric(f"{kind}_sens_lr_b1", sens_lr_b1.tolist())

                # lr vs beta2 (fix beta1=0.9)
                # changed: replace parallel with serial list comp for sequential execution
                sens_lr_b2 = np.zeros((len(lr_grid), len(beta2_grid)))
                for i, lr in enumerate(lr_grid):
                    for j, b2 in enumerate(beta2_grid):
                        results = [run_seed(kind, seed, lr=lr, betas=(0.9, b2)) for seed in seeds]
                        steps_list = [res[2] for res in results if res]
                        sens_lr_b2[i, j] = np.mean(steps_list)
                plot_heatmaps(sens_lr_b2, beta2_grid, lr_grid, title=f"{kind} Sensitivity (LR vs Beta2)", save_path=f"{part2_path}/{kind}_lr_b2_heatmap.png")
                self.ml_logger.log_metric(f"{kind}_sens_lr_b2", sens_lr_b2.tolist())
            '''
            
# cumulative plot
        kinds = list(all_data.keys())
        # changed: use per-kind updates to handle potential length differences (robust)
        all_updates = [all_data[kind]['updates'] for kind in kinds]
        all_mean_curves = [np.median(all_data[kind]['curves'], axis=0) for kind in kinds]
        common_labels = kinds
        colors_list = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink']
        plot_curve(
            x=all_updates,  # changed: list of per-kind updates
            y_list=all_mean_curves,
            labels=common_labels,
            colors=colors_list,
            xlabel="Updates",
            ylabel="Validation Loss",
            title="Cumulative Optimizer Loss Curves (Medians)",
            save_path=f"{part2_path}/cumulative_loss_curves.png"
        )

        # summary table with added columns
        max_opt_name = max(len(kind) for kind in kinds)
        max_val_loss = max(len(f"{val:.4f} ± {std:.4f}") for _, val, std, _, _, _, _, _ in summary_table)
        max_test_metric = max(len(f"{test:.4f} ± {std:.4f}") for _, _, _, test, std, _, _, _ in summary_table)
        max_time_to_l = max(len(f"{time:.2f} ± {std:.2f}") for _, _, _, _, _, _, time, std in summary_table)
        max_grad_evals = max(len(f"{grad:.0f} ± {std:.0f}") for _, _, _, _, _, grad, _, std in summary_table)  # assume tracked in train_to_budget
        max_func_evals = max(len(f"{func:.0f} ± {std:.0f}") for _, _, _, _, func, _, _, std in summary_table)  # assume tracked in train_to_budget
        max_updates = max(len(f"{upd:.0f} ± {std:.0f}") for _, _, _, _, _, _, upd, std in summary_table)

        table_str = (f"| {'Optimizer':<{max_opt_name}} | {'Best Val Loss':>{max_val_loss}} | "
                    f"{'Test Metric':>{max_test_metric}} | {'Time to ℓ (s)':>{max_time_to_l}} | "
                    f"{'# Grad Evals':>{max_grad_evals}} | {'# Func Evals':>{max_func_evals}} | "
                    f"{'Updates':>{max_updates}} |\n")
        table_str += (f"|{'-' * (max_opt_name + 2)}|{'-' * (max_val_loss + 2)}|"
                    f"{'-' * (max_test_metric + 2)}|{'-' * (max_time_to_l + 2)}|"
                    f"{'-' * (max_grad_evals + 2)}|{'-' * (max_func_evals + 2)}|"
                    f"{'-' * (max_updates + 2)}|\n")

        for kind, avg_steps, std_steps, avg_test, std_test, gen_gap, avg_time, std_time in summary_table:
            # Approximate gradient and function evals based on updates and RO context
            avg_grad_evals = avg_steps * self.batch_size  # gradient evals per update * batch size
            std_grad_evals = std_steps * self.batch_size  # assume batch size consistent
            avg_func_evals = avg_steps  # for Part 2, func evals ≈ updates (simplified; adjust if RO mixed)
            std_func_evals = std_steps
            avg_val_loss = float(np.mean([c[-1] for c in all_data[kind]['curves']]))  # best val loss at end
            std_val_loss = float(np.std([c[-1] for c in all_data[kind]['curves']]))

            table_str += (f"| {kind:<{max_opt_name}} | {avg_val_loss:.4f} ± {std_val_loss:.4f} | "
                        f"{avg_test:.4f} ± {std_test:.4f} | {avg_time:.2f} ± {std_time:.2f} | "
                        f"{avg_grad_evals:.0f} ± {std_grad_evals:.0f} | {avg_func_evals:.0f} ± {std_func_evals:.0f} | "
                        f"{avg_steps:.0f} ± {std_steps:.0f} |\n")

        print(table_str)
        self.ml_logger.log_metric('part2_table', table_str)
        self.ml_logger.generate_log_report(output_file=f"{part2_path}/part2_report.txt", part=2)


    def update_logs(self):
        """Update experiment context in logger for traceability."""
        self.ml_logger.set_experiment_context(
            dataset=self.dataset,
            target=self.target,
            method=self.method,
            subsample=self.subsample,
            seed=self.seed
        )