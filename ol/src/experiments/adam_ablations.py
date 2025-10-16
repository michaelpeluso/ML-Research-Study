import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from core.training import eval_loss, train_to_budget
from core.optimizers import optimizer_factory
from utils.plotter import plot_curve, plot_heatmaps, stitch_images, plot_combined_heatmap
from core.models import set_seed, MLP

def ablations(
    self,
    max_updates: int = 10000,
    learning_threshold: float = 0.5,
    learning_rate: float = 0.01,
    seeds: List[int] = [42, 4242, 424242]
):
    '''
    Run optimizer ablations with sensitivity analysis for Part 2.
    Measures speed-to-target (steps_to_l), stability (curves/std), and generalization (test_metrics/gen_gap).
    Generates sensitivity data for Adam-family and combines into a single heatmap per dataset.
    Plots cumulative curves with baseline and optimized hypers.
    '''
    # get hidden layer config and print experiment details
    hidden_layers = self.best_params.get('hidden_layer_sizes', [128, 64])
    print(f"\n\nExecuting Optimizer Ablations...\nMethod: {self.method}\nDataset: {self.dataset}\nNetwork: {hidden_layers}\n".upper())
    part2_path = f"{self.save_path}/{os.path.splitext(os.path.basename(__file__))[0]}"
    os.makedirs(part2_path, exist_ok=True)

    # load data and setup model
    train_loader, val_loader, test_loader = self.get_data()
    in_dim = train_loader.dataset.tensors[0].shape[1]
    out_dim = len(torch.unique(train_loader.dataset.tensors[1])) if self.method == 'classification' else 1
    model = MLP(
        in_dim=in_dim,
        hidden=self.best_params.get('hidden_layer_sizes', [128, 64]),
        out_dim=out_dim,
        activation=self.best_params.get('activation', 'relu')
    ).to(self.device)

    # define loss function based on method
    loss_fn = nn.CrossEntropyLoss() if self.method == 'classification' else nn.MSELoss()

    # ablation kinds as per report
    kinds = ['sgd', 'sgd_momentum', 'nesterov', 'adam', 'adam_no_bias', 'rmsprop_like', 'adamw']

    # initialize data storage
    all_data: Dict[str, Dict[str, Any]] = {kind: {'curves': [], 'test_metrics': [], 'times_to_l': [], 'grad_evals': [], 'func_evals': [], 'updates': []} for kind in kinds}
    summary_table: List[Tuple[str, float, float, float, float, float, float, float]] = []  # (kind, avg_steps, std_steps, avg_test, std_test, gen_gap, avg_time, std_time)

    # helper function to run a single seed experiment
    def run_seed(kind: str, seed: int, **opt_kwargs) -> Optional[Tuple[List[float], float, int, float, float]]:
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
        test_metric = eval_loss(local_model, test_loader, loss_fn, self.device)
        return curves, test_metric, steps_to_l, wall_time, final_train_loss
    
    alpha_grid = [0.0, 1e-5, 1e-4, 1e-3]  # coarse grid as suggested
    beta1_grid = [0.9, 0.99, 0.999, 0.9999]
    beta2_grid = [0.9, 0.99, 0.999, 0.9999]
    default_b2 = 0.999  # fix for alpha vs beta1
    default_b1 = 0.9    # fix for alpha vs beta2
    data_dict = {}
    seed = seeds[0]
    sens_alpha_b1 = np.zeros((len(beta1_grid), len(alpha_grid)))
    sens_alpha_b2 = np.zeros((len(beta2_grid), len(alpha_grid)))

    # run baseline ablations over multiple seeds
    for kind in kinds:
        print(f"Running ablation for {kind}")
        results = [run_seed(kind, seed, lr=learning_rate, betas=(0.9, 0.999)) for seed in seeds]
        valid_results = [res for res in results if res]
        if not valid_results:
            continue

        # extract results
        curves_list = [res[0] for res in valid_results]
        test_metrics = [res[1] for res in valid_results]
        times_to_l = [res[3] for res in valid_results]
        steps_to_l = [res[2] for res in valid_results]

        # store for analysis
        all_data[kind]['curves'] = curves_list
        all_data[kind]['test_metrics'] = test_metrics
        all_data[kind]['times_to_l'] = times_to_l
        all_data[kind]['updates'] = steps_to_l

        # compute metrics
        avg_steps = np.mean(steps_to_l)
        std_steps = np.std(steps_to_l)
        avg_test = np.mean(test_metrics)
        std_test = np.std(test_metrics)
        gen_gap = np.mean([c[-1] - eval_loss(model, train_loader, loss_fn, self.device) for c in curves_list])
        avg_time = np.mean(times_to_l)
        std_time = np.std(times_to_l)
        summary_table.append((kind, avg_steps, std_steps, avg_test, std_test, gen_gap, avg_time, std_time)) #type:ignore
        self.ml_logger.log_metric(f'avg_steps_to_l_{kind}', avg_steps)
        self.ml_logger.log_metric(f'avg_test_metric_{kind}', avg_test)

        # plot mean curve with std band
        mean_curves = np.mean(all_data[kind]['curves'], axis=0)
        std_curves = np.std(all_data[kind]['curves'], axis=0)
        log_interval = 100  # assuming fixed per style; adjust if dynamic
        all_data[kind]['updates'] = np.arange(1, len(mean_curves) + 1) * log_interval
    
        plot_curve(
            x=all_data[kind]['updates'],
            y_list=[mean_curves],
            std=std_curves,
            xlabel="Updates",
            ylabel="Validation Loss",
            title=f"'{kind}' Ablation Mean Curve on {self.dataset.title()} dataset (hidden: {hidden_layers})",
            save_path=f"{part2_path}/curve_{kind}.png",
            band_label="± Std Dev"
        )

        # sensitivity analysis for adam variants with single seed
        if kind in ['adam', 'adam_no_bias', 'adamw', 'rmsprop_like']:
            
            # alpha vs beta1 sensitivity
            if kind != 'rmsprop_like':
                for i, b1 in enumerate(beta1_grid):
                    for j, alpha in enumerate(alpha_grid):
                        results = [run_seed(kind, seed, lr=learning_rate, betas=(b1, default_b2), weight_decay=alpha)]
                        if results[0]:
                            curves = results[0][0]
                            final_vals = curves[-5:] if len(curves) >= 5 else [curves[-1]]
                            sens_alpha_b1[i, j] = np.mean(final_vals)
            plot_heatmaps(
                sens_alpha_b1,
                beta1_grid,
                alpha_grid,
                title=f"'{kind}' Val Loss (α vs β1) on {self.dataset.title()} dataset (hidden: {hidden_layers})",
                save_path=f"{part2_path}/{kind}_val_loss_alpha_b1_heatmap.png"
            )

            # alpha vs beta2 sensitivity (all variants)
            for i, b2 in enumerate(beta2_grid):
                for j, alpha in enumerate(alpha_grid):
                    results = [run_seed(kind, seed, lr=learning_rate, betas=(default_b1, b2), weight_decay=alpha)]
                    if results[0]:
                        curves = results[0][0]
                        final_vals = curves[-5:] if len(curves) >= 5 else [curves[-1]]
                        sens_alpha_b2[i, j] = np.mean(final_vals)
                plot_heatmaps(
                    sens_alpha_b2,
                    beta2_grid,
                    alpha_grid,
                    title=f"'{kind}' Val Loss (α vs β2) on {self.dataset.title()} dataset (hidden: {hidden_layers})",
                    save_path=f"{part2_path}/{kind}_val_loss_alpha_b2_heatmap.png"
                )

            # collect data for combined heatmap (pad rmsprop_like beta1 with zeros for consistent shape)
            data_dict = {}
            for variant in ['adam', 'adam_no_bias', 'adamw', 'rmsprop_like']:
                sens_alpha_b1_var = sens_alpha_b1 if variant != 'rmsprop_like' else np.zeros((len(beta2_grid), len(alpha_grid)))  # pad with zeros
                data_dict[variant] = np.hstack([sens_alpha_b1_var, sens_alpha_b2])  # (4,8) for all

    # plot combined heatmap per dataset
    plot_combined_heatmap(
        data_dict=data_dict,
        alpha_grid=alpha_grid,
        beta1_grid=beta1_grid,
        beta2_grid=beta2_grid,
        optimizers=['adam', 'adam_no_bias', 'adamw', 'rmsprop_like'],
        title=f"Combined Sensitivity Heatmap (Val Loss) on {self.dataset.title()} dataset (hidden: {hidden_layers})",
        save_path=f"{part2_path}/combined_heatmap.png"
    )

    # re-run baselines with best alpha/β1/β2 from sensitivity (for optimized cumulative curves)
    # find best hypers per optimizer (min mean val_loss from grids)
    best_hypers = {}
    for variant in ['adam', 'adam_no_bias', 'adamw', 'rmsprop_like']:
        b1_grid = beta1_grid if variant != 'rmsprop_like' else [0.0]
        b2_grid = beta2_grid
        best_val = float('inf')
        best_alpha, best_b1, best_b2 = alpha_grid[0], b1_grid[0], b2_grid[0]
        # check alpha vs b1
        if variant != 'rmsprop_like':
            min_idx_b1 = np.unravel_index(np.argmin(sens_alpha_b1), sens_alpha_b1.shape)
            if sens_alpha_b1[min_idx_b1] < best_val:
                best_val = sens_alpha_b1[min_idx_b1]
                best_alpha = alpha_grid[min_idx_b1[1]]
                best_b1 = b1_grid[min_idx_b1[0]]
                best_b2 = default_b2
        # check alpha vs b2
        min_idx_b2 = np.unravel_index(np.argmin(sens_alpha_b2), sens_alpha_b2.shape)
        if sens_alpha_b2[min_idx_b2] < best_val:
            best_val = sens_alpha_b2[min_idx_b2]
            best_alpha = alpha_grid[min_idx_b2[1]]
            best_b1 = default_b1
            best_b2 = b2_grid[min_idx_b2[0]]
        best_hypers[variant] = (best_alpha, best_b1, best_b2)

    # run optimized baselines
    optimized_data: Dict[str, Dict[str, Any]] = {kind: {'curves': [], 'updates': []} for kind in kinds}
    for kind in kinds:
        alpha, b1, b2 = best_hypers.get(kind, (self.best_params.get('alpha', 0.0), 0.9, 0.999))  # default if not Adam-family
        results = [run_seed(kind, seed, lr=learning_rate, betas=(b1, b2), weight_decay=alpha) for seed in seeds]
        valid_results = [res for res in results if res]
        if valid_results:
            optimized_data[kind]['curves'] = [res[0] for res in valid_results]
            optimized_data[kind]['updates'] = np.arange(1, len(np.mean(optimized_data[kind]['curves'], axis=0)) + 1) * 100

    # plot cumulative curves with optimized hypers
    kinds = list(optimized_data.keys())
    all_updates = [optimized_data[kind]['updates'] for kind in kinds]
    all_mean_curves = [np.median(optimized_data[kind]['curves'], axis=0) for kind in kinds]
    plot_curve(
        x=all_updates,
        y_list=all_mean_curves,
        labels=kinds,
        colors=['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink'],
        xlabel="Updates",
        ylabel="Validation Loss",
        title=f"Optimized Cumulative Optimizer Loss Curves on {self.dataset.title()} dataset (hidden: {hidden_layers})",
        save_path=f"{part2_path}/optimized_cumulative_loss_curves.png"
    )

    # summary table
    max_opt_name = max(len(kind) for kind in kinds)
    max_val_loss = max(len(f"{val:.4f} ± {std:.4f}") for _, val, std, _, _, _, _, _ in summary_table)
    max_test_metric = max(len(f"{test:.4f} ± {std:.4f}") for _, _, _, test, std, _, _, _ in summary_table)
    max_time_to_l = max(len(f"{time:.2f} ± {std:.2f}") for _, _, _, _, _, _, time, std in summary_table)
    max_grad_evals = max(len(f"{grad:.0f} ± {std:.0f}") for _, _, _, _, _, grad, _, std in summary_table)
    max_func_evals = max(len(f"{func:.0f} ± {std:.0f}") for _, _, _, _, func, _, _, std in summary_table)
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
        avg_grad_evals = avg_steps * self.batch_size
        std_grad_evals = std_steps * self.batch_size
        avg_func_evals = avg_steps
        std_func_evals = std_steps
        avg_val_loss = float(np.mean([c[-1] for c in all_data[kind]['curves']]))
        std_val_loss = float(np.std([c[-1] for c in all_data[kind]['curves']]))

        table_str += (f"| {kind:<{max_opt_name}} | {avg_val_loss:.4f} ± {std_val_loss:.4f} | "
                    f"{avg_test:.4f} ± {std_test:.4f} | {avg_time:.2f} ± {std_time:.2f} | "
                    f"{avg_grad_evals:.0f} ± {std_grad_evals:.0f} | {avg_func_evals:.0f} ± {std_func_evals:.0f} | "
                    f"{avg_steps:.0f} ± {std_steps:.0f} |\n")

    print(table_str)
    self.ml_logger.log_metric('part2_table', table_str)
    self.ml_logger.generate_log_report(output_file=f"{part2_path}/part2_report.txt", part=2)