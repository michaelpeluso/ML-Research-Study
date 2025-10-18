import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import time

from core.training import eval_loss, train_to_budget
from core.optimizers import optimizer_factory
from utils.plotter import plot_curve, plot_heatmaps, plot_combined_heatmap
from core.models import set_seed, MLP

def adam_ablations(
    self,
    max_updates: int = 10000,
    learning_threshold: float = 0.5,
    learning_rate: float = 0.01,
    seeds: List[int] = [42, 4242, 424242]
) -> Dict[str, Tuple[float, float, float]]:
    """optimizer ablation study with sensitivity analysis for adam-family:
    - measures speed, stability, generalization
    - generates sensitivity heatmaps and combined mosaic
    - plots baseline and optimized learning curves"""
    
    function_start = time.perf_counter()
    
    hidden_layers = self.best_params.get('hidden_layer_sizes', [128, 64])
    print(f"\n\nExecuting Optimizer Ablations...\nMethod: {self.method}\nDataset: {self.dataset}\nNetwork: {hidden_layers}\n".upper())
    part2_path = f"{self.save_path}/{os.path.splitext(os.path.basename(__file__))[0]}"
    os.makedirs(part2_path, exist_ok=True)
    
    # log experiment configuration
    self.ml_logger.log_metric('part2_config_method', self.method)
    self.ml_logger.log_metric('part2_config_dataset', self.dataset)
    self.ml_logger.log_metric('part2_config_hidden_layers', str(hidden_layers))
    self.ml_logger.log_metric('part2_config_max_updates', max_updates)
    self.ml_logger.log_metric('part2_config_learning_threshold', learning_threshold)
    self.ml_logger.log_metric('part2_config_seeds', str(seeds))

    # setup data and model
    train_loader, val_loader, test_loader = self.get_data()
    in_dim = train_loader.dataset.tensors[0].shape[1]
    out_dim = len(torch.unique(train_loader.dataset.tensors[1])) if self.method == 'classification' else 1
    model = MLP(
        in_dim=in_dim,
        hidden=self.best_params.get('hidden_layer_sizes', [128, 64]),
        out_dim=out_dim,
        activation=self.best_params.get('activation', 'relu')
    ).to(self.device)

    loss_fn = nn.CrossEntropyLoss() if self.method == 'classification' else nn.MSELoss()

    # optimizer variants to test
    kinds = ['sgd', 'sgd_momentum', 'nesterov', 'adam', 'adam_no_bias', 'rmsprop_like', 'adamw']

    # storage for metrics and curves
    all_data: Dict[str, Dict[str, Any]] = {kind: {'curves': [], 'test_metrics': [], 'times_to_l': [], 'grad_evals': [], 'func_evals': [], 'updates': []} for kind in kinds}
    summary_table: List[Tuple[str, float, float, float, float, float, float, float]] = []

    learning_rate = self.best_params.get('alpha', learning_rate)

    def run_seed(kind: str, seed: int, **opt_kwargs) -> Optional[Tuple[List[float], float, int, float, float]]:
        """train single model instance with given optimizer"""
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
    
    def plot_optimizer_curves(data_dict, kinds, colors, suffix="", optimized=False):
        """helper to plot individual and combined curves"""
        # individual curves with stability bands
        for i, kind in enumerate(kinds):
            curves_list = data_dict[kind]['curves']
            if not curves_list:
                continue
            mean_curve = np.median(curves_list, axis=0)
            std_curve = np.std(curves_list, axis=0)
            updates = np.arange(1, len(mean_curve) + 1) * 100            
            label = f'{kind} Median (Optimized)' if optimized else f'{kind} Median (Baseline)'
            prefix = "_optimized" if optimized else "_baseline"
            filename = f"{kind}_curve{prefix}.png"
            
            plot_curve(
                x=updates,
                y_list=[mean_curve],
                labels=[label],
                xlabel="Updates",
                ylabel="Validation Loss",
                title=f"{'Optimized ' if optimized else ''}Loss Curve for {kind} on {self.dataset.title()} (hidden: {hidden_layers})",
                save_path=f"{part2_path}/{filename}",
                colors=[colors[i]],
                std=std_curve,
                band_label='± Std Dev'
            )
        
        # combined cumulative curves
        all_mean_curves = [np.median(data_dict[kind]['curves'], axis=0) 
                          for kind in kinds if data_dict[kind]['curves']]
        # create x-axis list based on each curve's length
        all_updates = [np.arange(1, len(curve) + 1) * 100 for curve in all_mean_curves]

        linestyles = ['-',  '-.','--', ':', '-', '--', ':']  # solid, dashed, dashdot, dotted
        title_prefix = "Optimized" if optimized else "Baseline"
        plot_curve(
            x=all_updates,
            y_list=all_mean_curves,
            labels=[k for k in kinds if data_dict[k]['curves']],
            colors=[colors[i] for i, k in enumerate(kinds) if data_dict[k]['curves']],
            linestyles=[linestyles[i] for i, k in enumerate(kinds) if data_dict[k]['curves']],
            xlabel="Updates",
            ylabel="Validation Loss",
            title=f"{title_prefix} Cumulative Optimizer Loss Curves on {self.dataset.title()} (hidden: {hidden_layers})",
            save_path=f"{part2_path}/cumulative_{title_prefix.lower()}_loss_curves_dotted.png"
        )

        plot_curve(
            x=all_updates,
            y_list=all_mean_curves,
            labels=[k for k in kinds if data_dict[k]['curves']],
            colors=[colors[i] for i, k in enumerate(kinds) if data_dict[k]['curves']],
            linestyles=[linestyles[i] for i, k in enumerate(kinds) if data_dict[k]['curves']],
            xlabel="Updates",
            ylabel="Validation Loss",
            title=f"{title_prefix} Cumulative Optimizer Loss Curves on {self.dataset.title()} (hidden: {hidden_layers})",
            save_path=f"{part2_path}/cumulative_{title_prefix.lower()}_loss_curves.png"
        )
    
    # baseline runs: default hyperparameters
    for kind in kinds:
        print(f"Running ablation for {kind}")
        # use optimizer-specific defaults for betas
        if kind == 'rmsprop_like':
            default_betas = (0.0, 0.999)  # rmsprop: no momentum (beta1=0)
        else:
            default_betas = (0.9, 0.999)  # standard adam defaults
        
        results = [run_seed(kind, s, lr=learning_rate, betas=default_betas) for s in seeds]
        valid_results = [res for res in results if res]
        print(f"For {kind}, valid_results count: {len(valid_results)}")
        if not valid_results:
            continue

        curves_list = [res[0] for res in valid_results]
        test_metrics = [res[1] for res in valid_results]
        steps_to_l = [res[2] for res in valid_results]
        times_to_l = [res[3] for res in valid_results]
        final_train_losses = [res[4] for res in valid_results]
        gen_gaps = [test - train for test, train in zip(test_metrics, final_train_losses)]

        # store metrics
        all_data[kind]['curves'] = curves_list
        all_data[kind]['test_metrics'] = test_metrics
        all_data[kind]['times_to_l'] = times_to_l
        all_data[kind]['updates'] = steps_to_l
        all_data[kind]['gen_gaps'] = gen_gaps

        # compute statistics
        avg_steps = np.mean(steps_to_l)
        std_steps = np.std(steps_to_l)
        avg_test = np.mean(test_metrics)
        std_test = np.std(test_metrics)
        avg_time = np.mean(times_to_l)
        std_time = np.std(times_to_l)
        gen_gap = np.mean(gen_gaps)

        summary_table.append((kind, avg_steps, std_steps, avg_test, std_test, gen_gap, avg_time, std_time)) # type: ignore
        
        # log individual optimizer baseline metrics
        self.ml_logger.log_metric(f'{kind}_baseline_avg_steps', avg_steps)
        self.ml_logger.log_metric(f'{kind}_baseline_avg_test', avg_test)
        self.ml_logger.log_metric(f'{kind}_baseline_gen_gap', gen_gap)
        self.ml_logger.log_metric(f'{kind}_baseline_avg_time', avg_time)

    # plot baseline curves
    kinds = list(all_data.keys())
    colors_list = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#ff7f0e', '#17becf', '#000000']
    plot_optimizer_curves(all_data, kinds, colors_list, suffix="", optimized=False)

    # sensitivity analysis: adam-family hyperparameter grids
    alpha_grid = [0.0, 1e-5, 1e-4, 1e-3]
    beta1_grid = [0.9, 0.99, 0.999, 0.9999]
    beta2_grid = [0.9, 0.99, 0.999, 0.9999]
    default_b2 = 0.999
    default_b1 = 0.9
    data_dict = {}
    seed = seeds[0]

    # log sensitivity grid configurations
    self.ml_logger.log_metric('sensitivity_alpha_grid', str(alpha_grid))
    self.ml_logger.log_metric('sensitivity_beta1_grid', str(beta1_grid))
    self.ml_logger.log_metric('sensitivity_beta2_grid', str(beta2_grid))
    self.ml_logger.log_metric('sensitivity_default_b1', default_b1)
    self.ml_logger.log_metric('sensitivity_default_b2', default_b2)

    adam_variants = ['adam', 'adam_no_bias', 'rmsprop_like', 'adamw']
    sens_data = {}
    best_hypers = {}
    
    for variant in adam_variants:
        print(f"Sensitivity for {variant}")
        data_dict[variant] = {
            'alpha_b1': {'sens': np.zeros((len(beta1_grid), len(alpha_grid))), 'gen_gaps': np.zeros((len(beta1_grid), len(alpha_grid)))}, 
            'alpha_b2': {'sens': np.zeros((len(beta2_grid), len(alpha_grid))), 'gen_gaps': np.zeros((len(beta2_grid), len(alpha_grid)))}
        }

        # alpha vs beta1 (skip for rmsprop_like since b1=0)
        if variant != 'rmsprop_like':
            for i, b1 in enumerate(beta1_grid):
                for j, alpha in enumerate(alpha_grid):
                    betas = (b1, default_b2)
                    results = [run_seed(variant, seed, lr=learning_rate, betas=betas, weight_decay=alpha)]
                    valid_results = [res for res in results if res]
                    if valid_results:
                        curves_list = [res[0] for res in valid_results]
                        final_vals = [c[-1] for c in curves_list if c]
                        gen_gaps = [res[1] - res[4] for res in valid_results]
                        data_dict[variant]['alpha_b1']['sens'][i, j] = np.mean(final_vals)
                        data_dict[variant]['alpha_b1']['gen_gaps'][i, j] = np.mean(gen_gaps)
            sens_data[f'{variant}_alpha_b1'] = data_dict[variant]['alpha_b1']['sens']
            plot_heatmaps(
                data_dict[variant]['alpha_b1']['sens'], 
                beta1_grid, 
                alpha_grid, 
                title=f"'{variant}' Val Loss (Alpha vs Beta1) on {self.dataset.title()} dataset (hidden: {hidden_layers})", 
                save_path=f"{part2_path}/{variant}_alpha_b1_heatmap.png"
            )
            self.ml_logger.log_metric(f"{variant}_gen_gap_alpha_b1", np.mean(data_dict[variant]['alpha_b1']['gen_gaps']))

        # alpha vs beta2 (always run, including rmsprop_like)
        for i, b2 in enumerate(beta2_grid):
            for j, alpha in enumerate(alpha_grid):
                if variant == 'rmsprop_like':
                    betas = (0.0, b2)
                else:
                    betas = (default_b1, b2)
                results = [run_seed(variant, seed, lr=learning_rate, betas=betas, weight_decay=alpha)]
                valid_results = [res for res in results if res]
                if valid_results:
                    curves_list = [res[0] for res in valid_results]
                    final_vals = [c[-1] for c in curves_list if c]
                    gen_gaps = [res[1] - res[4] for res in valid_results]
                    data_dict[variant]['alpha_b2']['sens'][i, j] = np.mean(final_vals)
                    data_dict[variant]['alpha_b2']['gen_gaps'][i, j] = np.mean(gen_gaps)
        sens_data[f'{variant}_alpha_b2'] = data_dict[variant]['alpha_b2']['sens']
        plot_heatmaps(
            data_dict[variant]['alpha_b2']['sens'], 
            beta2_grid, 
            alpha_grid, 
            title=f"'{variant}' Val Loss (Alpha vs Beta2) on {self.dataset.title()} dataset (hidden: {hidden_layers})", 
            save_path=f"{part2_path}/{variant}_alpha_b2_heatmap.png"
        )
        self.ml_logger.log_metric(f"{variant}_gen_gap_alpha_b2", np.mean(data_dict[variant]['alpha_b2']['gen_gaps']))

    # combined mosaic heatmap (7 panels)
    plot_combined_heatmap(
        sens_data,
        alpha_grid, beta1_grid, beta2_grid,
        optimizers=['adam', 'adam_no_bias', 'adamw', 'rmsprop_like'],
        title=f"Adam-Family Sensitivity Mosaic on {self.dataset.title()} dataset (hidden: {hidden_layers})",
        save_path=f"{part2_path}/combined_sensitivity_mosaic.png"
    )

    # find best hyperparameters per variant
    for variant in adam_variants:
        best_val = float('inf')
        best_alpha, best_b1, best_b2 = 0.0, default_b1, default_b2
        if variant != 'rmsprop_like':
            min_idx_b1 = np.unravel_index(np.argmin(data_dict[variant]['alpha_b1']['sens']), data_dict[variant]['alpha_b1']['sens'].shape)
            if data_dict[variant]['alpha_b1']['sens'][min_idx_b1] < best_val:
                best_val = data_dict[variant]['alpha_b1']['sens'][min_idx_b1]
                best_alpha = alpha_grid[min_idx_b1[1]]
                best_b1 = beta1_grid[min_idx_b1[0]]
                best_b2 = default_b2
        min_idx_b2 = np.unravel_index(np.argmin(data_dict[variant]['alpha_b2']['sens']), data_dict[variant]['alpha_b2']['sens'].shape)
        if data_dict[variant]['alpha_b2']['sens'][min_idx_b2] < best_val:
            best_val = data_dict[variant]['alpha_b2']['sens'][min_idx_b2]
            best_alpha = alpha_grid[min_idx_b2[1]]
            best_b1 = default_b1 if variant != 'rmsprop_like' else 0.0
            best_b2 = beta2_grid[min_idx_b2[0]]
        best_hypers[variant] = (best_alpha, best_b1, best_b2)

        self.ml_logger.log_metric(f"{variant}_best_hypers", best_hypers[variant])

    # run optimized baselines with best hyperparameters
    optimized_data: Dict[str, Dict[str, Any]] = {kind: {'curves': [], 'updates': [], 'test_metrics': [], 'times': [], 'gen_gaps': []} for kind in kinds}
    for kind in kinds:
        alpha, b1, b2 = best_hypers.get(kind, (self.best_params.get('alpha', 0.0), 0.9, 0.999))
        results = [run_seed(kind, seed, lr=learning_rate, betas=(b1, b2), weight_decay=alpha) for seed in seeds]
        valid_results = [res for res in results if res]
        if valid_results:
            curves_list = [res[0] for res in valid_results]
            test_metrics = [res[1] for res in valid_results]
            times = [res[3] for res in valid_results]
            final_train_losses = [res[4] for res in valid_results]
            gen_gaps = [test - train for test, train in zip(test_metrics, final_train_losses)]
            
            optimized_data[kind]['curves'] = curves_list
            optimized_data[kind]['updates'] = np.arange(1, len(np.mean(curves_list, axis=0)) + 1) * 100
            optimized_data[kind]['test_metrics'] = test_metrics
            optimized_data[kind]['times'] = times
            optimized_data[kind]['gen_gaps'] = gen_gaps
            
            # log optimized metrics
            self.ml_logger.log_metric(f'{kind}_optimized_avg_test', np.mean(test_metrics))
            self.ml_logger.log_metric(f'{kind}_optimized_gen_gap', np.mean(gen_gaps))
            self.ml_logger.log_metric(f'{kind}_optimized_avg_time', np.mean(times))
            self.ml_logger.log_metric(f'{kind}_optimized_hypers', f'alpha={alpha:.2e}, b1={b1}, b2={b2}')

    # plot optimized curves
    plot_optimizer_curves(optimized_data, kinds, colors_list, suffix="_optimized", optimized=True)

    # summary table with results
    if not summary_table:
        table_str = "No data in summary_table. Check if optimizers ran successfully (e.g., opt not None in run_seed)."
        print(table_str)
    else:
        max_opt_name = max(len(kind) for kind in kinds)
        max_val_loss = max(len(f"{val:.4f} ± {std:.4f}") for _, val, std, _, _, _, _, _ in summary_table)
        max_test_metric = max(len(f"{test:.4f} ± {std:.4f}") for _, _, _, test, std, _, _, _ in summary_table)
        max_time_to_l = max(len(f"{time:.2f} ± {std:.2f}") for _, _, _, _, _, _, time, std in summary_table)
        max_grad_evals = max(len(f"{grad:.0f} ± {std:.0f}") for _, _, _, _, _, grad, _, std in summary_table)
        max_func_evals = max(len(f"{func:.0f} ± {std:.0f}") for _, _, _, _, func, _, _, std in summary_table)
        max_updates = max(len(f"{upd:.0f} ± {std:.0f}") for _, _, _, _, _, _, upd, std in summary_table)
        max_gen_gap = max(len(f"{gap:.4f}") for _, _, _, _, _, gap, _, _ in summary_table)

        table_str = (f"| {'Optimizer':<{max_opt_name}} | {'Best Val Loss':>{max_val_loss}} | "
                    f"{'Test Metric':>{max_test_metric}} | {'Gen Gap':>{max_gen_gap}} | "
                    f"{'Time to ℓ (s)':>{max_time_to_l}} | "
                    f"{'# Grad Evals':>{max_grad_evals}} | {'# Func Evals':>{max_func_evals}} | "
                    f"{'Updates':>{max_updates}} |\n")
        table_str += (f"|{'-' * (max_opt_name + 2)}|{'-' * (max_val_loss + 2)}|"
                    f"{'-' * (max_test_metric + 2)}|{'-' * (max_gen_gap + 2)}|"
                    f"{'-' * (max_time_to_l + 2)}|"
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
                        f"{avg_test:.4f} ± {std_test:.4f} | {gen_gap:.4f} | "
                        f"{avg_time:.2f} ± {std_time:.2f} | "
                        f"{avg_grad_evals:.0f} ± {std_grad_evals:.0f} | {avg_func_evals:.0f} ± {std_func_evals:.0f} | "
                        f"{avg_steps:.0f} ± {std_steps:.0f} |\n")

    print(table_str)
    self.ml_logger.log_metric('part2_table', table_str)
    
    # log comparison summary
    baseline_optimizers = ['sgd', 'sgd_momentum', 'nesterov']
    adam_family = ['adam', 'adam_no_bias', 'rmsprop_like', 'adamw']
    
    # find best baseline optimizer
    baseline_gaps = [(k, float(np.mean(all_data[k]['gen_gaps']))) for k in baseline_optimizers if all_data[k]['gen_gaps']]
    if baseline_gaps:
        best_baseline = min(baseline_gaps, key=lambda x: x[1])
        self.ml_logger.log_metric('best_baseline_optimizer', f'{best_baseline[0]} (gen_gap={best_baseline[1]:.6f})')
    
    # find best adam-family optimizer
    adam_gaps = [(k, float(np.mean(all_data[k]['gen_gaps']))) for k in adam_family if all_data[k]['gen_gaps']]
    if adam_gaps:
        best_adam = min(adam_gaps, key=lambda x: x[1])
        self.ml_logger.log_metric('best_adam_family_optimizer', f'{best_adam[0]} (gen_gap={best_adam[1]:.6f})')
    
    # log overall best
    all_gaps = [(k, float(np.mean(all_data[k]['gen_gaps']))) for k in kinds if all_data[k]['gen_gaps']]
    if all_gaps:
        overall_best = min(all_gaps, key=lambda x: x[1])
        self.ml_logger.log_metric('overall_best_optimizer', f'{overall_best[0]} (gen_gap={overall_best[1]:.6f})')
    
    # log total function timing
    function_elapsed = time.perf_counter() - function_start
    self.ml_logger.log_metric('total_duration', function_elapsed)
    print(f"\n[Adam Ablations] Total execution time: {function_elapsed:.2f}s")
    
    self.ml_logger.generate_log_report(output_file=f"{part2_path}/part2_report.txt", part=2)

    print(f"\nBest hypers per Adam-family variant: {best_hypers}")
    return best_hypers