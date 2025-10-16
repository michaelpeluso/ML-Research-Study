import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from core.training import eval_loss, train_to_budget
from core.optimizers import optimizer_factory
from utils.plotter import plot_curve, plot_heatmaps, stitch_images
from core.models import set_seed, MLP


def ablations(
    self,
    max_updates: int = 10000,
    learning_threshold: float = 0.5,
    learning_rate: float = 0.01,
    seeds: List[int] = [42, 4242, 424242]
):
    '''run optimizer ablations with sensitivity analysis.'''  # existing comment preserved
    hidden_layers = self.best_params.get('hidden_layer_sizes', [128, 64])
    print(f"\n\nExecuting Optimizer Ablations...\nMethod: {self.method}\nDataset: {self.dataset}\nNetwork: {hidden_layers}\n".upper())
    part2_path = f"{self.save_path}/{os.path.splitext(os.path.basename(__file__))[0]}"
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

    # run_seed function
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
        test_metric = eval_loss(local_model, test_loader, loss_fn, self.device)  # test loss as metric
        return curves, test_metric, steps_to_l, wall_time, final_train_loss

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
            title=f"'{kind}' Ablation Mean Curve on {self.dataset.title()} dataset (hidden: {hidden_layers})",
            save_path=f"{part2_path}/curve_{kind}.png",
            band_label="± Std Dev"
        )

        # sensitivity analysis for adam variants
        if kind in ['adam', 'adam_no_bias', 'adamw']:
            lr_grid = [1e-4, 1e-3, 1e-2, 1e-1]  # log-spaced lr
            beta1_grid = [0.9, 0.99, 0.999, 0.9999]  # common beta1
            beta2_grid = [0.9, 0.99, 0.999, 0.9999]  # common beta2
            default_b2 = 0.999  # fix for lr vs beta1
            default_b1 = 0.9    # fix for lr vs beta2

            # lr vs beta1 sensitivity
            sens_lr_b1 = np.zeros((len(beta1_grid), len(lr_grid)))
            gen_gaps_lr_b1 = np.zeros((len(beta1_grid), len(lr_grid)))
            for i, b1 in enumerate(beta1_grid):
                for j, lr in enumerate(lr_grid):
                    results = [run_seed(kind, seed, lr=lr, betas=(b1, default_b2)) for seed in seeds]
                    valid_results = [res for res in results if res]
                    if not valid_results:
                        continue
                    final_vals = [res[0][-1] for res in valid_results]  # final val loss (positive)
                    gen_gaps = [res[0][-1] - res[4] for res in valid_results]  # val[-1] - final_train_loss
                    sens_lr_b1[i, j] = np.mean(final_vals)  # mean val loss
                    gen_gaps_lr_b1[i, j] = np.mean(gen_gaps)  # mean gen gap

            # plot val loss (ensure positive; no negation)
            plot_heatmaps(
                sens_lr_b1, 
                beta1_grid, 
                lr_grid, 
                title=f"'{kind}' Val Loss (LR vs Beta1) on {self.dataset.title()} dataset (hidden: {hidden_layers})", 
                save_path=f"{part2_path}/{kind}_lr_b1_heatmap.png"
            )
            # add gen gap heatmap
            plot_heatmaps(
                gen_gaps_lr_b1, 
                beta1_grid, 
                lr_grid, 
                title=f"'{kind}' Gen Gap (LR vs Beta1) on {self.dataset.title()} dataset (hidden: {hidden_layers})", 
                save_path=f"{part2_path}/{kind}_gen_gap_lr_b1_heatmap.png"
            )
            self.ml_logger.log_metric(f"{kind}_gen_gap_lr_b1", np.mean(gen_gaps_lr_b1))
        
    # cumulative plot
    kinds = list(all_data.keys())
    # changed: use per-kind updates to handle potential length differences
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
        title=f"Cumulative Optimizer Loss Curves on {self.dataset.title()} dataset (hidden: {hidden_layers})",
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

