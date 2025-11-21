import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from core.training import eval_loss, train_to_budget, print_experiment_config
from core.optimizers import optimizer_factory
from utils.plotter import plot_curve
from core.models import set_seed, MLP

def apply_augmentation(x: torch.Tensor, method: str, **kwargs) -> torch.Tensor:
    """simple data augmentation for tabular data
    - gaussian: adds random noise (regression)
    - feature_mask: zeros random features (classification)
    - feature_noise: adds uniform noise within range"""
    if method == 'gaussian':
        # add gaussian noise
        return x + torch.randn_like(x) * kwargs.get('noise_std', 0.1)
    
    elif method == 'feature_mask':
        # randomly zero out features
        mask = torch.bernoulli(torch.full_like(x, 1 - kwargs.get('mask_prob', 0.1)))
        return x * mask
    
    elif method == 'feature_noise':
        # add uniform noise
        low, high = kwargs.get('noise_range', (-0.1, 0.1))
        return x + torch.rand_like(x) * (high - low) + low
    
    return x

def targeted_regularization(
    self,
    max_updates: int = 10000,
    learning_rate: Optional[float] = None,  # use best from ablations
    betas: Tuple[float, float] = (0.9, 0.999),  # best from ablations
    seeds: List[int] = [42, 4242, 424242],
    train_loader=None,
    val_loader=None,
    test_loader=None
) -> Dict:
    """evaluate regularization techniques with standard adam from part 2:
    - l2, early stopping, dropout, smoothing, augmentation
    - measures test impact and generalization gap
    - plots sensitivity curves and combined results"""

    function_start = time.perf_counter()
    
    eval_interval = 25
    log_interval = 100

    hidden_layers = self.best_params.get('hidden_layer_sizes', [128, 64])
    part3_path = f"{self.save_path}/{os.path.splitext(os.path.basename(__file__))[0]}"
    os.makedirs(part3_path, exist_ok=True)

    # Use provided loaders or load data (optimization: avoid redundant loading)
    if train_loader is None:
        train_loader, val_loader, test_loader = self.get_data()
        
    in_dim = train_loader.dataset.tensors[0].shape[1]
    out_dim = len(torch.unique(train_loader.dataset.tensors[1])) if self.method == 'classification' else 1
    learning_rate = learning_rate or self.best_params.get('alpha', 0.01)

    # setup loss functions
    base_loss_fn = nn.CrossEntropyLoss() if self.method == 'classification' else nn.MSELoss()

    def smoothed_loss_fn(out, y, smoothing=0.0):
        """applies label smoothing for classification"""
        if self.method != 'classification' or smoothing == 0.0:
            return base_loss_fn(out, y)
        
        # compute smooth labels
        conf = 1.0 - smoothing
        log_probs = F.log_softmax(out, dim=-1)
        with torch.no_grad():
            smooth_labels = torch.ones_like(log_probs) * (smoothing / (out_dim - 1))
            smooth_labels.scatter_(-1, y.unsqueeze(-1), conf)
        return -(smooth_labels * log_probs).sum(dim=-1).mean()

    # initialize regularization types
    kinds = ['baseline', 'l2', 'early_stopping', 'dropout', 'label_smoothing', 'augmentation']
    
    # Print detailed experiment configuration
    print_experiment_config(
        part_name="TR: Targeted Regularization",
        dataset=self.dataset,
        method=self.method,
        architecture=hidden_layers,
        device=self.device,
        optimizer_name="Adam",
        learning_rate=learning_rate,
        max_updates=max_updates,
        l_threshold=0.0,  # No threshold for regularization experiments
        train_loader=train_loader,
        val_loader=val_loader, #type:ignore
        test_loader=test_loader, #type:ignore
        model=MLP(in_dim=in_dim, hidden=hidden_layers, out_dim=out_dim,
                  activation=self.best_params.get('activation', 'relu')).to(self.device),
        seeds=seeds,
        betas=betas,
        regularization_types=kinds
    )
    
    # store metrics and curves
    all_data = {kind: {
        'curves': [], 'test_metrics': [], 'times': [], 
        'gen_gaps': [], 'updates': []
    } for kind in kinds}
    
    # summary stats: (name, val±std, test±std, gap, time±std)
    summary_table = []

    def run_seed(
        kind: str, 
        seed: int, 
        weight_decay=0.0, 
        patience=5, 
        min_delta=1e-4, 
        dropout_p=0.0, 
        smoothing=0.0
    ) -> Optional[Tuple[List[float], float, float, float, float]]:
        """run one training instance with given regularization settings"""
        
        # setup model with regularization
        set_seed(seed)
        model = MLP(
            in_dim=in_dim,
            hidden=hidden_layers,
            out_dim=out_dim,
            dropout_p=dropout_p if kind == 'dropout' else 0.0,
            activation=self.best_params.get('activation', 'relu')
        ).to(self.device)
        
        # setup optimizer with regularization params
        opt_params = {
            'lr': learning_rate,
            'betas': betas,
            'weight_decay': weight_decay if kind == 'l2' else 0.0
        }
        opt = optimizer_factory(model, 'adam', **opt_params)
        if opt is None:
            return None
            
        loss_fn = lambda out, y: (
            smoothed_loss_fn(out, y, smoothing) 
            if kind == 'label_smoothing' 
            else base_loss_fn(out, y)
        )

        # train for full budget
        curves, _, wall_time, final_train_loss = train_to_budget(
            model, opt, train_loader, val_loader, #type:ignore
            max_updates, float('inf'), loss_fn, 
            self.device, 
            log_interval=log_interval,
            eval_interval=eval_interval,
            optimizer_name='adam'
        )

        # early stopping: simulate by monitoring curves, stop if no improvement
        if kind == 'early_stopping':
            best_loss = float('inf')
            best_idx = 0
            for idx, val_loss in enumerate(curves):
                if val_loss < best_loss - min_delta:
                    best_loss = val_loss
                    best_idx = idx
                elif idx - best_idx >= patience:
                    curves = curves[:idx + 1]  # truncate
                    break

        # evaluate final performance
        test_metric = eval_loss(model, test_loader, base_loss_fn, self.device) #type:ignore
        gen_gap = test_metric - final_train_loss
        return curves, test_metric, wall_time, gen_gap, len(curves)

    # baseline: standard adam no reg
    print(f"\n{'='*70}")
    print(f"[TR] Running technique: BASELINE (no regularization)")
    print(f"{'='*70}")
    
    for kind in ['baseline']:
        results = [run_seed(kind, s) for s in seeds]
        valid_results = [res for res in results if res]
        if not valid_results:
            continue
        # store data
        curves_list = [res[0] for res in valid_results]
        test_metrics = [res[1] for res in valid_results]
        times = [res[2] for res in valid_results]
        gen_gaps = [res[3] for res in valid_results]
        updates_list = [res[4] for res in valid_results]
        all_data[kind]['curves'] = curves_list
        all_data[kind]['test_metrics'] = test_metrics
        all_data[kind]['times'] = times
        all_data[kind]['gen_gaps'] = gen_gaps
        all_data[kind]['updates'] = updates_list

        avg_test = np.mean(test_metrics)
        std_test = np.std(test_metrics)
        avg_gen_gap = np.mean(gen_gaps)
        avg_time = np.mean(times)
        std_time = np.std(times)
        avg_val = np.mean([c[-1] for c in curves_list])
        std_val = np.std([c[-1] for c in curves_list])
        summary_table.append((kind, avg_val, std_val, avg_test, std_test, avg_gen_gap, avg_time, std_time)) # type: ignore

    # sensitivity for each reg
    l2_grid = [1e-4, 1e-3, 1e-2, 1e-1]
    patience_grid = [3, 5, 10, 20]
    dropout_grid = [0.1, 0.2, 0.3, 0.4]
    
    # Classification-specific grids
    if self.method == 'classification':
        smooth_grid = [0.05, 0.1, 0.15, 0.2]
        aug_grid = [0.05, 0.1, 0.15, 0.2]  # mask probabilities
        aug_method = 'feature_mask'
    else:
        smooth_grid = []
        aug_grid = [0.01, 0.05, 0.1, 0.15]  # noise std for regression
        aug_method = 'gaussian'

    # L2 Weight Decay Grid Search
    print(f"\n{'='*70}")
    print(f"[TR] Running technique: L2 WEIGHT DECAY")
    print(f"Grid: {l2_grid}")
    print(f"{'='*70}")
    
    sens_l2 = np.zeros((len(l2_grid),))
    gen_gaps_l2 = np.zeros((len(l2_grid),))
    for i, wd in enumerate(l2_grid):
        results = [run_seed('l2', s, weight_decay=wd) for s in seeds]
        valid_results = [res for res in results if res]
        final_vals = [c[-1] for c, _, _, _, _ in valid_results]
        gen_gaps = [g for _, _, _, g, _ in valid_results]
        sens_l2[i] = np.mean(final_vals)
        gen_gaps_l2[i] = np.mean(gen_gaps)
    best_l2 = l2_grid[np.argmin(sens_l2)]

    # early stopping: grid over patience
    print(f"\n{'='*70}")
    print(f"[TR] Running technique: EARLY STOPPING")
    print(f"Grid: {patience_grid}")
    print(f"{'='*70}")
    
    patience_grid = [3, 5, 10, 20, 50]
    sens_es = np.zeros((len(patience_grid),))
    gen_gaps_es = np.zeros((len(patience_grid),))
    for i, pat in enumerate(patience_grid):
        results = [run_seed('early_stopping', s, patience=pat) for s in seeds]
        valid_results = [res for res in results if res]
        final_vals = [c[-1] for c, _, _, _, _ in valid_results]
        gen_gaps = [g for _, _, _, g, _ in valid_results]
        sens_es[i] = np.mean(final_vals)
        gen_gaps_es[i] = np.mean(gen_gaps)
    best_patience = patience_grid[np.argmin(sens_es)]

    # dropout: grid over p (excluding 0 to force dropout)
    print(f"\n{'='*70}")
    print(f"[TR] Running technique: DROPOUT")
    print(f"Grid: {dropout_grid}")
    print(f"Dropout Placement: After each hidden layer activation (ReLU/Tanh), before next linear layer")
    print(f"Architecture: Input → [Linear → Activation → Dropout]* → Linear → Output")
    print(f"{'='*70}")
    
    dropout_grid = [0.1, 0.2, 0.3, 0.4, 0.5]
    sens_dropout = np.zeros((len(dropout_grid),))
    gen_gaps_dropout = np.zeros((len(dropout_grid),))
    for i, p in enumerate(dropout_grid):
        results = [run_seed('dropout', s, dropout_p=p) for s in seeds]
        valid_results = [res for res in results if res]
        final_vals = [c[-1] for c, _, _, _, _ in valid_results]
        gen_gaps = [g for _, _, _, g, _ in valid_results]
        sens_dropout[i] = np.mean(final_vals)
        gen_gaps_dropout[i] = np.mean(gen_gaps)
    best_dropout = dropout_grid[np.argmin(sens_dropout)]

    # label smoothing
    smooth_grid = []
    sens_smooth = np.array([])
    gen_gaps_smooth = np.array([])

    # label smoothing: grid over alpha (classification only, excluding 0 to force smoothing)
    if self.method == 'classification':
        smooth_grid = [0.01, 0.05, 0.1, 0.15, 0.2]
        
        print(f"\n{'='*70}")
        print(f"[TR] Running technique: LABEL SMOOTHING")
        print(f"Grid: {smooth_grid}")
        print(f"{'='*70}")
        
        sens_smooth = np.zeros((len(smooth_grid),))
        gen_gaps_smooth = np.zeros((len(smooth_grid),))
        for i, sm in enumerate(smooth_grid):
            results = [run_seed('label_smoothing', s, smoothing=sm) for s in seeds]
            valid_results = [res for res in results if res]
            final_vals = [c[-1] for c, _, _, _, _ in valid_results]
            gen_gaps = [g for _, _, _, g, _ in valid_results]
            sens_smooth[i] = np.mean(final_vals)
            gen_gaps_smooth[i] = np.mean(gen_gaps)
        best_smooth = smooth_grid[np.argmin(sens_smooth)]
    else:
        best_smooth = 0.0  # skip for regression

    # Modality-appropriate augmentation (excluding 0 to force augmentation)
    print(f"\n{'='*70}")
    print(f"[TR] Running technique: DATA AUGMENTATION")
    print(f"Method: {aug_method}")
    print(f"Grid: {aug_grid}")
    print(f"{'='*70}")
    
    if self.method == 'classification':
        # For classification: feature masking
        aug_grid = [0.05, 0.1, 0.15, 0.2, 0.25]  # mask probabilities
        aug_method = 'feature_mask'
    else:
        # For regression: gaussian noise
        aug_grid = [0.01, 0.05, 0.1, 0.15, 0.2]  # noise std
        aug_method = 'gaussian'
    
    sens_aug = np.zeros((len(aug_grid),))
    gen_gaps_aug = np.zeros((len(aug_grid),))
    
    class AugmentedDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, method, **kwargs):
            self.dataset = dataset
            self.method = method
            self.kwargs = kwargs
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            x, y = self.dataset[idx]
            x_aug = apply_augmentation(x, self.method, **self.kwargs)
            return x_aug, y

    # Grid search over augmentation parameters
    for i, param in enumerate(aug_grid):
        # Configure augmentation kwargs based on method
        if aug_method == 'gaussian':
            aug_kwargs = {'noise_std': param}
        else:  # feature_mask
            aug_kwargs = {'mask_prob': param}
            
        # Wrap datasets with augmentation
        aug_train = AugmentedDataset(train_loader.dataset, aug_method, **aug_kwargs)
        aug_train_loader = torch.utils.data.DataLoader(
            aug_train, 
            batch_size=train_loader.batch_size,
            shuffle=True
        )
        
        results = [run_seed('augmentation', s) for s in seeds]
        valid_results = [res for res in results if res]
        final_vals = [c[-1] for c, _, _, _, _ in valid_results]
        gen_gaps = [g for _, _, _, g, _ in valid_results]
        sens_aug[i] = np.mean(final_vals)
        gen_gaps_aug[i] = np.mean(gen_gaps)
    
    # Select best augmentation parameter
    best_aug_param = aug_grid[np.argmin(sens_aug)]
    
    # Plot augmentation sensitivity
    plot_curve(
        x=aug_grid,
        y_list=[sens_aug, gen_gaps_aug],
        labels=['Val Loss', 'Gen Gap'],
        xlabel=f"{'Noise STD' if aug_method == 'gaussian' else 'Mask Probability'}",
        ylabel="Metric Value",
        title=f"Augmentation Sensitivity on {self.dataset.title()}",
        save_path=f"{part3_path}/augmentation_sensitivity.png"
    )

    # optimized runs with best params
    optimized_data: Dict[str, Dict[str, Any]] = {k: {} for k in kinds[1:]}  # exclude baseline
    for kind in kinds[1:]:
        if kind == 'l2':
            kw = {'weight_decay': best_l2}
        elif kind == 'early_stopping':
            kw = {'patience': best_patience, 'min_delta': 1e-4}
        elif kind == 'dropout':
            kw = {'dropout_p': best_dropout}
        elif kind == 'label_smoothing':
            kw = {'smoothing': best_smooth}
        else:
            kw = {}
        results = [run_seed(kind, s, **kw) for s in seeds] # type: ignore
        valid_results = [res for res in results if res]
        if not valid_results:
            continue
        curves_list = [res[0] for res in valid_results]
        test_metrics = [res[1] for res in valid_results]
        times = [res[2] for res in valid_results]
        gen_gaps = [res[3] for res in valid_results]
        updates_list = [res[4] for res in valid_results]
        all_data[kind]['curves'] = curves_list
        all_data[kind]['test_metrics'] = test_metrics
        all_data[kind]['times'] = times
        all_data[kind]['gen_gaps'] = gen_gaps
        all_data[kind]['updates'] = updates_list

        avg_test = np.mean(test_metrics)
        std_test = np.std(test_metrics)
        avg_gen_gap = np.mean(gen_gaps)
        avg_time = np.mean(times)
        std_time = np.std(times)
        avg_val = np.mean([c[-1] for c in curves_list])
        std_val = np.std([c[-1] for c in curves_list])
        summary_table.append((kind, avg_val, std_val, avg_test, std_test, avg_gen_gap, avg_time, std_time)) # type: ignore

    # simple color selection without extra imports
    base_colors = ['#000000', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    colors = [base_colors[i % len(base_colors)] for i in range(len(kinds))]
    # L2
    plot_curve(
        x=l2_grid,
        y_list=[sens_l2, gen_gaps_l2],
        labels=['Val Loss', 'Gen Gap'],
        xlabel="L2 Weight Decay",
        ylabel="Metric Value",
        title=f"L2 Sensitivity on {self.dataset.title()} (hidden: {hidden_layers})",
        save_path=f"{part3_path}/l2_sensitivity.png",
        colors=['#1f77b4', '#ff7f0e']  # blue for Val Loss, orange for Gen Gap
    )

    # early stopping (patience)
    plot_curve(
        x=patience_grid,
        y_list=[sens_es, gen_gaps_es],
        labels=['Val Loss', 'Gen Gap'],
        xlabel="Early Stopping Patience",
        ylabel="Metric Value",
        title=f"Early Stopping Sensitivity on {self.dataset.title()} (hidden: {hidden_layers})",
        save_path=f"{part3_path}/early_stopping_sensitivity.png",
        colors=['#1f77b4', '#ff7f0e']  # blue for Val Loss, orange for Gen Gap
    )

    # dropout
    plot_curve(
        x=dropout_grid,
        y_list=[sens_dropout, gen_gaps_dropout],
        labels=['Val Loss', 'Gen Gap'],
        xlabel="Dropout Probability",
        ylabel="Metric Value",
        title=f"Dropout Sensitivity on {self.dataset.title()} (hidden: {hidden_layers})",
        save_path=f"{part3_path}/dropout_sensitivity.png",
        colors=['#1f77b4', '#ff7f0e']  # blue for Val Loss, orange for Gen Gap
    )

    # label smoothing (classification only) - guard against unbound variables
    if self.method == 'classification' and all(v in locals() for v in ('smooth_grid', 'sens_smooth', 'gen_gaps_smooth')):
        plot_curve(
            x=smooth_grid,
            y_list=[sens_smooth, gen_gaps_smooth],
            labels=['Val Loss', 'Gen Gap'],
            xlabel="Label Smoothing (alpha)",
            ylabel="Metric Value",
            title=f"Label Smoothing Sensitivity on {self.dataset.title()} (hidden: {hidden_layers})",
            save_path=f"{part3_path}/label_smoothing_sensitivity.png",
            colors=['#1f77b4', '#ff7f0e']  # blue for Val Loss, orange for Gen Gap
        )

    # augmentation sensitivity
    plot_curve(
        x=aug_grid,
        y_list=[sens_aug, gen_gaps_aug],
        labels=['Val Loss', 'Gen Gap'],
        xlabel=f"{'Noise STD' if aug_method == 'gaussian' else 'Mask Probability'}",
        ylabel="Metric Value",
        title=f"Augmentation Sensitivity on {self.dataset.title()} (hidden: {hidden_layers})",
        save_path=f"{part3_path}/augmentation_sensitivity.png",
        colors=['#1f77b4', '#ff7f0e']  # blue for Val Loss, orange for Gen Gap
    )

    # Create a combined subplot showing all sensitivities with proper units
    from utils.plotter import plot_sensitivity_subplots
    
    # Prepare sensitivity data for plotting
    sensitivity_plots = [
        {
            'x': l2_grid,
            'y': sens_l2,
            'xlabel': 'L2 Weight Decay',
            'title': 'L2 Regularization',
            'color': '#1f77b4'
        },
        {
            'x': patience_grid,
            'y': sens_es,
            'xlabel': 'Patience (epochs)',
            'title': 'Early Stopping',
            'color': '#ff7f0e'
        },
        {
            'x': dropout_grid,
            'y': sens_dropout,
            'xlabel': 'Dropout Probability',
            'title': 'Dropout',
            'color': '#2ca02c'
        },
        {
            'x': aug_grid,
            'y': sens_aug,
            'xlabel': f"{'Noise STD' if aug_method == 'gaussian' else 'Mask Probability'}",
            'title': 'Data Augmentation',
            'color': '#d62728'
        }
    ]
    
    # Add label smoothing for classification only
    if self.method == 'classification' and len(smooth_grid) > 0:
        sensitivity_plots.append({
            'x': smooth_grid,
            'y': sens_smooth,
            'xlabel': 'Smoothing Alpha',
            'title': 'Label Smoothing',
            'color': '#9467bd'
        })
    
    # Create the combined sensitivity subplot
    plot_sensitivity_subplots(
        sensitivity_data=sensitivity_plots,
        overall_title=f'Regularization Sensitivity Analysis - {self.dataset.title()}',
        save_path=f"{part3_path}/combined_sensitivity_subplots.png"
    )

    # cumulative curves: baseline vs optimized regs
    # calculate mean curves with proper padding for alignment
    max_len = max(max(len(c) for c in all_data[kind]['curves']) if all_data[kind]['curves'] else 0 for kind in kinds)
    all_mean_curves = []
    
    for kind in kinds:
        curves = all_data[kind]['curves']
        if curves:
            # pad each curve to max length
            padded_curves = [np.pad(c, (0, max_len - len(c)), 'edge') for c in curves]
            mean_curve = np.median(padded_curves, axis=0)
            all_mean_curves.append(mean_curve)
        else:
            # no data for this kind, use zeros
            all_mean_curves.append(np.zeros(max_len))
    
    # create common x-axis scaled by eval_interval
    x_vals = np.arange(max_len) * eval_interval
    
    # Diverse linestyles to distinguish overlapping lines
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2, 1, 2))]
    
    plot_curve(
        x=x_vals,
        y_list=all_mean_curves,
        labels=kinds,
        xlabel="Updates",
        ylabel="Validation Loss",
        title=f"Regularization Curves on {self.dataset.title()} (hidden: {hidden_layers})",
        save_path=f"{part3_path}/cumulative_reg_curves.png",
        colors=['#000000', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(kinds)],
        linestyles=linestyles[:len(kinds)]
    )

    # summary table focusing on validation, test, and generalization metrics
    if not summary_table:
        table_str = "No data in summary_table. Check if runs completed successfully."
    else:
        # calculate maximum widths for alignment
        max_reg_name = max(len(str(kind)) for kind, *_ in summary_table)
        max_val_loss = max(len(f"{val:.6f} ± {val_std:.6f}") for _, val, val_std, *_ in summary_table)
        max_test_metric = max(len(f"{test:.6f} ± {test_std:.6f}") for _, _, _, test, test_std, *_ in summary_table)
        max_gen_gap = max(len(f"{gap:.6f}") for _, _, _, _, _, gap, *_ in summary_table)
        max_time = max(len(f"{measure_time:.4f} ± {time_std:.4f}") for _, _, _, _, _, _, measure_time, time_std in summary_table)

        # build header
        table_str = (f"| {'Regularizer':<{max_reg_name}} | {'Val Loss':>{max_val_loss}} | "
                    f"{'Test Metric':>{max_test_metric}} | {'Gen Gap':>{max_gen_gap}} | "
                    f"{'Time (s)':>{max_time}} |\n")
        table_str += (f"|{'-' * (max_reg_name + 2)}|{'-' * (max_val_loss + 2)}|"
                    f"{'-' * (max_test_metric + 2)}|{'-' * (max_gen_gap + 2)}|"
                    f"{'-' * (max_time + 2)}|\n")

        # add rows
        for kind, val, val_std, test, test_std, gap, measure_time, time_std in summary_table:
            table_str += (f"| {kind:<{max_reg_name}} | {val:.6f} ± {val_std:.6f} | "
                        f"{test:.6f} ± {test_std:.6f} | {gap:>{max_gen_gap}.6f} | "
                        f"{measure_time:.4f} ± {time_std:.4f} |\n")

        # add configuration summary footer
        table_str += "\nBest configurations:"
        table_str += f"\nL2 weight decay: {best_l2:.2e}"
        table_str += f"\nEarly stopping patience: {best_patience}"
        table_str += f"\nDropout probability: {best_dropout:.6f}"
        if self.method == 'classification':
            table_str += f"\nLabel smoothing alpha: {best_smooth:.6f}"
            table_str += f"\nAugmentation ({aug_method}): {best_aug_param:.6f}"

    print(f"\n{'='*70}")
    print("[TR] Regularization Results Summary")
    print(f"{'='*70}")
    print(table_str)
    self.ml_logger.log_metric('regularization_table', table_str)    # integrate best recipe: combine top regularizers and evaluate
    combined_kw = {
        'weight_decay': best_l2, 
        'dropout_p': best_dropout, 
        'smoothing': best_smooth, 
        'patience': best_patience
    }
    combined_results = [run_seed('combined', s, **combined_kw) for s in seeds]
    valid_combined = [res for res in combined_results if res]
    if valid_combined:
        # store data similar to individual runs
        curves_list = [res[0] for res in valid_combined]
        test_metrics = [res[1] for res in valid_combined]
        times = [res[2] for res in valid_combined]
        gen_gaps = [res[3] for res in valid_combined]
        updates_list = [res[4] for res in valid_combined]
        
        # calculate statistics
        avg_test = np.mean(test_metrics)
        std_test = np.std(test_metrics)
        avg_gen_gap = np.mean(gen_gaps)
        avg_time = np.mean(times)
        std_time = np.std(times)
        avg_val = np.mean([c[-1] for c in curves_list])
        std_val = np.std([c[-1] for c in curves_list])
        
        # add to summary table
        summary_table.append(('combined', float(avg_val), float(std_val), float(avg_test), float(std_test), float(avg_gen_gap), float(avg_time), float(std_time)))
        
        # plot combined recipe against individual best regularizers
        # get best individual regularizers by gen gap
        reg_by_gap = [(k, float(np.mean(d['gen_gaps']))) for k, d in all_data.items() if k != 'baseline']
        reg_by_gap = sorted(reg_by_gap, key=lambda x: x[1])[:2]  # top 2 individual regularizers
        
        comparison_kinds = ['baseline'] + [k for k, _ in reg_by_gap] + ['combined']
        plot_data = []
        
        for kind in comparison_kinds:
            if kind == 'combined':
                curve = np.median(curves_list, axis=0)
                plot_data.append(curve)
        
            else:
                curve = np.median(all_data[kind]['curves'], axis=0)
                plot_data.append(curve)
        
        # use max length for x-axis to ensure alignment
        max_len = max(len(curve) for curve in plot_data)
        x = np.arange(max_len) * eval_interval
        
        # debug: print curve statistics
        print(f"\nCombined Recipe Comparison Debug:")
        for kind, curve in zip(comparison_kinds, plot_data):
            print(f"  {kind}: len={len(curve)}, final_val={curve[-1]:.6f}, mean={np.mean(curve):.6f}")
        
        # pad shorter curves with their last value
        plot_data = [np.pad(curve, (0, max_len - len(curve)), 'edge') for curve in plot_data]
        
        plot_curve(
            x=x,
            y_list=plot_data,
            labels=comparison_kinds,
            xlabel="Updates",
            ylabel="Validation Loss",
            title=f"Combined Recipe vs Best Individual on {self.dataset.title()}",
            save_path=f"{part3_path}/combined_recipe_comparison.png",
            colors=['#000000', '#1f77b4', '#ff7f0e', '#d62728'][:len(comparison_kinds)],
            linestyles=['-', '--', '-.', ':'][:len(comparison_kinds)]  # use different linestyles to distinguish overlapping curves
        )
        
        # log combined recipe performance
        recipe_summary = "\nCombined Recipe Performance:"
        recipe_summary += f"\nValidation Loss: {avg_val:.6f} ± {std_val:.6f}"
        recipe_summary += f"\nTest Metric: {avg_test:.6f} ± {std_test:.6f}"
        recipe_summary += f"\nGeneralization Gap: {avg_gen_gap:.6f}"
        recipe_summary += f"\nTraining Time: {avg_time:.4f} ± {std_time:.4f}s"
        
        # compare against best individual
        best_single_gap = min(float(np.mean(d['gen_gaps'])) for d in all_data.values())
        hypothesis = f"\nHypothesis Test - Combined Recipe Performance:"
        hypothesis += f"\n- Best Single Regularizer Gap: {best_single_gap:.6f}"
        hypothesis += f"\n- Combined Recipe Gap: {avg_gen_gap:.6f}"
        
        print(recipe_summary)
        print(hypothesis)
        
        self.ml_logger.log_metric('combined_recipe_summary', recipe_summary)
        self.ml_logger.log_metric('combined_recipe_hypothesis', hypothesis)
    else:
        print("\nWarning: Combined recipe runs failed to complete successfully")

    # log total function timing
    function_elapsed = time.perf_counter() - function_start
    self.ml_logger.log_metric('total_duration', function_elapsed)
    print(f"\n[Targeted Regularization] Total execution time: {function_elapsed:.2f}s")

    # generate report
    self.ml_logger.generate_log_report(output_file=f"{part3_path}/execution_report.txt", part=3)
    
    # Return comprehensive results for comparison report
    summary_results = {}
    for kind in all_data.keys():
        if not all_data[kind]['test_metrics']:
            continue
        summary_results[kind] = {
            'mean_val': float(np.mean([c[-1] for c in all_data[kind]['curves']])),
            'std_val': float(np.std([c[-1] for c in all_data[kind]['curves']])),
            'mean_test': float(np.mean(all_data[kind]['test_metrics'])),
            'std_test': float(np.std(all_data[kind]['test_metrics'])),
            'mean_gen_gap': float(np.mean(all_data[kind]['gen_gaps'])),
            'std_gen_gap': float(np.std(all_data[kind]['gen_gaps'])),
        }
    
    # Add total execution time to summary
    summary_results['_execution_time'] = function_elapsed
    
    return summary_results