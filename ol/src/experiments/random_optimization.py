import os
import numpy as np
import torch
import torch.nn as nn
import time

from utils.plotter import plot_curve
from core.models import MLP, set_seed
from core.random_optimizers import rhc, sa, ga, validation_objective, get_trainable_params
from core.training import print_experiment_config


def random_optimization(self, max_param: int = 50000, max_evals: int = 10000, plateau_threshold: int = 250, seeds: list = [42]):
    '''
    Run Part 1 RO: freeze model, run rhc/sa/ga, log history for analysis, generate report,
    and plot curves including best-so-far objective vs. evals.
    '''
    function_start = time.perf_counter()
    
    hidden_layers = self.best_params.get('hidden_layer_sizes', [128, 64])
    part1_path = f"{self.save_path}/{os.path.splitext(os.path.basename(__file__))[0]}"
    os.makedirs(part1_path, exist_ok=True)

    train_loader, val_loader, test_loader = self.get_data()

    # Model initialization
    in_dim = train_loader.dataset.tensors[0].shape[1]  # get feature count from first batch
    out_dim = len(torch.unique(train_loader.dataset.tensors[1])) if self.method == 'classification' else 1
    model = MLP(
        in_dim=in_dim,
        hidden=hidden_layers,
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

    # Print detailed experiment configuration
    print_experiment_config(
        part_name="RO: Random Optimization",
        dataset=self.dataset,
        method=self.method,
        architecture=hidden_layers,
        device=self.device,
        optimizer_name="RHC, SA, GA",
        learning_rate=0.0,  # N/A for random optimization
        max_updates=max_evals,
        l_threshold=0.0,  # N/A for random optimization
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model=model,
        max_param=max_param,
        plateau_threshold=plateau_threshold,
        selected_k=selected_k,
        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        seeds=seeds
    )

    # Loss function setup
    loss_fn = nn.CrossEntropyLoss() if self.method == 'classification' else nn.MSELoss()

    # Run RO algorithms with multiple seeds for stability
    algo_results = {}  # Store results per algorithm
    colors_list = ['blue', 'green', 'red']
    
    for i, algo in enumerate([rhc, sa, ga]):
        print(f"\n{'='*70}")
        print(f"[RO] Running algorithm: {algo.__name__.upper()} across {len(seeds)} seeds")
        print(f"{'='*70}")
        
        algo_results[algo.__name__] = {}
        
        # Run algorithm across all seeds
        for seed_idx, seed in enumerate(seeds):
            print(f"\n[RO] {algo.__name__.upper()} - Seed {seed_idx+1}/{len(seeds)}: {seed}")
            set_seed(seed)
            
            # Reinitialize model with same architecture but new seed
            seed_model = MLP(
                in_dim=in_dim,
                hidden=hidden_layers,
                out_dim=out_dim,
                activation=self.best_params.get('activation', 'relu')
            ).to(self.device)
            seed_model.freeze_all_but_last_k(k=selected_k, limit=max_param)
            
            optimized_model, history = algo(
                model=seed_model,
                val_loader=val_loader,
                loss_fn=loss_fn,
                device=self.device,
                max_evals=max_evals,
                plateau_threshold=plateau_threshold,
                logger=self.ml_logger
            )

            # Process history
            evals = [e for e, _ in history]
            losses = [l for _, l in history]
            if len(evals) < 2:
                print(f"Skipping seed {seed}: insufficient data")
                continue
            
            best_so_far = np.minimum.accumulate(np.array(losses)).tolist()
            
            # Evaluate on test set
            test_loss = validation_objective(get_trainable_params(optimized_model), optimized_model, test_loader, loss_fn, self.device)
            
            # Store results for this seed
            algo_results[algo.__name__][seed] = {
                'evals': evals,
                'losses': losses,
                'best_so_far': best_so_far,
                'test_loss': test_loss,
                'final_val_loss': best_so_far[-1]
            }
            
            # Log per-seed metrics
            self.ml_logger.log_metric(f"{algo.__name__}_seed_{seed}_test_loss", test_loss)
            self.ml_logger.log_metric(f"{algo.__name__}_seed_{seed}_final_val_loss", best_so_far[-1])
        
        # Aggregate results across seeds
        if len(algo_results[algo.__name__]) == 0:
            print(f"[RO] No valid results for {algo.__name__}, skipping")
            continue
            
        # Collect final losses and test losses
        final_val_losses = [algo_results[algo.__name__][s]['final_val_loss'] for s in algo_results[algo.__name__]]
        test_losses_list = [algo_results[algo.__name__][s]['test_loss'] for s in algo_results[algo.__name__]]
        
        # Compute statistics
        mean_final_val = np.mean(final_val_losses)
        std_final_val = np.std(final_val_losses)
        mean_test = np.mean(test_losses_list)
        std_test = np.std(test_losses_list)
        
        print(f"\n[RO] {algo.__name__.upper()} Summary across {len(seeds)} seeds:")
        print(f"  Final Val Loss: {mean_final_val:.6f} ± {std_final_val:.6f}")
        print(f"  Test Loss:      {mean_test:.6f} ± {std_test:.6f}")
        
        # Log aggregate statistics
        self.ml_logger.log_metric(f"{algo.__name__}_mean_final_val_loss", mean_final_val)
        self.ml_logger.log_metric(f"{algo.__name__}_std_final_val_loss", std_final_val)
        self.ml_logger.log_metric(f"{algo.__name__}_mean_test_loss", mean_test)
        self.ml_logger.log_metric(f"{algo.__name__}_std_test_loss", std_test)
        
        # Create stability plot with variance bands (median ± std)
        # Interpolate all curves to common eval points for averaging
        max_evals_seen = max(len(algo_results[algo.__name__][s]['evals']) for s in algo_results[algo.__name__])
        common_evals = np.linspace(1, max_evals, min(1000, max_evals_seen))
        
        interpolated_curves = []
        for seed in algo_results[algo.__name__]:
            result = algo_results[algo.__name__][seed]
            interp_losses = np.interp(common_evals, result['evals'], result['best_so_far'])
            interpolated_curves.append(interp_losses)
        
        interpolated_curves = np.array(interpolated_curves)
        median_curve = np.median(interpolated_curves, axis=0)
        std_curve = np.std(interpolated_curves, axis=0)
        
        # Plot with stability bands
        plot_curve(
            x=common_evals,
            y_list=[median_curve],
            labels=[f"{algo.__name__.upper()} Median (n={len(seeds)})"],
            xlabel="Function Evaluations",
            ylabel="Validation Loss (Best-so-Far)",
            title=f"{algo.__name__.upper()} Stability on {self.dataset.title()} dataset (hidden: {hidden_layers})",
            save_path=f"{part1_path}/{algo.__name__}_stability.png",
            colors=[colors_list[i]],
            std=std_curve,
            band_label='± Std Dev'
        )

    # Combined comparison plot: median curves for all algorithms
    print(f"\n{'='*70}")
    print(f"[RO] Generating combined comparison plot")
    print(f"{'='*70}")
    
    x_combined = []
    y_combined = []
    labels_combined = []
    colors_combined = []
    colors_list = ['blue', 'green', 'red']
    
    for i, algo_name in enumerate(algo_results.keys()):
        if len(algo_results[algo_name]) == 0:
            continue
        
        # Interpolate to common eval points
        max_evals_seen = max(len(algo_results[algo_name][s]['evals']) for s in algo_results[algo_name])
        common_evals = np.linspace(1, max_evals, min(1000, max_evals_seen))
        
        interpolated_curves = []
        for seed in algo_results[algo_name]:
            result = algo_results[algo_name][seed]
            interp_losses = np.interp(common_evals, result['evals'], result['best_so_far'])
            interpolated_curves.append(interp_losses)
        
        median_curve = np.median(interpolated_curves, axis=0)
        
        x_combined.append(common_evals)
        y_combined.append(median_curve)
        labels_combined.append(f"{algo_name.upper()} Median")
        colors_combined.append(colors_list[i])
    
    plot_curve(
        x_combined, y_combined, labels=labels_combined, colors=colors_combined,
        xlabel="Function Evaluations", ylabel="Validation Loss (Best-so-Far)",
        title=f"RO Algorithms Comparison on {self.dataset.title()} dataset (hidden: {hidden_layers})",
        save_path=f"{part1_path}/combined_comparison.png"
    )

    # Log summary table with statistics
    print(f"\n{'='*70}")
    print(f"[RO] Summary Statistics")
    print(f"{'='*70}")
    
    table_str = "| Algorithm | Final Val Loss (Mean ± Std) | Test Loss (Mean ± Std) | Seeds |\n"
    table_str += "|-----------|------------------------------|------------------------|-------|\n"
    for algo_name in algo_results.keys():
        if len(algo_results[algo_name]) == 0:
            continue
        final_val_losses = [algo_results[algo_name][s]['final_val_loss'] for s in algo_results[algo_name]]
        test_losses_list = [algo_results[algo_name][s]['test_loss'] for s in algo_results[algo_name]]
        mean_val = np.mean(final_val_losses)
        std_val = np.std(final_val_losses)
        mean_test = np.mean(test_losses_list)
        std_test = np.std(test_losses_list)
        table_str += f"| {algo_name.upper()} | {mean_val:.6f} ± {std_val:.6f} | {mean_test:.6f} ± {std_test:.6f} | {len(algo_results[algo_name])} |\n"
    
    print(f"\nRO Summary Statistics:\n{table_str}")
    self.ml_logger.log_metric('ro_summary_table', table_str)

    # Log total function timing
    function_elapsed = time.perf_counter() - function_start
    self.ml_logger.log_metric('total_duration', function_elapsed)
    print(f"\n[Random Optimization] Total execution time: {function_elapsed:.2f}s")

    # Generate report
    self.ml_logger.generate_log_report(output_file=f"{part1_path}/part1_report.txt", part=1)