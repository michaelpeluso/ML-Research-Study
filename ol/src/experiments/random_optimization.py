import os
import numpy as np
import torch
import torch.nn as nn

from utils.plotter import plot_curve
from core.models import MLP
from core.random_optimizers import rhc, sa, ga, validation_objective, get_trainable_params


def random_optimization(self, max_param: int = 50000, max_evals: int = 10000, plateau_threshold: int = 250):
        '''
        Run Part 1 RO: freeze model, run rhc/sa/ga, log history for analysis, generate report,
        and plot curves including best-so-far objective vs. evals.
        '''
        hidden_layers = self.best_params.get('hidden_layer_sizes', [128, 64])
        print(f"\n\nExecuting Randomized optimization...\nMethod: {self.method}\nDataset: {self.dataset}\nNetwork: {hidden_layers}\n".upper())
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
                title=f"{algo.__name__.upper()} Condensed Curves on {self.dataset.title()} dataset (hidden: {hidden_layers})",
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
            title=f"Raw and Best Evaluations on {self.dataset.title()} dataset (hidden: {hidden_layers})",
            save_path=f"{part1_path}/combined_ro_curves.png",
            colors=colors_combined, linestyles=linestyles_combined
        )

        # test losses
        for algo_name, test_loss in test_losses.items():
            print(f"{algo_name} Test Loss: {test_loss}")
            self.ml_logger.log_metric(f"{algo_name}_test_loss", test_loss)

        # generate report
        self.ml_logger.generate_log_report(output_file=f"{part1_path}/part1_report.txt", part=1)

