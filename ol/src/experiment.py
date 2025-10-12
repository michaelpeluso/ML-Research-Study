import os
import warnings

import numpy as np
warnings.filterwarnings("ignore", message="resource_tracker")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.neural_network")  # Suppress NN batch_size/convergence warnings

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

    def run_part1_ro(self):
        '''
        Run Part 1 RO: freeze model, run rhc/sa/ga, log history for analysis, generate report, and plot curves including best-so-far objective vs. evals.
        '''
        print(f"\n\nExecuting {self.method} neural network on {self.dataset}...".upper())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_loader, val_loader, test_loader = self.get_data()

        # initialize model with dynamic in_dim
        in_dim = train_loader.dataset.tensors[0].shape[1]  # type: ignore # get feature count from first batch
        out_dim = len(torch.unique(train_loader.dataset.tensors[1])) if self.method == 'classification' else 1 # type: ignore
        model = MLP(in_dim=in_dim, hidden=self.best_params.get('hidden_layer_sizes', [128, 64]), out_dim=out_dim).to(device)

        # dynamically select max k (1-3) s.t. trainable <=50k
        max_k = 3
        selected_k = 1  # default min
        for try_k in range(max_k, 0, -1):
            if model.compute_trainable_for_k(try_k) <= 50000:
                selected_k = try_k
                break
        # freeze with selected k (<=50k params)
        model.freeze_all_but_last_k(k=selected_k)  # now asserts ok

        # loss fn per task
        loss_fn = nn.CrossEntropyLoss() if self.method == 'classification' else nn.MSELoss()

        # run algorithms, log history, and plot curves
        test_losses = {}  # to store test_loss per algorithm
        for algo in [rhc, sa, ga]:
            # get algo default params for caption
            algo_params = {
                'rhc': {'restarts': 5, 'initial_perturb_scale': 0.1, 'decay_rate': 0.99},
                'sa': {'initial_temp': 10.0, 'step_size': 0.1},
                'ga': {'pop_size': 50, 'mutation_rate': 0.1, 'mutation_std': 0.001}
            }[algo.__name__]
            caption = f"Settings: restarts={algo_params.get('restarts', 'N/A')}, scale={algo_params.get('initial_perturb_scale', 'N/A')}, decay={algo_params.get('decay_rate', 'N/A')}" if algo.__name__ == 'rhc' else \
                      f"Settings: initial_temp={algo_params.get('initial_temp', 'N/A')}, step_size={algo_params.get('step_size', 'N/A')}" if algo.__name__ == 'sa' else \
                      f"Settings: pop_size={algo_params.get('pop_size', 'N/A')}, mutation_rate={algo_params.get('mutation_rate', 'N/A')}, mutation_std={algo_params.get('mutation_std', 'N/A')}"

            optimized_model, history = algo(model, val_loader, loss_fn, device, logger=self.ml_logger)
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
                           save_path=f"{self.save_path}/{algo.__name__}_best_so_far.png", log_x=True, caption=caption)

            # evaluate on test for generalization
            with self.ml_logger.log_step(f"{algo.__name__} Test Evaluation") as step_info:
                test_loss = validation_objective(get_trainable_params(optimized_model), optimized_model, test_loader, loss_fn, device)
                step_info.update({'test_loss': test_loss})
                self.ml_logger.log_metric('test_loss', test_loss)
                test_losses[algo.__name__] = test_loss

        # plot test loss comparison as a curve
        if test_losses:
            os.makedirs(self.save_path, exist_ok=True)
            x_indices = np.arange(len(test_losses))  # Dummy x-axis for each algorithm
            self.ml_logger.log_learning_curve({'comparison': 'test_losses', 'algorithms': list(test_losses.keys()), 'values': list(test_losses.values())})
            plot_epoch_curve(x_indices, list(test_losses.values()), title="Test Loss Comparison Across RO Algorithms",  # type: ignore
                           save_path=f"{self.save_path}/test_loss_comparison.png", log_x=False, caption="Curve comparing test loss after RO optimization")

        # generate execution report after all RO steps
        self.ml_logger.generate_log_report(output_file=f"{self.save_path}/execution_report.txt")
        

    def update_logs(self):
        self.ml_logger.set_experiment_context(
            dataset=self.dataset,
            target=self.target,
            method=self.method,
            subsample=self.subsample,
            seed=self.seed
        )