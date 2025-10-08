import os
import warnings
warnings.filterwarnings("ignore", message="resource_tracker")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.neural_network")  # Suppress NN batch_size/convergence warnings


import torch
import torch.nn as nn
from src.models import freeze_all_but_last_k
from src.ro import rhc, sa, ga

from src.utils.logger import MLLogger
from src.data_processing import  load_or_process_data
from src.utils.plotter import plot_complexity_curve, plot_feature_importance, plot_model_evaluation, plot_pruning_path, plot_epoch_curve
from src.utils.tuning_config import get_primary_params, get_tuning_grid

class ModelingExperiment:
    def __init__(self, id, dataset, target, method, model, subsample, seed, tuning=True, cv_splits=5, combination_cap=50, best_params={}):
        self.experiment_id = id
        self.dataset = dataset
        self.target = target
        self.method = method
        self.model = model
        self.subsample = subsample
        self.seed = seed
        self.tuning = tuning
        self.cv_splits = cv_splits
        self.combination_cap = combination_cap
        self.best_params = best_params
        self.save_path = f"figures/{self.dataset}/{self.model}"

        self.ml_logger = MLLogger()
        self.ml_logger.current_logs = []
        self.update_logs()

    # Load or process data
    def get_data(self):
        with self.ml_logger.log_step("Load Data") as step_info:
            train_loader, val_loader, test_loader, data_info = load_or_process_data(self.dataset, self.target, self.method, self.subsample, self.seed)
            print(data_info)
            step_info.update(data_info) # type: ignore
        return train_loader, val_loader, test_loader


    def run_part1_ro(self, max_evals=10000, k_layers=2, reg_lambda=0.0):
        print(f"\nRunning Part 1 RO on {self.dataset} with {self.model} (frozen last {k_layers} layers)...")

        train_loader, val_loader, test_loader = self.get_data()

        # Define criterion based on method
        if self.method == "classification":
            criterion = nn.CrossEntropyLoss()
        elif self.method == "regression":
            criterion = nn.MSELoss()
        else:
            raise ValueError("Invalid method")

        # Freeze model
        self.model = freeze_all_but_last_k(self.model, k=k_layers)
        device = next(self.model.parameters()).device

        # Helper to get/set trainable params
        def get_trainable_params(model):
            return torch.cat([p.view(-1) for p in model.parameters() if p.requires_grad]).cpu().numpy()

        def set_trainable_params(model, flat_params_np):
            flat_params = torch.from_numpy(flat_params_np).to(device)
            offset = 0
            for p in model.parameters():
                if p.requires_grad:
                    numel = p.numel()
                    p.data.copy_(flat_params[offset:offset + numel].view_as(p))
                    offset += numel

        # Objective: val loss + L2 reg if lambda >0
        def objective(flat_params_np):
            set_trainable_params(self.model, flat_params_np)
            self.model.eval()
            loss = 0.0
            with torch.no_grad():
                for data, target in val_loader:
                    out = self.model(data)
                    loss += criterion(out, target).item()
            # Add L2 reg if evaluating "with regularization"
            if reg_lambda > 0:
                l2_reg = sum(p.norm(2).item()**2 for p in self.model.parameters() if p.requires_grad)
                loss += reg_lambda * l2_reg
            return loss / len(val_loader)

        init_params = get_trainable_params(self.model)

        # Run RHC
        with self.ml_logger.log_step("RHC") as step_info:
            rhc_best, rhc_eval, rhc_hist = rhc(objective, init_params, max_evals)
            step_info.update({'best_params': rhc_best, 'best_loss': rhc_eval, 'history': rhc_hist})  # type: ignore

        # Run SA
        with self.ml_logger.log_step("SA") as step_info:
            sa_best, sa_eval, sa_hist = sa(objective, init_params, max_evals)
            step_info.update({'best_params': sa_best, 'best_loss': sa_eval, 'history': sa_hist})  # type: ignore

        # Run GA
        with self.ml_logger.log_step("GA") as step_info:
            ga_best, ga_eval, ga_hist = ga(objective, init_params, max_evals)
            step_info.update({'best_params': ga_best, 'best_loss': ga_eval, 'history': ga_hist})  # type: ignore

        # Plot histories, etc., using plotter.py for curves vs. evals
        # self.ml_logger.save_logs() or similar

    def run(self):
        print(f"\n\nExecuting  {self.model}  {self.method}  on  {self.dataset}...".upper())

        X_train, X_test, y_train, y_test = self.get_data()
       
            
    def update_logs(self):
        self.ml_logger.set_experiment_context(
            dataset=self.dataset,
            target=self.target,
            method=self.method,
            model=self.model,
            subsample=self.subsample,
            seed=self.seed,
            tuning=self.tuning,
            cv_splits=self.cv_splits,
            combination_cap=self.combination_cap
        )