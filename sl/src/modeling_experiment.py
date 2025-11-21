import os
import warnings
warnings.filterwarnings("ignore", message="resource_tracker")

from src.utils.logger import MLLogger
from src.data_processing import get_col_types, load_or_process_data
from src.tuning import tune_parameter_grid, train_nn_with_curves
from src.utils.diagnostics import get_hardware_info, get_model_info, measure_runtime
from src.evaluation import compute_complexity_curve, compute_learning_curve, evaluate_test_set
from src.pipelines import build_column_transformer, build_pipeline
from src.utils.plotter import plot_complexity_curve, plot_feature_importance, plot_model_evaluation, plot_pruning_path, plot_epoch_curve
from src.utils.tuning_config import get_primary_params, get_tuning_grid

class ModelingExperiment:
    def __init__(self, id, dataset, target, method, model, subsample, seed, tuning=True, cv_splits=5, combination_cap=50, best_params=None):
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
        self.network_arc = None
        self.save_path = f"figures/{self.dataset}/{self.model}"

        self.ml_logger = MLLogger()
        self.ml_logger.current_logs = []
        self.update_logs()
        
    
    def run(self):
        print(f"\n\nExecuting  {self.model}  {self.method}  on  {self.dataset}...".upper())
        # 1. Load or process data
        with self.ml_logger.log_step("Load Data") as step_info:
            X_train, X_test, y_train, y_test, data_info = load_or_process_data(self.dataset, self.target, self.method, self.subsample, self.seed)
            step_info.update(data_info)

        # 2. Build initial pipeline
        with self.ml_logger.log_step("Pipeline Building") as step_info:
            numeric_cols, categorical_cols, high_card_cols = get_col_types(X_train, self.dataset)
            transformer = build_column_transformer(numeric_cols, categorical_cols, high_card_cols)
            pipeline = build_pipeline(transformer, self.model, self.method, self.seed)
            pipeline_info = {
                'samples': len(X_train),
                'features': len(X_train.columns),
                'numeric_features': len(numeric_cols),
                'categorical_features': len(categorical_cols),
                'high_cardinality_features': len(high_card_cols) if high_card_cols else 0,'model_type': self.model
                }
            step_info.update(pipeline_info)

        # 3. Tuning if enabled
        if self.tuning or self.best_params is None:
            with self.ml_logger.log_step("Hyperparameter Tuning") as step_info:
                param_grid = get_tuning_grid(self.method, self.model)
                if self.model == "nn" and self.network_arc is not None: 
                    param_grid["estimator__hidden_layer_sizes"] = self.network_arc
                    self.save_path = f"figures/{self.dataset}/{self.model}_{len(self.network_arc)}"
                if param_grid: 
                    pipeline, best_params = tune_parameter_grid(self.dataset, pipeline, X_train, y_train, param_grid, self.cv_splits, self.combination_cap, self.seed)
                else: 
                    print(f"No tuning parameters defined for {self.model}")
        else: best_params = self.best_params

        print("Refitting pipeline with best parameters.")
        pipeline.set_params(**best_params)
        pipeline.fit(X_train, y_train)
        tuning_info = {'parameter_count': len(best_params),'best_params':best_params}
        step_info.update(tuning_info)

        # 4. Runtime Measurement
        with self.ml_logger.log_step("Runtime Measurement") as step_info:
            runtime_data = measure_runtime(pipeline, X_train, y_train, X_test)
            hardware_info = get_hardware_info()
            model_info = get_model_info(pipeline)
            data_info = {"train_samples": len(X_train),"test_samples": len(X_test),"n_features": X_train.shape[1],"memory_usage_mb": data_info.get('memory_usage_mb', 0)}
            os.makedirs(self.save_path, exist_ok=True)
            step_info.update({**runtime_data, **hardware_info, **model_info})

        # 5. Model Complexity Data
        with self.ml_logger.log_step("Model Complexity Data") as step_info:
            param_name = get_primary_params(self.model)
            if param_name:
                complexity_data = compute_complexity_curve(pipeline, X_train, y_train, param_name, n_splits=self.cv_splits, seed=self.seed)
                os.makedirs(self.save_path, exist_ok=True)
                complexity_scoring = 'R2' if self.method=="regression" else 'Weighted F1'
                plot_complexity_curve(complexity_data, scoring=complexity_scoring, save_path=f"{self.save_path}/complexity_curve.png")
                step_info.update({"complexity_param": param_name})

        # 6. Compute learning data
        with self.ml_logger.log_step("Compute Learning Data") as step_info:    
            scoring = "f1_weighted" if self.method == "classification" else "r2"
            train_sizes, train_scores, val_scores = compute_learning_curve(pipeline, X_train, y_train, scoring, self.seed) 
            curve_info = {'learning_curve_points': len(train_sizes)}
            step_info.update(curve_info)
            
        # 7. Evaluate on test set
        with self.ml_logger.log_step("Evaluate Model") as step_info: 
            metrics, y_pred = evaluate_test_set(pipeline, X_test, y_test)
            eval_info = {'test_metrics': metrics}
            step_info.update(eval_info)

        # 8. Decision Tree specific Pruning path
        if self.model == "dt":
            with self.ml_logger.log_step("DT Pruning Path") as step_info:
                os.makedirs(self.save_path, exist_ok=True)
                step_info.update({"best_ccp_alpha": plot_pruning_path(pipeline, X_train, y_train, save_path=self.save_path)})

        # 9. NN-specific epoch curve
        if self.model == "nn":
            with self.ml_logger.log_step("NN Training with Epoch Curves") as step_info:
                train_losses, val_losses = train_nn_with_curves(pipeline, X_train, y_train, self.method, self.seed)
                os.makedirs(self.save_path, exist_ok=True)
                plot_epoch_curve(train_losses, val_losses, save_path=f"{self.save_path}/epoch_curve.png")
                step_info.update({
                    "epochs_run": len(train_losses),
                    "final_train_loss": train_losses[-1] if train_losses else None,
                    "final_val_loss": val_losses[-1] if val_losses else None
                })

        # 10. Generate plots
        with self.ml_logger.log_step("Generate Plots") as step_info:
            estimator = pipeline.named_steps["estimator"]
            if self.model == "dt":
                importances = estimator.feature_importances_
                feature_names = X_train.columns.tolist()
                plot_feature_importance(dict(zip(feature_names, importances)), n_count=estimator.tree_.node_count, save_path=f"{self.save_path}/feature_importance.png")

            y_proba = pipeline.predict_proba(X_test) if self.method == "classification" and hasattr(estimator, "predict_proba") else None
            class_names = [str(c) for c in estimator.classes_] if hasattr(estimator, "classes_") else None
            name_dict = {"dt":"Decision Tree", "knn": "K Nearest Neighbor", "lsvm": "Linear SVM", "svm": "Kernel SVM", "nn": "Neural Network"}
            stitched_title = f"{name_dict[self.model]} model on {self.dataset.capitalize()} dataset"
            plot_model_evaluation(y_test, y_proba, y_pred, metrics, class_names, train_sizes, train_scores, val_scores, scoring, self.save_path, stitched_title)
            self.ml_logger.generate_log_report(output_file=f"{self.save_path}/execution_report.txt")
            
    
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