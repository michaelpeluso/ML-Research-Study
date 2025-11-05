import os
import time
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Optional
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from utils.plotter import plot_curve, plot_component_heatmap, plot_dr_2d_projection, plot_dr_3d_projection, plot_multiple_y_axes, plot_ica_kurtosis
from utils.logger import MLLogger, print_t as print


class BaseDR(ABC):
    """Abstract base class for dimensionality reduction algorithms."""
    
    # Configuration overridden by subclasses
    algorithm_name = 'DR'
    
    def __init__(self, dataset: str, save_path: str, ml_logger: MLLogger | None = None, seed: int = 42):
        self.dataset = dataset
        self.save_path = save_path
        self.ml_logger = ml_logger or MLLogger()
        self.seed = seed
        os.makedirs(self.save_path, exist_ok=True)
    
    @abstractmethod
    def fit_transform(self, X, n_components, **kwargs) -> tuple[Any, np.ndarray]:
        """Fit the DR model and transform X."""
        pass
    
    @abstractmethod
    def get_explained_variance(self, model) -> Optional[np.ndarray]:
        """Get explained variance or similar metric."""
        pass
    
    def reconstruction_error(self, X_original, X_transformed, model) -> Optional[float]:
        """Compute reconstruction error"""
        return None
    
    def evaluate_downstream_task(self, X_train, y_train, n_components_range, task='classification'):
        """Evaluate how dimensionality affects downstream task performance using cross-validation."""
        print(f"Evaluating {self.algorithm_name} for downstream {task} task")
        
        # Convert tuple to list if needed
        if isinstance(n_components_range, tuple) and len(n_components_range) == 2:
            n_components_range = list(range(n_components_range[0], n_components_range[1]))
        
        # Validate n_components range
        max_components = min(X_train.shape[0], X_train.shape[1])
        valid_range = [n for n in n_components_range if n <= max_components]
        
        if not valid_range:
            print(f"Warning: No valid components in range. Max possible: {max_components}")
            valid_range = [min(n_components_range[0] if isinstance(n_components_range, tuple) else n_components_range[0], max_components)]
        
        if len(valid_range) < len(list(n_components_range)):
            print(f"Note: Limiting components to max={max_components} (was {max(n_components_range)})")
        
        results = []
        for n_comp in valid_range:
            print(f"  Evaluating n_components={n_comp}")
            
            # Transform data
            model, X_transformed = self.fit_transform(X_train, n_components=n_comp)
            
            # Use cross-validation for robust evaluation
            if task == 'classification':
                estimator = RandomForestClassifier(n_estimators=100, random_state=self.seed, n_jobs=-1)
                scoring = 'accuracy'
            else:
                estimator = RandomForestRegressor(n_estimators=100, random_state=self.seed, n_jobs=-1)
                scoring = 'r2'
            
            scores = cross_val_score(estimator, X_transformed, y_train, cv=5, scoring=scoring, n_jobs=-1)
            
            result = {
                'n_components': n_comp,
                'mean_score': float(np.mean(scores)),
                'std_score': float(np.std(scores)),
                'explained_variance': self.get_explained_variance(model),
                'reconstruction_error': self.reconstruction_error(X_train, X_transformed, model),
                'model': model,  # Store model to avoid refitting
                'X_transformed': X_transformed  # Store transformed data
            }
            
            # Pre-compute cumulative variance if available (avoid recalculation in plotting)
            if result['explained_variance'] is not None:
                var = result['explained_variance']
                if isinstance(var, (list, np.ndarray)):
                    result['cumulative_variance'] = float(np.sum(var[:n_comp]))
                else:
                    result['cumulative_variance'] = float(var)
            else:
                result['cumulative_variance'] = None
            
            results.append(result)
            
            print(f"    {scoring}: {result['mean_score']:.4f} ± {result['std_score']:.4f}")
        
        return results
    
    def plot_all_metrics(self, results, metric_name='Score', save_path=None):
        """Plot all metrics on a single multi-y-axis chart with optional elbow detection."""
        n_components = [r['n_components'] for r in results]
        
        # Collect all available metrics
        y_series = []
        labels = []
        
        # Downstream performance
        scores = [r['mean_score'] for r in results]
        y_series.append(scores)
        labels.append(f'{metric_name}')
        
        # Cumulative variance (if available)
        cumulative = None
        if not all(r.get('cumulative_variance') is None for r in results):
            cumulative = [r.get('cumulative_variance', 0) for r in results]
            y_series.append(cumulative)
            labels.append('Cumulative Variance')
        
        # Reconstruction error (if available)
        errors = [r['reconstruction_error'] for r in results if r['reconstruction_error'] is not None]
        if errors and not all(e is None for e in errors):
            y_series.append(errors)
            labels.append('Reconstruction Error (MSE)')
        
        # Detect elbow for PCA
        elbow_component = None
        elbow_label = None
        if self.algorithm_name == 'PCA' and cumulative is not None and len(cumulative) > 2:
            # Use cumulative variance for elbow detection
            diffs = np.diff(cumulative)
            elbow_idx = np.argmax(np.abs(np.diff(diffs))) + 1
            if 0 < elbow_idx < len(n_components):
                elbow_component = n_components[elbow_idx]
                elbow_label = f'Elbow at n={elbow_component}'
        
        if save_path is None:
            save_path = os.path.join(self.save_path, "all_metrics_combined.png")
        
        plot_multiple_y_axes(
            x=n_components,
            y_series=y_series,
            labels=labels,
            xlabel='Number of Components',
            title=f'{self.algorithm_name}: All Metrics on {self.dataset}',
            save_path=save_path,
            vline_x=elbow_component,
            vline_label=elbow_label
        )
    
    def plot_variance_explained(self, results, save_path=None):
        """Plot variance explained vs number of components."""
        n_components = [r['n_components'] for r in results]
        
        # Check if any result has explained variance
        if all(r.get('cumulative_variance') is None for r in results):
            print(f"No explained variance available for {self.algorithm_name}")
            return
        
        # Use pre-computed cumulative variance
        cumulative = [r.get('cumulative_variance', 0) for r in results]
        
        if save_path is None:
            save_path = os.path.join(self.save_path, "variance_explained.png")
        
        plot_curve(
            x=n_components,
            y_list=cumulative,
            labels=['Cumulative Variance'],
            xlabel='Number of Components',
            ylabel='Variance Explained',
            title=f'{self.algorithm_name}: Variance Explained on {self.dataset}',
            save_path=save_path,
            colors=['blue'],
            marker='o'
        )
    
    def plot_downstream_performance(self, results, metric_name='Score', save_path=None):
        """Plot downstream task performance vs number of components."""
        n_components = [r['n_components'] for r in results]
        scores = [r['mean_score'] for r in results]
        stds = [r['std_score'] for r in results]
        
        if save_path is None:
            save_path = os.path.join(self.save_path, "downstream_performance.png")
        
        plot_curve(
            x=n_components,
            y_list=scores,
            labels=[f'{metric_name} (CV)'],
            xlabel='Number of Components',
            ylabel=metric_name,
            title=f'{self.algorithm_name}: Downstream Task Performance on {self.dataset}',
            save_path=save_path,
            colors=['green'],
            marker='o'
        )
    
    def plot_reconstruction_error(self, results, save_path=None):
        """Plot reconstruction error vs number of components."""
        n_components = [r['n_components'] for r in results]
        errors = [r['reconstruction_error'] for r in results if r['reconstruction_error'] is not None]
        
        if not errors or all(e is None for e in errors):
            print(f"No reconstruction error available for {self.algorithm_name}")
            return
        
        if save_path is None:
            save_path = os.path.join(self.save_path, "reconstruction_error.png")
        
        plot_curve(
            x=n_components[:len(errors)],
            y_list=errors,
            labels=['Reconstruction Error'],
            xlabel='Number of Components',
            ylabel='MSE',
            title=f'{self.algorithm_name}: Reconstruction Error on {self.dataset}',
            save_path=save_path,
            colors=['orange'],
            marker='o'
        )
    
    def run_dimensionality_reduction(self, X_train, y_train, n_components_range, task='classification'):
        """Main entry point for dimensionality reduction analysis."""
        start_log_index = len(self.ml_logger.current_logs)
        with self.ml_logger.log_step(f"{self.algorithm_name} Dimensionality Reduction") as step_info:
            start_time = time.perf_counter()
            
            # Evaluate across component range
            results = self.evaluate_downstream_task(X_train, y_train, n_components_range, task)
            
            # Create individuals directory for individual metric plots
            individuals_path = os.path.join(self.save_path, "individuals")
            os.makedirs(individuals_path, exist_ok=True)
            
            # Generate combined multi-y-axis plot (overview)
            self.plot_all_metrics(results, metric_name='Accuracy' if task == 'classification' else 'R²')
            
            # Generate individual plots in individuals/ folder
            self.plot_variance_explained(results, save_path=os.path.join(individuals_path, "variance_explained.png"))
            self.plot_downstream_performance(results, metric_name='Accuracy' if task == 'classification' else 'R²',
                                            save_path=os.path.join(individuals_path, "downstream_performance.png"))
            self.plot_reconstruction_error(results, save_path=os.path.join(individuals_path, "reconstruction_error.png"))
            
            # PCA-specific: compute variance thresholds
            if self.algorithm_name == 'PCA':
                # Import PCA class to call its method
                from dimensionality_reduction.pca import PCAReduction
                if isinstance(self, PCAReduction):
                    variance_thresholds = self.compute_variance_thresholds(results)
                else:
                    variance_thresholds = {}
            else:
                variance_thresholds = {}
            
            # Log results
            elapsed = time.perf_counter() - start_time
            best_result = max(results, key=lambda r: r['mean_score'])
            
            step_info.update({
                'total_time': elapsed,
                'n_components_tested': len(n_components_range),
                'best_n_components': best_result['n_components'],
                'best_score': best_result['mean_score'],
                'variance_thresholds': variance_thresholds,
                'results': results
            })
            
            print(f"Best n_components: {best_result['n_components']} with score: {best_result['mean_score']:.4f}")
            if variance_thresholds:
                print(f"Variance thresholds: {variance_thresholds}")
            print(f"Total execution time: {elapsed:.2f}s")
            
            # Reuse best model and transformed data from results (no refitting)
            best_model = best_result['model']
            X_transformed = best_result['X_transformed']
            
            # Component heatmap (PCA and ICA only)
            n_comp = min(10, best_result['n_components'])
            if self.algorithm_name == 'PCA':
                components = best_model.components_[:n_comp]
                labels = [f'PC{i+1}' for i in range(n_comp)]
                plot_component_heatmap(
                    data=components,
                    component_labels=labels,
                    dataset_name=self.dataset,
                    title='PCA Component Loadings',
                    colorbar_label='Loading Magnitude',
                    save_path=os.path.join(self.save_path, "component_heatmap.png")
                )
            elif self.algorithm_name == 'ICA':
                mixing = best_model.mixing_[:, :n_comp].T
                labels = [f'IC{i+1}' for i in range(n_comp)]
                plot_component_heatmap(
                    data=mixing,
                    component_labels=labels,
                    dataset_name=self.dataset,
                    title='ICA Mixing Matrix',
                    colorbar_label='Mixing Coefficient',
                    save_path=os.path.join(self.save_path, "component_heatmap.png")
                )
                
                # ICA-specific: kurtosis plot showing component independence
                plot_ica_kurtosis(
                    X_transformed, 
                    self.dataset, 
                    save_path=os.path.join(self.save_path, "ica_kurtosis.png")
                )
                
                # ICA-specific: correlation matrix (should be near-identity for independent components)
                corr_matrix = np.corrcoef(X_transformed.T)
                ic_labels = [f'IC{i+1}' for i in range(corr_matrix.shape[0])]
                plot_component_heatmap(
                    data=corr_matrix,
                    component_labels=ic_labels,
                    dataset_name=self.dataset,
                    title='ICA Component Correlation Matrix (should be near-identity)',
                    colorbar_label='Correlation',
                    save_path=os.path.join(self.save_path, "ica_correlation_matrix.png"),
                    xlabel='Independent Component',
                    ylabel='Independent Component'
                )
            # RP doesn't have interpretable components, skip heatmap
            
            # 2d component vectors for projection plots
            components_for_plot = None
            if self.algorithm_name == 'PCA':
                components_for_plot = best_model.components_.T
                xlabel, ylabel = 'First Principal Component (PC1)', 'Second Principal Component (PC2)'
                zlabel = 'Third Principal Component (PC3)'
            elif self.algorithm_name == 'ICA':
                components_for_plot = best_model.mixing_
                xlabel, ylabel = 'First Independent Component (IC1)', 'Second Independent Component (IC2)'
                zlabel = 'Third Independent Component (IC3)'
            else:
                xlabel, ylabel = f'First {self.algorithm_name} Component', f'Second {self.algorithm_name} Component'
                zlabel = f'Third {self.algorithm_name} Component'
            
            plot_dr_2d_projection(
                X_transformed, y_train, self.algorithm_name, self.dataset, 
                save_path=os.path.join(self.save_path, "projection_2d.png"),
                xlabel=xlabel, ylabel=ylabel, components=components_for_plot
            )
            
            # 3D projection if we have at least 3 components
            if X_transformed.shape[1] >= 3:
                plot_dr_3d_projection(
                    X_transformed, y_train, self.algorithm_name, self.dataset,
                    save_path=os.path.join(self.save_path, "projection_3d.png"),
                    xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, components=components_for_plot
                )
        
        self.ml_logger.generate_log_report(output_file=f"{self.save_path}/execution_report.txt", start_index=start_log_index)
        
        return best_model, best_result['n_components']


def generate_dr_comparison_table(save_path):
    """Generate a comparison table across all DR methods."""
    import pandas as pd
    
    methods = ['pca', 'ica', 'rp']
    table_data = []
    
    for method in methods:
        report_path = os.path.join(save_path, f"dr/{method}/execution_report.txt")
        if not os.path.exists(report_path):
            continue
        
        # Parse execution report for key metrics
        with open(report_path, 'r') as f:
            content = f.read()
            
        # Extract key metrics (basic parsing)
        import re
        best_n = re.search(r'Best N Components: (\d+)', content)
        best_score = re.search(r'Best Score: ([\d.]+)', content)
        total_time = re.search(r'Total Time: ([\d.]+)', content)
        
        row = {
            'Method': method.upper(),
            'Best_n_components': int(best_n.group(1)) if best_n else 'N/A',
            'Best_Score': float(best_score.group(1)) if best_score else 'N/A',
            'Time_seconds': float(total_time.group(1)) if total_time else 'N/A'
        }
        table_data.append(row)
    
    if table_data:
        df = pd.DataFrame(table_data)
        table_path = os.path.join(save_path, "dr_comparison_table.csv")
        df.to_csv(table_path, index=False)
        print(f"\nDR Comparison Table:\n{df.to_string(index=False)}")
        print(f"Saved to {table_path}\n")
        return df
    
    return None