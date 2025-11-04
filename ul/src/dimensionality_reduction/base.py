import os
import time
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Optional
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from utils.plotter import plot_curve
from utils.logger import MLLogger, print_t as print


class BaseDR(ABC):
    """Abstract base class for dimensionality reduction algorithms."""
    
    # Configuration - subclasses should override these class attributes
    algorithm_name = 'DR'
    
    def __init__(self, dataset: str, save_path: str, ml_logger: MLLogger | None = None, seed: int = 42):
        self.dataset = dataset
        self.save_path = save_path
        self.ml_logger = ml_logger or MLLogger()  # Auto-create if not provided
        self.seed = seed
        os.makedirs(self.save_path, exist_ok=True)
    
    @abstractmethod
    def fit_transform(self, X, n_components, **kwargs) -> tuple[Any, np.ndarray]:
        """Fit the DR model and transform X. Must be implemented by subclasses.  """
        pass
    
    @abstractmethod
    def get_explained_variance(self, model) -> Optional[np.ndarray]:
        """Get explained variance or similar metric. Return None if not applicable."""
        pass
    
    def reconstruction_error(self, X_original, X_transformed, model) -> Optional[float]:
        """Compute reconstruction error (if applicable)."""
        return None
    
    def evaluate_downstream_task(self, X_train, y_train, n_components_range, task='classification'):
        """Evaluate how dimensionality affects downstream task performance using cross-validation.
        
        Args:
            n_components_range: Tuple (start, end) or iterable of component counts to evaluate
        """
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
                'reconstruction_error': self.reconstruction_error(X_train, X_transformed, model)
            }
            results.append(result)
            
            print(f"    {scoring}: {result['mean_score']:.4f} ± {result['std_score']:.4f}")
        
        return results
    
    def plot_variance_explained(self, results, save_path=None):
        """Plot variance explained vs number of components."""
        n_components = [r['n_components'] for r in results]
        
        # Check if any result has explained variance
        if all(r['explained_variance'] is None for r in results):
            print(f"No explained variance available for {self.algorithm_name}")
            return
        
        variance = []
        cumulative = []
        for r in results:
            if r['explained_variance'] is not None:
                var = r['explained_variance']
                if isinstance(var, (list, np.ndarray)):
                    variance.append(np.sum(var[:r['n_components']]))
                    cumulative.append(np.sum(var[:r['n_components']]))
                else:
                    variance.append(var)
                    cumulative.append(var)
            else:
                variance.append(0)
                cumulative.append(0)
        
        if save_path is None:
            save_path = os.path.join(self.save_path, "variance_explained.png")
        
        plot_curve(
            x=n_components,
            y_list=[variance, cumulative],
            labels=['Explained Variance', 'Cumulative Variance'],
            xlabel='Number of Components',
            ylabel='Variance Explained',
            title=f'{self.algorithm_name}: Variance Explained on {self.dataset}',
            save_path=save_path,
            colors=['blue', 'red'],
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
        with self.ml_logger.log_step(f"{self.algorithm_name} Dimensionality Reduction") as step_info:
            start_time = time.perf_counter()
            
            # Evaluate across component range
            results = self.evaluate_downstream_task(X_train, y_train, n_components_range, task)
            
            # Generate plots
            self.plot_variance_explained(results)
            self.plot_downstream_performance(results, metric_name='Accuracy' if task == 'classification' else 'R²')
            self.plot_reconstruction_error(results)
            
            # Log results
            elapsed = time.perf_counter() - start_time
            best_result = max(results, key=lambda r: r['mean_score'])
            
            step_info.update({
                'total_time': elapsed,
                'n_components_tested': len(n_components_range),
                'best_n_components': best_result['n_components'],
                'best_score': best_result['mean_score'],
                'results': results
            })
            
            print(f"Best n_components: {best_result['n_components']} with score: {best_result['mean_score']:.4f}")
            print(f"Total execution time: {elapsed:.2f}s")
        
        self.ml_logger.generate_log_report(output_file=f"{self.save_path}/execution_report.txt")
        
        # Return model fitted with best n_components
        best_model, _ = self.fit_transform(X_train, n_components=best_result['n_components'])
        return best_model, best_result['n_components']
