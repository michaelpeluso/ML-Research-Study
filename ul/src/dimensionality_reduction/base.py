import os
import re
import time
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Optional

from utils.plotter import plot_scatter, plot_scatter_multicomponent_rgb, stitch_specific_images
from utils.logger import MLLogger, print_t as print
from utils.cache_manager import CacheManager


class BaseDR(ABC):
    """Abstract base class for dimensionality reduction algorithms."""
    
    # Configuration overridden by subclasses
    algorithm_name = 'DR'
    
    def __init__(self, dataset: str, save_path: str, ml_logger: MLLogger | None = None, seed: int = 42):
        """Initialize BaseDR."""
        self.dataset = dataset
        self.base_save_path = save_path
        self.save_path = os.path.join(save_path, self.algorithm_name.lower().replace(' ', '_'))
        self.ml_logger = ml_logger or MLLogger()
        self.seed = seed
        
        self.cache_manager = CacheManager()  # Use centralized cache manager
        os.makedirs(self.save_path, exist_ok=True)
    
    @abstractmethod
    def fit_transform(self, X, n_components) -> tuple[Any, np.ndarray]:
        """Fit the DR model and transform X."""
        pass
    
    @abstractmethod
    def get_explained_variance(self, model) -> Optional[np.ndarray]:
        """Get explained variance or similar metric."""
        pass
    
    @abstractmethod
    def reconstruction_error(self, X_original, X_transformed, model) -> Optional[float]:
        """Compute reconstruction error"""
        pass
    
    @abstractmethod
    def plot_selection_metric(self, results):
        """Plot the algorithm-specific selection metric (variance/kurtosis/error)"""
        pass
    
    @abstractmethod
    def plot_component_interpretation(self, model):
        """Plot component interpretability heatmap (loadings/mixing/projection matrix)."""
        pass
    
    @abstractmethod
    def compute_selection_score(self, model, n_comp: int, recon_error: Optional[float]) -> tuple[float, str]:
        """Compute algorithm-specific selection score"""
        pass
    
    def evaluate_downstream_task(self, X_train, n_components_range):
        """Evaluate dimensionality reduction label-free"""
        print(f"Evaluating {self.algorithm_name} on {self.dataset} for components in {n_components_range}")
        
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
        total_components = len(valid_range)
        for idx, n_comp in enumerate(valid_range):
            print(f"  Evaluating n_components={n_comp}")
            
            # Transform data
            model, X_transformed = self.fit_transform(X_train, n_components=n_comp)
            
            # Algorithm-specific unsupervised metrics
            recon_error = self.reconstruction_error(X_train, X_transformed, model)
            explained_var = self.get_explained_variance(model)
            
            # Delegate to subclass for algorithm-specific scoring
            mean_score, scoring_metric = self.compute_selection_score(
                model, n_comp, recon_error
            )
            
            result = {
                'n_components': n_comp,
                'mean_score': float(mean_score),
                'explained_variance': explained_var,
                'reconstruction_error': recon_error,
                'model': model,
                'X_transformed': X_transformed
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
            
            if recon_error is not None: print(f"    {scoring_metric}: {recon_error:.4f}")
            else: print(f"    {scoring_metric}: {mean_score:.4f}")
        
        return results
    
    def run_dimensionality_reduction(self, X_train, y_train, n_components_range, task='classification'):
        """Main entry point for dimensionality reduction analysis."""
        # Check cache
        params = {'seed': self.seed, 'n_components_range': n_components_range, 'task': task, 'data_shape': X_train.shape}
        cached_result = self.cache_manager.load(self.dataset, f'{self.algorithm_name.lower()}_dr', params)
        if cached_result is not None: return cached_result
    
        # Main
        start_log_index = len(self.ml_logger.current_logs)
        with self.ml_logger.log_step(f"{self.algorithm_name} Dimensionality Reduction") as step_info:
            start_time = time.perf_counter()
            
            # Evaluate across component range
            results = self.evaluate_downstream_task(X_train, n_components_range)
            
            elapsed = time.perf_counter() - start_time
            
            # Select best n_components based on algorithm-specific criteria
            # Goal: Find sweet spot - minimum components with good score
            max_score = max(r['mean_score'] for r in results)
            
            # For variance-based (PCA): 90% of max variance is good enough
            # For kurtosis-based (ICA): 85% of max kurtosis is good enough  
            # For error-based (RP): Use different logic (minimize error with fewer components)
            if self.algorithm_name == 'PCA':
                target_threshold = 0.90  # 90% of maximum variance
            elif self.algorithm_name == 'ICA':
                target_threshold = 0.85  # 85% of maximum kurtosis
            else:  # RP and others
                target_threshold = 0.95  # 95% of best (minimum) error
            
            # Find minimum n_components that reaches target threshold (as % of max_score)
            target_score = target_threshold * max_score
            candidates = [r for r in results if r['mean_score'] >= target_score]
            
            if candidates:
                best_result = min(candidates, key=lambda r: r['n_components'])
                pct_of_max = (best_result['mean_score'] / max_score) * 100
                print(f"Selected n_components: {best_result['n_components']} (score={best_result['mean_score']:.4f}, {pct_of_max:.1f}% of max={max_score:.4f})")
            else:
                # Fallback: if threshold too strict, take middle ground
                median_idx = len(results) // 2
                best_result = sorted(results, key=lambda r: r['n_components'])[median_idx]
                print(f"Using median n_components: {best_result['n_components']} (threshold {target_threshold*100:.0f}% not reached)")
            
            print(f"Best n_components: {best_result['n_components']} with score: {best_result['mean_score']:.4f}")
            
            # Plots
            self.plot_projection(best_result, best_result['X_transformed'], results, y_train)
        
            step_info.update({
                'n_samples': X_train.shape[0],
                'n_features': X_train.shape[1],
                'n_components_range': n_components_range if isinstance(n_components_range, tuple) else (n_components_range, n_components_range),
                'task': task,
                'algorithm': self.algorithm_name,
                'seed': self.seed,
                'whiten': True if self.algorithm_name in ['PCA', 'ICA'] else False,
                'total_time': elapsed,
                'n_components_tested': len(n_components_range),
                'best_n_components': best_result['n_components'],
                'best_score': best_result['mean_score'],
                'results': results
            })
        
        self.ml_logger.generate_log_report(output_file=f"{self.save_path}/execution_report.txt", start_index=start_log_index)
        
        # Return model, n_components, transformed data, and all results
        result = {
            'model': best_result['model'],
            'n_components': best_result['n_components'],
            'X_transformed': best_result['X_transformed'],
            'results': results,
            'best_result': best_result
        }
        
        params = {'seed': self.seed, 'n_components_range': n_components_range, 'task': task, 'data_shape': X_train.shape}
        self.cache_manager.save(self.dataset, f'{self.algorithm_name.lower()}_dr', result, params)
        
        return result
    
    def plot_projection(self, best_result, X_transformed, results, y_train):
        best_model = best_result['model']
        
        # selection metric plot
        self.plot_selection_metric(results)
        
        # component interpretation heatmap (use all components)
        self.plot_component_interpretation(best_model)
        
        # 2d projection colored by target
        components_for_plot = None
        if hasattr(best_model, 'components_') and best_model.components_ is not None: components_for_plot = best_model.components_
        elif hasattr(best_model, 'mixing_') and best_model.mixing_ is not None: components_for_plot = best_model.mixing_
        xlabel, ylabel = f'{self.algorithm_name}1', f'{self.algorithm_name}2'
        n_components = best_result['n_components']
        title_2d = f'{self.algorithm_name} 2D Projection ({n_components} components) - {self.dataset.title()}'
        
        plot_scatter(
            X_transformed, y_train, self.algorithm_name, self.dataset, 
            save_path=os.path.join(self.save_path, "projection_2d.png"),
            xlabel=xlabel, ylabel=ylabel, components=components_for_plot, title=title_2d
        )
        if X_transformed.shape[1] >= 4:
            title_rgb = f'{self.algorithm_name} Multi-Component RGB Projection ({n_components} components) - {self.dataset.title()}\nRed=Comp3, Green=Comp4'
            plot_scatter_multicomponent_rgb(
                X_transformed, y_train, self.algorithm_name, self.dataset,
                save_path=os.path.join(self.save_path, "projection_2d_rgb.png"),
                comp_x=0, comp_y=1, comp_red=2, comp_green=3, title=title_rgb
            )

        self.stitch_all()
    
    
    def stitch_all(self, title: Optional[str] = None, output_name: str = "all_outputs_master.png"):
        """Stitch all PNG images in the main save_path into a single master plot."""
        png_files = [os.path.join(self.save_path, f) for f in os.listdir(self.save_path) if f.lower().endswith('.png') and not output_name.split('.')[0] in f]
        if not png_files:
            print(f"No PNG images found in {self.save_path} to stitch.")
            return
        output_path = os.path.join(self.save_path, output_name)
        stitch_specific_images(
            image_paths=png_files,
            title=title or f"All {self.algorithm_name} Plots for {self.dataset}",
            output_path=output_path
        )
        print(f"Stitched {len(png_files)} PNG images into {output_path}")


    def generate_dr_comparison_table(self, save_path):
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
                
            # Extract key metrics
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