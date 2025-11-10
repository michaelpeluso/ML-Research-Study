import numpy as np
import os
from sklearn.decomposition import FastICA

from dimensionality_reduction.base import BaseDR
from utils.plotter import plot_curve, plot_heatmap, plot_bar
from utils.logger import print_t as print


class ICAReduction(BaseDR):
    """Independent Component Analysis (ICA) for dimensionality reduction."""
    
    algorithm_name = 'ICA'
    
    def __init__(self, dataset: str, save_path: str, ml_logger=None, seed: int = 42, max_iter: int = 200, tol: float = 1e-4):
        """Initialize ICA with algorithm-specific parameters."""
        super().__init__(dataset, save_path, ml_logger, seed)
        self.max_iter = max_iter
        self.tol = tol
    
    def fit_transform(self, X, n_components):
        """Fit ICA and transform X with whitening disabled for consistency."""
        whiten = False  # Disable whitening to match other methods
        print(f"Fitting ICA with n_components={n_components}, whiten={whiten}, max_iter={self.max_iter}, tol={self.tol}")
        
        ica = FastICA(
            n_components=n_components,
            whiten=whiten,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.seed
        )
        X_transformed = ica.fit_transform(X)
        
        # Compute kurtosis as a measure of non-Gaussianity (ICA objective)
        kurtosis_per_comp = [self._kurtosis(X_transformed[:, i]) for i in range(n_components)]
        kurtosis = np.mean(np.abs(kurtosis_per_comp))
        print(f"  Mean absolute kurtosis: {kurtosis:.4f}")
        
        # Store kurtosis in model for later retrieval (dynamic attributes)
        ica.mean_abs_kurtosis_ = kurtosis  # type: ignore[attr-defined]
        ica.kurtosis_per_component_ = kurtosis_per_comp  # type: ignore[attr-defined]
        
        return ica, X_transformed
    
    def _kurtosis(self, x):
        """Compute kurtosis (4th standardized moment) of a variable."""
        x_centered = x - np.mean(x)
        x_std = np.std(x)
        if x_std == 0:
            return 0
        return np.mean((x_centered / x_std) ** 4) - 3
    
    def get_explained_variance(self, model):
        """ICA doesn't have explained variance like PCA. Return None."""
        return None
    
    def get_kurtosis_score(self, model):
        """Get mean absolute kurtosis score for ICA (higher is better for independence)."""
        return getattr(model, 'mean_abs_kurtosis_', None)
    
    def reconstruction_error(self, X_original, X_transformed, model):
        """Compute reconstruction error for ICA."""
        # ICA mixing matrix
        X_reconstructed = np.dot(X_transformed, model.mixing_.T) + model.mean_
        mse = float(np.mean((X_original - X_reconstructed) ** 2))
        return mse
    
    def compute_selection_score(self, model, n_comp, recon_error):
        """ICA selection: Maximize mean absolute kurtosis (measure of non-Gaussianity)."""
        kurtosis_score = getattr(model, 'mean_abs_kurtosis_', None)
        if kurtosis_score is not None:
            score = kurtosis_score
            metric = 'mean_abs_kurtosis'
        elif recon_error is not None:
            score = -recon_error
            metric = 'reconstruction_error'
        else:
            score = 0
            metric = 'none'
        return score, metric
    
    def get_component_independence(self, model, X_transformed):
        """Measure independence of components using mutual information or correlation."""
        n_components = X_transformed.shape[1]
        
        # Correlation matrix (should be close to identity for independent components)
        corr_matrix = np.corrcoef(X_transformed.T)
        
        # Average absolute off-diagonal correlation (lower is better)
        mask = ~np.eye(n_components, dtype=bool)
        avg_corr = float(np.mean(np.abs(corr_matrix[mask])))
        
        # Kurtosis per component (measure of non-Gaussianity)
        kurtosis = [self._kurtosis(X_transformed[:, i]) for i in range(n_components)]
        
        return {
            'avg_correlation': avg_corr,
            'kurtosis_per_component': kurtosis,
            'mean_abs_kurtosis': float(np.mean(np.abs(kurtosis)))
        }
    
    def plot_selection_metric(self, results):
        """Plot ICA kurtosis curve (REQUIRED for Step 2)."""
        n_components = [r['n_components'] for r in results]
        scores = [r['mean_score'] for r in results]
        
        plot_curve(
            x=n_components,
            y_list=scores,
            labels=None,  # Single series - no legend needed
            xlabel='Number of Components',
            ylabel='Mean Absolute Kurtosis',
            title=f'ICA: Kurtosis vs Components on {self.dataset}',
            save_path=os.path.join(self.save_path, "kurtosis_vs_components.png"),
            colors=['blue'],
            marker='o'
        )
    
    def plot_component_interpretation(self, model):
        """Plot ICA mixing matrix heatmap and kurtosis bar plot (REQUIRED for Step 2)."""
        n_comp = model.mixing_.shape[1]  # Use all components
        mixing = model.mixing_.T
        labels = [f'IC{i+1}' for i in range(n_comp)]
        
        # 1. Heatmap of mixing matrix
        plot_heatmap(
            data=mixing,
            title=f'ICA Mixing Matrix ({self.dataset})',
            save_path=os.path.join(self.save_path, "component_heatmap.png"),
            xlabel='Feature Index',
            ylabel='Component',
            row_labels=labels,
            colorbar_label='Mixing Coefficient',
            cmap='RdBu_r'
        )
        
        # 2. Bar plot of per-component kurtosis
        # Kurtosis is computed and stored during fit_transform
        if not hasattr(model, 'kurtosis_per_component_'):
            raise ValueError("Model does not have kurtosis_per_component_ attribute. "
                           "Ensure fit_transform was called and kurtosis was computed.")
        
        kurtosis_per_comp = model.kurtosis_per_component_
        
        # Sort by absolute kurtosis (higher = more non-Gaussian = more independent)
        abs_kurtosis = np.abs(kurtosis_per_comp)
        sorted_indices = np.argsort(abs_kurtosis)[::-1]
        sorted_kurtosis = [kurtosis_per_comp[i] for i in sorted_indices]
        sorted_labels = [f'IC{i+1}' for i in sorted_indices]
        
        kurtosis_bar_path = os.path.join(self.save_path, "kurtosis_per_component.png")
        plot_bar(
            x_labels=sorted_labels,
            values=sorted_kurtosis,
            xlabel='Component',
            ylabel='Kurtosis',
            title=f'ICA Component Kurtosis (Non-Gaussianity)\nHigher = More Independent',
            save_path=kurtosis_bar_path,
            color='steelblue'
        )
        
        print(f"  Generated ICA component kurtosis bar plot in {self.save_path}")
    
    def _get_projection_labels(self):
        """Get x/y labels for ICA projection plot."""
        return 'IC1', 'IC2'
