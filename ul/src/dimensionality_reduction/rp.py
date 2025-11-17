import numpy as np
import os
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

from dimensionality_reduction.base import BaseDR
from utils.plotter import plot_curve, plot_heatmap
from utils.logger import print_t as print


class RandomProjection(BaseDR):
    """Random Projection for dimensionality reduction using Johnson-Lindenstrauss lemma."""
    
    algorithm_name = 'Random Projection'
    
    def __init__(self, dataset: str, save_path: str, ml_logger=None, seed: int = 42, projection_type: str = 'gaussian', eps: float = 0.1):
        """Initialize Random Projection with algorithm-specific parameters."""
        super().__init__(dataset, save_path, ml_logger, seed)
        self.projection_type = projection_type
        self.eps = eps
    
    def fit_transform(self, X, n_components):
        """Fit Random Projection and transform X."""
        print(f"Fitting {self.projection_type.title()} Random Projection with n_components={n_components}, eps={self.eps}")
        
        if self.projection_type == 'gaussian':
            rp = GaussianRandomProjection(
                n_components=n_components,
                eps=self.eps,
                random_state=self.seed
            )
        else:  # sparse
            rp = SparseRandomProjection(
                n_components=n_components,
                eps=self.eps,
                random_state=self.seed,
                dense_output=False
            )
        
        X_transformed = rp.fit_transform(X)
        
        print(f"  Projected from {X.shape[1]} to {X_transformed.shape[1]} dimensions")
        
        return rp, X_transformed
    
    def get_explained_variance(self, model):
        """Random Projection doesn't have explained variance. Return None."""
        return None
    
    def reconstruction_error(self, X_original, X_transformed, model):
        """Compute reconstruction error using Moore-Penrose pseudo-inverse."""
        projection_matrix = model.components_.T
        # Moore-Penrose pseudo-inverse for best least-squares reconstruction
        inverse_projection = np.linalg.pinv(projection_matrix)
        # Reconstruct: project back to original space
        X_reconstructed = X_transformed @ inverse_projection
        mse = float(np.mean((X_original - X_reconstructed) ** 2))
        
        print(f"    Reconstruction MSE: {mse:.4f}")
        
        return mse
    
    def compute_selection_score(self, model, n_comp, recon_error):
        """RP selection: Minimize reconstruction error (MSE via pseudo-inverse)."""
        if recon_error is not None:
            score = -recon_error  # Negative so max(score) gives min(error)
            metric = 'reconstruction_error'
        else:
            score = -n_comp  # Fallback: penalize higher dimensions
            metric = 'dimension_penalty'
        return score, metric
    
    @staticmethod
    def johnson_lindenstrauss_min_dim(n_samples, eps=0.1):
        """Compute minimum dimension according to Johnson-Lindenstrauss lemma."""
        return int(4 * np.log(n_samples) / (eps**2 / 2 - eps**3 / 3))
    
    def plot_selection_metric(self, results):
        """Plot RP reconstruction error curve"""
        n_components = [r['n_components'] for r in results]
        errors = [r['reconstruction_error'] for r in results if r['reconstruction_error'] is not None]
        
        if not errors or all(e is None for e in errors):
            print(f"No reconstruction error available for {self.algorithm_name}")
            return
        
        plot_curve(
            x=n_components[:len(errors)],
            y_list=errors,
            labels=None,  # Single series - no legend needed
            xlabel='Number of Components',
            ylabel='MSE',
            title=f'RP: Reconstruction Error on {self.dataset}',
            save_path=os.path.join(self.save_path, "reconstruction_error_curve.png"),
            colors=['orange'],
            marker='o'
        )
    
    def plot_component_interpretation(self, model):
        """Plot RP projection matrix heatmap (RECOMMENDED for Step 2)."""
        if hasattr(model, 'components_'):
            n_comp = model.components_.shape[0]  # Use all components
            proj_matrix = model.components_
            labels = [f'RP{i+1}' for i in range(n_comp)]
            plot_heatmap(
                data=proj_matrix,
                title=f'Random Projection Matrix ({self.dataset})',
                save_path=os.path.join(self.save_path, "component_heatmap.png"),
                xlabel='Feature Index',
                ylabel='Component',
                row_labels=labels,
                colorbar_label='Projection Weight',
                cmap='RdBu_r'
            )
        else:
            print(f"RP model does not have components_ attribute, skipping heatmap")
