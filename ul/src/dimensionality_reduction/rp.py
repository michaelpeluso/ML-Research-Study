import numpy as np
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from dimensionality_reduction.base import BaseDR
from utils.logger import print_t as print


class RandomProjection(BaseDR):
    """Random Projection for dimensionality reduction using Johnson-Lindenstrauss lemma."""
    
    algorithm_name = 'Random Projection'
    
    def fit_transform(self, X, n_components, projection_type='gaussian', eps=0.1, **kwargs):
        """Fit Random Projection and transform X."""
        print(f"Fitting {projection_type.title()} Random Projection with n_components={n_components}")
        
        if projection_type == 'gaussian':
            rp = GaussianRandomProjection(
                n_components=n_components,
                eps=eps,
                random_state=self.seed
            )
        else:  # sparse
            rp = SparseRandomProjection(
                n_components=n_components,
                eps=eps,
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
        """Compute pairwise distance preservation error. """
        n_samples = min(1000, X_original.shape[0])  # Sample for efficiency
        indices = np.random.RandomState(self.seed).choice(X_original.shape[0], n_samples, replace=False)
        
        X_sample_orig = X_original[indices]
        X_sample_proj = X_transformed[indices]
        
        # Compute pairwise distances
        from scipy.spatial.distance import pdist
        dist_orig = pdist(X_sample_orig, metric='euclidean')
        dist_proj = pdist(X_sample_proj, metric='euclidean')
        
        # Normalize distances
        dist_orig_norm = dist_orig / (np.mean(dist_orig) + 1e-10)
        dist_proj_norm = dist_proj / (np.mean(dist_proj) + 1e-10)
        
        # Mean squared error of normalized distances
        mse = float(np.mean((dist_orig_norm - dist_proj_norm) ** 2))
        
        return mse
    
    def get_reconstruction_quality(self, X_original, X_transformed):
        """Measure how well pairwise distances are preserved."""
        n_samples = min(1000, X_original.shape[0])
        indices = np.random.RandomState(self.seed).choice(X_original.shape[0], n_samples, replace=False)
        
        X_sample_orig = X_original[indices]
        X_sample_proj = X_transformed[indices]
        
        from scipy.spatial.distance import pdist
        from scipy.stats import pearsonr
        
        dist_orig = pdist(X_sample_orig, metric='euclidean')
        dist_proj = pdist(X_sample_proj, metric='euclidean')
        
        # Correlation between distance matrices
        correlation, _ = pearsonr(dist_orig, dist_proj)
        
        # Relative error
        relative_error = float(np.mean(np.abs(dist_orig - dist_proj) / (dist_orig + 1e-10)))
        
        return {
            'distance_correlation': float(correlation), #type:ignore
            'relative_distance_error': relative_error
        }
    
    @staticmethod
    def johnson_lindenstrauss_min_dim(n_samples, eps=0.1):
        """Compute minimum dimension according to Johnson-Lindenstrauss lemma."""
        return int(4 * np.log(n_samples) / (eps**2 / 2 - eps**3 / 3))
