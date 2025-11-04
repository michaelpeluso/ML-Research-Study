import numpy as np
from sklearn.decomposition import FastICA
from dimensionality_reduction.base import BaseDR
from utils.logger import print_t as print


class ICAReduction(BaseDR):
    """Independent Component Analysis (ICA) for dimensionality reduction."""
    
    algorithm_name = 'ICA'
    
    def fit_transform(self, X, n_components, whiten='unit-variance', max_iter=200, tol=1e-4, **kwargs):
        """Fit ICA and transform X."""
        print(f"Fitting ICA with n_components={n_components}, whiten={whiten}")
        
        ica = FastICA(
            n_components=n_components,
            whiten=whiten,
            max_iter=max_iter,
            tol=tol,
            random_state=self.seed
        )
        X_transformed = ica.fit_transform(X)
        
        # Compute kurtosis as a measure of non-Gaussianity (ICA objective)
        kurtosis = np.mean([np.abs(self._kurtosis(X_transformed[:, i])) for i in range(n_components)])
        print(f"  Mean absolute kurtosis: {kurtosis:.4f}")
        
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
    
    def reconstruction_error(self, X_original, X_transformed, model):
        """Compute reconstruction error for ICA."""
        # ICA mixing matrix
        X_reconstructed = np.dot(X_transformed, model.mixing_.T) + model.mean_
        mse = float(np.mean((X_original - X_reconstructed) ** 2))
        return mse
    
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
