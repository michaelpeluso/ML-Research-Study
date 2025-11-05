import numpy as np
from sklearn.decomposition import PCA
from dimensionality_reduction.base import BaseDR
from utils.logger import print_t as print


class PCAReduction(BaseDR):
    """Principal Component Analysis (PCA) for dimensionality reduction."""
    
    algorithm_name = 'PCA'
    
    def fit_transform(self, X, n_components, whiten=False, **kwargs):
        """Fit PCA and transform X."""
        print(f"Fitting PCA with n_components={n_components}, whiten={whiten}")
        
        pca = PCA(n_components=n_components, whiten=whiten, random_state=self.seed)
        X_transformed = pca.fit_transform(X)
        
        print(f"  Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
        
        return pca, X_transformed
    
    def get_explained_variance(self, model):
        """Get explained variance ratio from PCA model."""
        return model.explained_variance_ratio_
    
    def reconstruction_error(self, X_original, X_transformed, model):
        """Compute reconstruction error (MSE) for PCA."""
        X_reconstructed = model.inverse_transform(X_transformed)
        mse = float(np.mean((X_original - X_reconstructed) ** 2))
        return mse
    
    def get_component_loadings(self, model, feature_names=None):
        """Get component loadings (eigenvectors) for interpretation."""
        components = model.components_
        n_components = components.shape[0]
        
        loadings = {}
        for i in range(n_components):
            # Get absolute loadings and sort
            abs_loadings = np.abs(components[i])
            top_indices = np.argsort(abs_loadings)[-10:][::-1]  # Top 10 features
            
            if feature_names is not None:
                top_features = [(feature_names[idx], components[i][idx]) for idx in top_indices]
            else:
                top_features = [(f"Feature_{idx}", components[i][idx]) for idx in top_indices]
            
            loadings[f'PC{i+1}'] = top_features
        
        return loadings
    
    def compute_variance_thresholds(self, results):
        """Compute n_components needed for 80%, 90%, 95%, 99% variance."""
        thresholds = {}
        
        for threshold in [0.80, 0.90, 0.95, 0.99]:
            for r in results:
                if r['explained_variance'] is not None:
                    var = r['explained_variance']
                    if isinstance(var, (list, np.ndarray)):
                        cumsum = np.cumsum(var)
                        if cumsum[-1] >= threshold:
                            n_needed = np.argmax(cumsum >= threshold) + 1
                            thresholds[f'{int(threshold*100)}%'] = int(n_needed)
                            break
        
        return thresholds
