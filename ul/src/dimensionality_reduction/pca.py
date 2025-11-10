import numpy as np
import os
from sklearn.decomposition import PCA

from dimensionality_reduction.base import BaseDR
from utils.plotter import plot_curve, plot_heatmap, plot_bar, stitch_specific_images
from utils.logger import print_t as print


class PCAReduction(BaseDR):
    """Principal Component Analysis (PCA) for dimensionality reduction."""
    
    algorithm_name = 'PCA'
    
    def fit_transform(self, X, n_components):
        """Fit PCA and transform X with whitening disabled for consistency."""
        whiten = False  # Disable whitening to match other methods
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
    
    def compute_selection_score(self, model, n_comp, recon_error):
        """PCA selection: Maximize cumulative explained variance (aim for ≥95%)."""
        explained_var = self.get_explained_variance(model)
        if explained_var is not None and isinstance(explained_var, (list, np.ndarray)):
            score = float(np.sum(explained_var[:n_comp]))
            metric = 'cumulative_variance'
        else:
            score = -recon_error if recon_error else 0
            metric = 'reconstruction_error'
        return score, metric
    
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
    
    def plot_selection_metric(self, results):
        """Plot PCA variance explained curve (REQUIRED for Step 2)."""
        n_components = [r['n_components'] for r in results]
        cumulative = [r.get('cumulative_variance', 0) for r in results]
        
        plot_curve(
            x=n_components,
            y_list=cumulative,
            labels=None,  # Single series - no legend needed
            xlabel='Number of Components',
            ylabel='Variance Explained',
            title=f'PCA: Variance Explained on {self.dataset}',
            save_path=os.path.join(self.save_path, "variance_explained.png"),
            colors=['blue'],
            marker='o'
        )
    
    def plot_component_interpretation(self, model):
        """Plot PCA component loadings heatmap and only PC1 top feature loadings"""

        components = model.components_
        labels = [f'PC{i+1}' for i in range(components.shape[0])]

        # Create subdirectory for individual component plots
        os.makedirs(self.save_path, exist_ok=True)

        # 1. Heatmap of all component loadings
        heatmap_path = os.path.join(self.save_path, "component_heatmap.png")
        plot_heatmap(
            data=components,
            title=f'PCA Component Loadings ({self.dataset})',
            save_path=heatmap_path,
            xlabel='Feature Index',
            ylabel='Component',
            row_labels=labels,
            colorbar_label='Loading Magnitude',
            cmap='RdBu_r'
        )

        # 2. Bar plot of top features for PC1 only
        pc1_loadings = components[0]
        abs_loadings = np.abs(pc1_loadings)
        top_feature_indices = np.argsort(abs_loadings)[::-1]
        top_loadings = pc1_loadings[top_feature_indices]
        feature_labels = [f'{idx}' for idx in top_feature_indices]

        pc1_bar_path = os.path.join(self.save_path, "loadings_pc1.png")
        explained_var = model.explained_variance_ratio_
        plot_bar(
            x_labels=feature_labels,
            values=top_loadings,
            xlabel='Feature',
            ylabel='Loading Coefficient',
            title=f'PC1 Top Feature Loadings\nExplains {explained_var[0]*100:.2f}% variance',
            save_path=pc1_bar_path,
            color='steelblue'
        )

        print(f"  Generated PC1 component loading bar plot in {self.save_path}")
    
    def _get_components_for_projection(self, model):
        """Get PCA components for projection plot"""
        return model.components_.T