import os
import joblib
import hashlib
import json
from typing import Any, Optional, Callable, Dict

from utils.logger import print_t as print


class CacheManager:
    """Manages caching of experiment results to avoid re-computation."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            cache_dir = os.path.join(os.environ.get('ROOT', '.'), 'cache')
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _generate_cache_key(self, dataset: str, step: str, params: dict) -> str:
        """Generate a unique cache key based on dataset, step, and parameters."""
        # Sort params to ensure consistent ordering
        params_str = json.dumps(params, sort_keys=True)
        hash_obj = hashlib.md5(params_str.encode())
        hash_str = hash_obj.hexdigest()[:8]
        return f"{dataset}_{step}_{hash_str}.pkl"
    
    def _get_step_folder(self, step: str) -> str:
        """Map step names to organized folder structure.
        
        Step Folders:
        - step1_clustering/     # Original data clustering
            - k-means/
            - gmm/
        - step2_dr/            # Dimensionality reduction
            - pca/
            - ica/
            - random_projection/
        - step3_clustering_dr/  # Clustering on reduced data
            - pca/
                - k-means/
                - gmm/
            - ica/
                - k-means/
                - gmm/
            - random_projection/
                - k-means/
                - gmm/
        - step4_nn_dr/         # Neural networks on reduced data
        - step5_nn_clusters/   # Neural networks with cluster features
        """
        step_lower = step.lower()
        
        # Step 1: Clustering on original data
        if 'clustering' in step_lower and '_dr' not in step_lower and 'comparison' not in step_lower:
            if 'k-means' in step_lower or 'kmeans' in step_lower:
                return 'step1_clustering/k-means'
            elif 'gmm' in step_lower or 'em' in step_lower:
                return 'step1_clustering/gmm'
            else:
                return 'step1_clustering'
            
        # Step 2: Dimensionality Reduction
        elif '_dr' in step_lower and 'clustering' not in step_lower:
            if 'pca' in step_lower:
                return 'step2_dr/pca'
            elif 'ica' in step_lower:
                return 'step2_dr/ica'
            elif 'random' in step_lower and 'projection' in step_lower:
                return 'step2_dr/random_projection'
            else:
                return 'step2_dr'
            
        # Step 3: Clustering on DR
        elif 'comparison' in step_lower or ('clustering' in step_lower and '_dr' in step_lower):
            # Extract DR method from step name
            dr_method = None
            if 'pca' in step_lower:
                dr_method = 'pca'
            elif 'ica' in step_lower:
                dr_method = 'ica'
            elif 'random' in step_lower and 'projection' in step_lower:
                dr_method = 'random_projection'
            
            if dr_method:
                # Extract clustering method
                if 'k-means' in step_lower or 'kmeans' in step_lower:
                    return f'step3_clustering_dr/{dr_method}/k-means'
                elif 'gmm' in step_lower or 'em' in step_lower:
                    return f'step3_clustering_dr/{dr_method}/gmm'
                else:
                    return f'step3_clustering_dr/{dr_method}'
            return 'step3_clustering_dr'
            
        # Step 4: Neural Networks on DR
        elif 'neural_networks_on_dr' in step_lower:
            return 'step4_nn_dr'
            
        # Step 5: Neural Networks with Clusters
        elif 'neural_networks_with_clusters' in step_lower:
            return 'step5_nn_clusters'
            
        # Default fallback
        return 'other'
    
    def save(self, dataset: str, step: str, data: Any, params: dict) -> str:
        """Save data to cache with metadata in organized folder structure."""
        cache_key = self._generate_cache_key(dataset, step, params)
        step_folder = self._get_step_folder(step)
        step_dir = os.path.join(self.cache_dir, step_folder)
        os.makedirs(step_dir, exist_ok=True)
        cache_path = os.path.join(step_dir, cache_key)
        
        cache_data = {
            'data': data,
            'params': params,
            'dataset': dataset,
            'step': step
        }
        
        joblib.dump(cache_data, cache_path)
        print(f"Cached {step} results to {cache_path}")
        return cache_path
    
    def load(self, dataset: str, step: str, params: dict) -> Optional[Any]:
        """Load data from cache if it exists and matches parameters."""
        cache_key = self._generate_cache_key(dataset, step, params)
        step_folder = self._get_step_folder(step)
        step_dir = os.path.join(self.cache_dir, step_folder)
        cache_path = os.path.join(step_dir, cache_key)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            cache_data = joblib.load(cache_path)
            
            # Verify parameters match
            if cache_data['params'] != params:
                print(f"Cache found but params differ. Recomputing...")
                return None
            
            print(f"Loaded {step} from cache: {cache_path}")
            return cache_data['data']
        except Exception as e:
            print(f"Failed to load cache: {e}")
            return None
    
    def run_cached(self, dataset: str, step: str, func: Callable[[], Any], params: Dict[str, Any], use_cache: bool = True) -> Any:
        """Run a function with caching. Checks cache first, runs function if needed, saves result."""
        if use_cache:
            cached_result = self.load(dataset, step, params)
            if cached_result is not None:
                return cached_result
        
        # Run the function
        result = func()
        
        # Save to cache (always save, regardless of use_cache flag)
        self.save(dataset, step, result, params)
        
        return result
    
    def run_with_args(self, dataset: str, step: str, func: Callable, args_dict: Dict[str, Any], use_cache: bool = True) -> Any:
        """Run a function with arguments dict and automatic parameter extraction for caching."""
        # Extract parameters for caching based on step
        if step == 'clustering':
            params = {k: args_dict[k] for k in ['seed', 'n_components_range', 'stability_runs', 'n_init', 'silhouette_subsample']}
            params['data_shape'] = args_dict['X_train'].shape
        elif step == 'dimensionality_reduction':
            params = {k: args_dict[k] for k in ['seed', 'n_components_range', 'method']}
            params['data_shape'] = args_dict['X_train'].shape
        elif step == 'clustering_on_dr':
            params = {k: args_dict[k] for k in ['seed', 'n_components_range', 'stability_runs', 'n_init', 'silhouette_subsample']}
            params['dr_n_components'] = {k: v['n_components'] for k, v in args_dict['dr_results'].items()}
        else:
            params = {}  # Fallback
        
        return self.run_cached(dataset, step, lambda: func(**args_dict), params, use_cache)
    
    def clear(self, dataset: Optional[str] = None, step: Optional[str] = None):
        """Clear cache files. If dataset/step specified, only clear those."""
        import glob
        
        if step:
            # Clear specific step folder
            step_folder = self._get_step_folder(step)
            step_dir = os.path.join(self.cache_dir, step_folder)
            if dataset:
                pattern = os.path.join(step_dir, f"{dataset}_{step}_*.pkl")
            else:
                pattern = os.path.join(step_dir, "*.pkl")
        elif dataset:
            pattern = os.path.join(self.cache_dir, "**", f"{dataset}_*.pkl")
        else:
            pattern = os.path.join(self.cache_dir, "**", "*.pkl")
        
        files = glob.glob(pattern, recursive=True)
        for f in files:
            try:
                os.remove(f)
                print(f"Removed cache: {f}")
            except Exception as e:
                print(f"Failed to remove {f}: {e}")
        
        if files:
            print(f"Cleared {len(files)} cache file(s)")
        else:
            print("No cache files to clear")
