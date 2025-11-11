from typing import Dict, Any, List, Tuple
import numpy as np

from utils.logger import print_t as print


class ExperimentDataValidator:
    """Validates that all expected data is present in experiment results."""
    
    @staticmethod
    def validate_clustering_results(results: Dict[str, Any], dataset: str) -> Tuple[bool, List[str]]:
        """
        Validate Step 1 clustering results.
        
        Args:
            results: clustering_results dict from run_experiments()
            dataset: 'hotels' or 'accidents'
        
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        # Check for both algorithms
        for algo in ['kmeans', 'em']:
            if algo not in results:
                issues.append(f"Missing '{algo}' in clustering results")
                continue
            
            algo_results = results[algo]
            
            # Check required keys
            required_keys = ['chosen_n', 'labels', 'centers', 'scores', 'stability']
            for key in required_keys:
                if key not in algo_results:
                    issues.append(f"Missing '{key}' in {algo} results")
            
            # Check score metrics
            if 'scores' in algo_results:
                required_scores = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
                for metric in required_scores:
                    if metric not in algo_results['scores']:
                        issues.append(f"Missing '{metric}' score in {algo}")
                    elif not isinstance(algo_results['scores'][metric], list):
                        issues.append(f"{algo} {metric} should be a list, got {type(algo_results['scores'][metric])}")
            
            # Validate data shapes
            if 'labels' in algo_results and 'centers' in algo_results:
                labels = np.asarray(algo_results['labels'])
                centers = np.asarray(algo_results['centers'])
                if centers.shape[0] != algo_results['chosen_n']:
                    issues.append(f"{algo} centers shape {centers.shape[0]} != chosen_n {algo_results['chosen_n']}")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def validate_dr_results(results: Dict[str, Any], dataset: str) -> Tuple[bool, List[str]]:
        """
        Validate Step 2 dimensionality reduction results.
        
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        # Check for all three DR methods
        for method in ['pca', 'ica', 'rp']:
            if method not in results:
                issues.append(f"Missing '{method}' in DR results")
                continue
            
            method_results = results[method]
            
            # Check required keys
            required_keys = ['X_transformed', 'n_components', 'model', 'scores']
            for key in required_keys:
                if key not in method_results:
                    issues.append(f"Missing '{key}' in {method} DR results")
            
            # Validate data
            if 'X_transformed' in method_results and 'n_components' in method_results:
                X_t = np.asarray(method_results['X_transformed'])
                n_comp = method_results['n_components']
                if X_t.shape[1] != n_comp:
                    issues.append(f"{method} X_transformed shape {X_t.shape[1]} != n_components {n_comp}")
            
            # Check scores
            if 'scores' in method_results:
                required_scores = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
                for metric in required_scores:
                    if metric not in method_results['scores']:
                        issues.append(f"Missing '{metric}' score in {method}")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def validate_nn_results(results: Dict[str, Any], step: int = 4) -> Tuple[bool, List[str]]:
        """
        Validate Step 4 or 5 neural network results.
        
        Args:
            results: nn_results or nn_cluster_results dict
            step: 4 or 5
        
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        if results is None:
            issues.append(f"Step {step} results are None")
            return False, issues
        
        # Check for total_time
        if 'total_time' not in results:
            issues.append(f"Missing 'total_time' in Step {step} results")
        
        # Determine expected configs
        if step == 4:
            expected_configs = ['original', 'pca', 'ica', 'rp']
        else:  # step 5
            expected_configs = ['baseline', 'kmeans', 'em']
        
        # Check each configuration
        for config in expected_configs:
            if config not in results:
                issues.append(f"Missing '{config}' configuration in Step {step}")
                continue
            
            config_result = results[config]
            
            # Check required keys
            required_keys = ['test_loss', 'final_train_loss', 'wall_time', 'curves', 'n_params', 'n_trainable']
            for key in required_keys:
                if key not in config_result:
                    issues.append(f"Missing '{key}' in Step {step} {config}")
            
            # Validate data types
            if 'test_loss' in config_result:
                if not isinstance(config_result['test_loss'], (int, float)):
                    issues.append(f"Step {step} {config} test_loss should be numeric, got {type(config_result['test_loss'])}")
            
            if 'curves' in config_result:
                curves = config_result['curves']
                if not isinstance(curves, (list, np.ndarray)):
                    issues.append(f"Step {step} {config} curves should be list/array, got {type(curves)}")
                elif len(curves) == 0:
                    issues.append(f"Step {step} {config} curves is empty")
            
            # Check for n_components in Step 4 DR methods
            if step == 4 and config in ['pca', 'ica', 'rp']:
                if 'n_components' not in config_result:
                    issues.append(f"Missing 'n_components' in Step {step} {config}")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def validate_full_pipeline(result: Dict[str, Any], dataset: str) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Validate complete experiment pipeline results.
        
        Args:
            result: Dict returned from run_experiments()
            dataset: 'hotels' or 'accidents'
        
        Returns:
            (all_valid, dict_of_step_issues)
        """
        all_issues = {}
        
        # Check dataset
        if result.get('dataset') != dataset:
            all_issues['metadata'] = [f"Dataset mismatch: expected {dataset}, got {result.get('dataset')}"]
        
        # Step 1
        clustering_valid, clustering_issues = ExperimentDataValidator.validate_clustering_results(
            result.get('clustering_results', {}), dataset
        )
        if clustering_issues:
            all_issues['step1_clustering'] = clustering_issues
        
        # Step 2
        dr_valid, dr_issues = ExperimentDataValidator.validate_dr_results(
            result.get('dr_results', {}), dataset
        )
        if dr_issues:
            all_issues['step2_dr'] = dr_issues
        
        # Step 3 - check structure
        clustering_dr = result.get('clustering_dr_results', {})
        step3_issues = []
        if not clustering_dr:
            step3_issues.append("Missing clustering_dr_results")
        else:
            for dr_method in ['pca', 'ica', 'rp']:
                if dr_method not in clustering_dr:
                    step3_issues.append(f"Missing '{dr_method}' clustering in Step 3")
                else:
                    for algo in ['kmeans', 'em']:
                        if algo not in clustering_dr[dr_method]:
                            step3_issues.append(f"Missing '{algo}' clustering on {dr_method} in Step 3")
        if step3_issues:
            all_issues['step3_clustering_on_dr'] = step3_issues
        
        # Step 4 (Accidents only)
        if dataset == 'accidents':
            # Check both Step 4a and 4b results
            step4_issues = []
            
            # Step 4a: Original data
            nn_original = result.get('step_4a_nn_original')
            if nn_original is not None:
                orig_valid, orig_issues = ExperimentDataValidator.validate_nn_results(
                    {'original': nn_original['original']} if 'original' in nn_original else {}, step=4
                )
                if orig_issues:
                    step4_issues.extend([f"Step 4a: {issue}" for issue in orig_issues])
            
            # Step 4b: Reduced data
            nn_reduced = result.get('step_4b_nn_reduced')
            if nn_reduced is not None:
                reduced_valid, reduced_issues = ExperimentDataValidator.validate_nn_results(
                    nn_reduced, step=4
                )
                if reduced_issues:
                    step4_issues.extend([f"Step 4b: {issue}" for issue in reduced_issues])
            
            # If no Step 4 results at all
            if nn_original is None and nn_reduced is None:
                step4_issues.append("Step 4 results are None")
            
            if step4_issues:
                all_issues['step4_nn'] = step4_issues
        
        # Step 5 (Accidents only)
        if dataset == 'accidents':
            nn_cluster = result.get('step_5_nn_with_clusters')
            if nn_cluster is not None:
                nn_cluster_valid, nn_cluster_issues = ExperimentDataValidator.validate_nn_results(
                    nn_cluster, step=5 #type:ignore
                )
                if nn_cluster_issues:
                    all_issues['step5_nn_clusters'] = nn_cluster_issues
            else:
                all_issues['step5_nn_clusters'] = ["Step 5 results are None"]
        
        all_valid = len(all_issues) == 0
        return all_valid, all_issues
    
    @staticmethod
    def print_validation_report(is_valid: bool, issues: Dict[str, List[str]]):
        """Pretty print validation results."""
        if is_valid:
            print("\n" + "="*80)
            print("VALIDATION PASSED - All expected data is present and valid")
            print("="*80 + "\n")
        else:
            print("\n" + "="*80)
            print("✗ VALIDATION FAILED - Issues detected:")
            print("="*80)
            for step, step_issues in issues.items():
                print(f"\n{step.upper()}:")
                for issue in step_issues:
                    print(f"  - {issue}")
            print("\n" + "="*80 + "\n")


def validate_experiment_results(result: Dict[str, Any], dataset: str = None) -> bool: #type:ignore
    """Convenience function to validate and print results."""
    if dataset is None:
        dataset = result.get('dataset', 'unknown')
    
    is_valid, issues = ExperimentDataValidator.validate_full_pipeline(result, dataset)
    ExperimentDataValidator.print_validation_report(is_valid, issues)
    
    return is_valid
