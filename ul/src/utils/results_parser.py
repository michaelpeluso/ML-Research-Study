import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


def parse_master_results(all_results: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    """
    Parse master results from all experiments into organized DataFrames.
    
    Args:
        all_results: List of result dicts from run_experiments()
        
    Returns:
        Dict containing DataFrames for each analysis type:
        - 'clustering_metrics': Step 1 clustering results
        - 'dr_metrics': Step 2 dimensionality reduction results
        - 'clustering_on_dr_metrics': Step 3 clustering on DR results
        - 'nn_original_metrics': Step 4a neural network on original data
        - 'nn_reduced_metrics': Step 4b neural network on reduced data
        - 'nn_cluster_metrics': Step 5 neural network with cluster features
        - 'summary': High-level summary across all steps
    """
    parsed = {}
    
    # Parse each experiment type
    parsed['clustering_metrics'] = parse_clustering_results(all_results)
    parsed['dr_metrics'] = parse_dr_results(all_results)
    parsed['clustering_on_dr_metrics'] = parse_clustering_on_dr_results(all_results)
    parsed['nn_original_metrics'] = parse_nn_original_results(all_results)
    parsed['nn_reduced_metrics'] = parse_nn_reduced_results(all_results)
    parsed['nn_cluster_metrics'] = parse_nn_cluster_results(all_results)
    parsed['summary'] = generate_summary_table(all_results)
    
    return parsed


def parse_clustering_results(all_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Extract Step 1 clustering metrics into DataFrame."""
    rows = []
    
    for result in all_results:
        dataset = result['dataset']
        clustering = result.get('step_1_clustering')
        
        if clustering is None:
            continue
            
        for algo in ['kmeans', 'em']:
            algo_results = clustering.get(algo, {})
            if not algo_results:
                continue
                
            row = {
                'dataset': dataset,
                'algorithm': algo.upper(),
                'chosen_k': algo_results.get('chosen_n'),
                'silhouette_score': algo_results.get('silhouette_score'),
                'bic': algo_results.get('bic'),
                'aic': algo_results.get('aic'),
                'log_likelihood': algo_results.get('log_likelihood'),
                'stability_mean': algo_results.get('stability_mean'),
                'stability_std': algo_results.get('stability_std'),
            }
            rows.append(row)
    
    return pd.DataFrame(rows)


def parse_dr_results(all_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Extract Step 2 dimensionality reduction metrics into DataFrame."""
    rows = []
    
    for result in all_results:
        dataset = result['dataset']
        dr_results = result.get('step_2_dr')
        
        if dr_results is None:
            continue
            
        for dr_method in ['pca', 'ica', 'rp']:
            dr_data = dr_results.get(dr_method, {})
            if not dr_data:
                continue
                
            row = {
                'dataset': dataset,
                'method': dr_method.upper(),
                'n_components': dr_data.get('n_components'),
                'explained_variance': dr_data.get('explained_variance_ratio'),
                'reconstruction_error': dr_data.get('reconstruction_error'),
                'kurtosis_mean': dr_data.get('kurtosis_mean'),
                'kurtosis_std': dr_data.get('kurtosis_std'),
            }
            rows.append(row)
    
    return pd.DataFrame(rows)


def parse_clustering_on_dr_results(all_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Extract Step 3 clustering on DR metrics into DataFrame."""
    rows = []
    
    for result in all_results:
        dataset = result['dataset']
        clustering_dr = result.get('step_3_clustering_on_dr')
        
        if clustering_dr is None:
            continue
            
        for dr_method in ['pca', 'ica', 'rp']:
            dr_clustering = clustering_dr.get(dr_method, {})
            if not dr_clustering:
                continue
                
            for algo in ['kmeans', 'em']:
                algo_results = dr_clustering.get(algo, {})
                if not algo_results:
                    continue
                    
                row = {
                    'dataset': dataset,
                    'dr_method': dr_method.upper(),
                    'algorithm': algo.upper(),
                    'chosen_k': algo_results.get('chosen_n'),
                    'silhouette_score': algo_results.get('silhouette_score'),
                    'bic': algo_results.get('bic'),
                    'aic': algo_results.get('aic'),
                    'stability_mean': algo_results.get('stability_mean'),
                    'stability_std': algo_results.get('stability_std'),
                }
                rows.append(row)
    
    return pd.DataFrame(rows)


def parse_nn_original_results(all_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Extract Step 4a neural network on original data metrics into DataFrame."""
    rows = []
    
    for result in all_results:
        dataset = result['dataset']
        nn_original = result.get('step_4a_nn_original')
        
        if nn_original is None:
            continue
            
        # Original data results
        original_data = nn_original.get('original', {})
        if original_data:
            row = {
                'dataset': dataset,
                'data_type': 'original',
                'test_loss': original_data.get('test_loss'),
                'train_loss': original_data.get('final_train_loss'),
                'wall_time': original_data.get('wall_time'),
                'n_features': original_data.get('n_features'),
            }
            rows.append(row)
    
    return pd.DataFrame(rows)


def parse_nn_reduced_results(all_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Extract Step 4b neural network on reduced data metrics into DataFrame."""
    rows = []
    
    for result in all_results:
        dataset = result['dataset']
        nn_reduced = result.get('step_4b_nn_reduced')
        
        if nn_reduced is None:
            continue
            
        for dr_method in ['pca', 'ica', 'rp']:
            dr_data = nn_reduced.get(dr_method, {})
            if not dr_data:
                continue
                
            row = {
                'dataset': dataset,
                'dr_method': dr_method.upper(),
                'n_components': dr_data.get('n_components'),
                'test_loss': dr_data.get('test_loss'),
                'train_loss': dr_data.get('final_train_loss'),
                'wall_time': dr_data.get('wall_time'),
            }
            rows.append(row)
    
    return pd.DataFrame(rows)


def parse_nn_cluster_results(all_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Extract Step 5 neural network with cluster features metrics into DataFrame."""
    rows = []
    
    for result in all_results:
        dataset = result['dataset']
        nn_cluster = result.get('step_5_nn_with_clusters')
        
        if nn_cluster is None:
            continue
            
        for config_name, config_data in nn_cluster.items():
            if config_name == 'total_time' or not isinstance(config_data, dict):
                continue
                
            row = {
                'dataset': dataset,
                'configuration': config_name,
                'test_loss': config_data.get('test_loss'),
                'train_loss': config_data.get('final_train_loss'),
                'wall_time': config_data.get('wall_time'),
            }
            rows.append(row)
    
    return pd.DataFrame(rows)


def generate_summary_table(all_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Generate high-level summary across all steps for each dataset."""
    rows = []
    
    for result in all_results:
        dataset = result['dataset']
        metadata = result.get('metadata', {})
        
        # Get best clustering results
        clustering = result.get('step_1_clustering', {})
        kmeans_k = clustering.get('kmeans', {}).get('chosen_n', 'N/A')
        em_k = clustering.get('em', {}).get('chosen_n', 'N/A')
        
        # Get DR components
        dr_results = result.get('step_2_dr', {})
        pca_n = dr_results.get('pca', {}).get('n_components', 'N/A')
        ica_n = dr_results.get('ica', {}).get('n_components', 'N/A')
        rp_n = dr_results.get('rp', {}).get('n_components', 'N/A')
        
        # Get NN performance
        nn_original = result.get('step_4a_nn_original')
        nn_reduced = result.get('step_4b_nn_reduced')
        
        original_loss = nn_original.get('original', {}).get('test_loss', 'N/A') if nn_original and isinstance(nn_original, dict) else 'N/A'
        pca_loss = nn_reduced.get('pca', {}).get('test_loss', 'N/A') if nn_reduced and isinstance(nn_reduced, dict) else 'N/A'
        ica_loss = nn_reduced.get('ica', {}).get('test_loss', 'N/A') if nn_reduced and isinstance(nn_reduced, dict) else 'N/A'
        rp_loss = nn_reduced.get('rp', {}).get('test_loss', 'N/A') if nn_reduced and isinstance(nn_reduced, dict) else 'N/A'
        
        row = {
            'dataset': dataset,
            'method': metadata.get('method', 'N/A'),
            'wall_time': metadata.get('wall_time', 'N/A'),
            'kmeans_k': kmeans_k,
            'em_k': em_k,
            'pca_components': pca_n,
            'ica_components': ica_n,
            'rp_components': rp_n,
            'nn_original_loss': original_loss,
            'nn_pca_loss': pca_loss,
            'nn_ica_loss': ica_loss,
            'nn_rp_loss': rp_loss,
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def save_parsed_results(parsed_results: Dict[str, pd.DataFrame], save_dir: str) -> None:
    """Save all parsed DataFrames to CSV files."""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    for name, df in parsed_results.items():
        if df is not None and not df.empty:
            filepath = os.path.join(save_dir, f"{name}.csv")
            df.to_csv(filepath, index=False)
            print(f"Saved {name} to {filepath}")


def print_summary(parsed_results: Dict[str, pd.DataFrame]) -> None:
    """Print summary of parsed results."""
    print("\n" + "="*80)
    print("MASTER RESULTS SUMMARY")
    print("="*80)
    
    for name, df in parsed_results.items():
        if df is not None and not df.empty:
            print(f"\n{name.upper()}:")
            print(f"  Rows: {len(df)}")
            print(f"  Columns: {list(df.columns)}")
        else:
            print(f"\n{name.upper()}: No data")
    
    print("\n" + "="*80)


# Example usage function
def extract_and_save_results(all_results: List[Dict[str, Any]], save_dir: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Main entry point: parse all results and optionally save to CSV.
    
    Usage in main.py:
        from utils.results_parser import extract_and_save_results
        
        # After all experiments complete
        parsed_results = extract_and_save_results(all_results, save_dir='figures/master_results')
        
        # Access specific tables
        clustering_df = parsed_results['clustering_metrics']
        nn_comparison_df = parsed_results['summary']
    """
    parsed_results = parse_master_results(all_results)
    print_summary(parsed_results)
    
    if save_dir:
        save_parsed_results(parsed_results, save_dir)
    
    return parsed_results
