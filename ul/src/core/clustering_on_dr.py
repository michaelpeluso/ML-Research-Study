import os
import time
import csv

from utils.cache_manager import CacheManager
from utils.logger import print_t as print

# Compare clustering results
def analyze_results(dataset, save_path, ml_logger, clustering_original, clustering_dr_results):
    """Compare clustering results between original and DR-reduced spaces."""
    
    # check cache
    cache_manager = CacheManager()  # Use centralized cache manager
    params = {
        'dataset': dataset,
        'has_original': clustering_original is not None,
        'original_algos': list(clustering_original.keys()) if clustering_original else [],
        'dr_methods': list(clustering_dr_results.keys()),
        'dr_algos': list(clustering_dr_results[list(clustering_dr_results.keys())[0]].keys()) if clustering_dr_results else []
    }
    cached_result = cache_manager.load(dataset, 'clustering_comparison', params)
    if cached_result is not None: return cached_result

    # Main
    with ml_logger.log_step(f"Clustering Comparison Analysis on {dataset}") as step_info:
        print("\n" + "="*80)
        print(f"Clustering Comparison Analysis on {dataset}")
        print("="*80 + "\n")

        comparison_table = []
        
        # Original space results
        if clustering_original is not None:
            for algo in ['kmeans', 'em']:
                algo_name = 'K-Means' if algo == 'kmeans' else 'EM/GMM'
                result = clustering_original[algo]['best_result']
                comparison_table.append({
                    'Space': 'Original',
                    'Algorithm': algo_name,
                    'n_clusters': clustering_original[algo]['chosen_n'],
                    'Silhouette': result.get('silhouette_score', 0),
                    'Calinski_Harabasz': result.get('calinski_harabasz_score', 0),
                    'Davies_Bouldin': result.get('davies_bouldin_score', 0),
                    'Dunn_Index': result.get('dunn_index', 0),
                    'BIC': result.get('bic', None),
                    'AIC': result.get('aic', None),
                })
        
        # DR space results
        for dr_method in ['pca', 'ica', 'rp']:
            for algo in ['kmeans', 'em']:
                algo_name = 'K-Means' if algo == 'kmeans' else 'EM/GMM'
                result = clustering_dr_results[dr_method][algo]['best_result']
                comparison_table.append({
                    'Space': dr_method.upper(),
                    'Algorithm': algo_name,
                    'n_clusters': clustering_dr_results[dr_method][algo]['chosen_n'],
                    'Silhouette': result.get('silhouette_score', 0),
                    'Calinski_Harabasz': result.get('calinski_harabasz_score', 0),
                    'Davies_Bouldin': result.get('davies_bouldin_score', 0),
                    'Dunn_Index': result.get('dunn_index', 0),
                    'BIC': result.get('bic', None),
                    'AIC': result.get('aic', None),
                })
        
        # Print table for console visibility
        print("\nClustering Performance Comparison:")
        print("-" * 150)
        print(f"{'Space':<10} {'Algorithm':<10} {'k':<5} {'Silhouette':<12} {'Calinski-H':<15} {'Davies-B':<12} {'Dunn':<10} {'BIC':<15} {'AIC':<15}")
        print("-" * 150)
        for row in comparison_table:
            bic_str = f"{row['BIC']:<15.2f}" if row['BIC'] is not None else f"{'N/A':<15}"
            aic_str = f"{row['AIC']:<15.2f}" if row['AIC'] is not None else f"{'N/A':<15}"
            print(f"{row['Space']:<10} {row['Algorithm']:<10} {row['n_clusters']:<5} "
                  f"{row['Silhouette']:<12.4f} {row['Calinski_Harabasz']:<15.2f} "
                  f"{row['Davies_Bouldin']:<12.4f} {row['Dunn_Index']:<10.4f} "
                  f"{bic_str} {aic_str}")
        print("-" * 150)
        
        # Find best configurations
        best_sil = max(comparison_table, key=lambda x: x['Silhouette'])
        best_ch = max(comparison_table, key=lambda x: x['Calinski_Harabasz'])
        best_db = min(comparison_table, key=lambda x: x['Davies_Bouldin'])
        best_dunn = max(comparison_table, key=lambda x: x['Dunn_Index'])
        
        # BIC and AIC (only for EM/GMM, lower is better)
        em_rows = [r for r in comparison_table if r['Algorithm'] == 'EM/GMM' and r['BIC'] is not None]
        best_bic = min(em_rows, key=lambda x: x['BIC']) if em_rows else None
        best_aic = min(em_rows, key=lambda x: x['AIC']) if em_rows else None
        
        # Summary insights
        print("\n" + "="*80)
        print("KEY INSIGHTS:")
        print("="*80)
        print(f"\nBest Silhouette Score: {best_sil['Silhouette']:.4f}")
        print(f"  Space: {best_sil['Space']}, Algorithm: {best_sil['Algorithm']}, k={best_sil['n_clusters']}")
        print(f"\nBest Calinski-Harabasz: {best_ch['Calinski_Harabasz']:.2f}")
        print(f"  Space: {best_ch['Space']}, Algorithm: {best_ch['Algorithm']}, k={best_ch['n_clusters']}")
        
        if best_bic:
            print(f"\nBest BIC (EM/GMM only): {best_bic['BIC']:.2f}")
            print(f"  Space: {best_bic['Space']}, Algorithm: {best_bic['Algorithm']}, k={best_bic['n_clusters']}")
        
        if best_aic:
            print(f"\nBest AIC (EM/GMM only): {best_aic['AIC']:.2f}")
            print(f"  Space: {best_aic['Space']}, Algorithm: {best_aic['Algorithm']}, k={best_aic['n_clusters']}")
        
        # Compare k selection across spaces
        print("\nOptimal k Selection:")
        for algo in ['K-Means', 'EM/GMM']:
            algo_rows = [r for r in comparison_table if r['Algorithm'] == algo]
            print(f"  {algo}:")
            for row in algo_rows:
                print(f"    {row['Space']}: k={row['n_clusters']}")
        
        print("\n" + "="*80 + "\n")
        
        # Save comparison table to CSV
        csv_path = os.path.join(save_path, "clustering_on_dr_comparison.csv")
        with open(csv_path, 'w', newline='') as f:
            if comparison_table:
                writer = csv.DictWriter(f, fieldnames=comparison_table[0].keys())
                writer.writeheader()
                writer.writerows(comparison_table)
        print(f"Comparison table saved to: {csv_path}\n")
        
        results = {
            'comparison_table': comparison_table,
            'best_configurations': {
                'silhouette': {
                    'space': best_sil['Space'],
                    'algorithm': best_sil['Algorithm'],
                    'k': best_sil['n_clusters'],
                    'score': best_sil['Silhouette']
                },
                'calinski_harabasz': {
                    'space': best_ch['Space'],
                    'algorithm': best_ch['Algorithm'],
                    'k': best_ch['n_clusters'],
                    'score': best_ch['Calinski_Harabasz']
                },
                'davies_bouldin': {
                    'space': best_db['Space'],
                    'algorithm': best_db['Algorithm'],
                    'k': best_db['n_clusters'],
                    'score': best_db['Davies_Bouldin']
                },
                'dunn_index': {
                    'space': best_dunn['Space'],
                    'algorithm': best_dunn['Algorithm'],
                    'k': best_dunn['n_clusters'],
                    'score': best_dunn['Dunn_Index']
                },
                'bic': {
                    'space': best_bic['Space'] if best_bic else None,
                    'algorithm': best_bic['Algorithm'] if best_bic else None,
                    'k': best_bic['n_clusters'] if best_bic else None,
                    'score': best_bic['BIC'] if best_bic else None
                } if best_bic else None,
                'aic': {
                    'space': best_aic['Space'] if best_aic else None,
                    'algorithm': best_aic['Algorithm'] if best_aic else None,
                    'k': best_aic['n_clusters'] if best_aic else None,
                    'score': best_aic['AIC'] if best_aic else None
                } if best_aic else None
            },
            'summary': {
                'total_configurations': len(comparison_table),
                'spaces_compared': len(set(r['Space'] for r in comparison_table)),
                'algorithms_compared': len(set(r['Algorithm'] for r in comparison_table))
            }
        }
        step_info.update(results)

        # Save to cache
        cache_manager.save(dataset, 'clustering_comparison', results, params)

        return results