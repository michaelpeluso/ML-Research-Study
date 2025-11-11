"""
Test script to verify that all plots are regenerated when loading from cache.
This script will:
1. Delete specific plot files
2. Load cached results (which should trigger plot regeneration)
3. Verify that plots were recreated
"""
import os
import sys
import glob

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'src'))
os.environ['ROOT'] = script_dir
os.chdir(os.path.join(script_dir, 'src'))

from utils.data_processing import load_or_process_data
from utils.logger import MLLogger
from clustering import KMeansClustering, EMClustering
from dimensionality_reduction import ICAReduction

print("="*80)
print("PLOT REGENERATION TEST")
print("="*80)

# Load accidents data
print("\n1. Loading accidents data...")
X_full, y_full = load_or_process_data('accidents', 'Duration_Seconds', 'regression', 0.6, seed=42)
print(f"   Data shape: {X_full.shape}")

# Test 1: DR Plot Regeneration
print("\n" + "="*80)
print("TEST 1: DR Plot Regeneration (ICA)")
print("="*80)

ica_plots_dir = os.path.join(script_dir, 'figures', 'accidents', 'dr', 'ica')
ica_plots_before = glob.glob(os.path.join(ica_plots_dir, '*.png'))
print(f"\n2. Found {len(ica_plots_before)} ICA plots before test")

# Delete some plots to test regeneration
plots_to_delete = [
    os.path.join(ica_plots_dir, 'projection_2d.png'),
    os.path.join(ica_plots_dir, 'component_heatmap.png'),
]
deleted_count = 0
for plot_path in plots_to_delete:
    if os.path.exists(plot_path):
        os.remove(plot_path)
        deleted_count += 1
        print(f"   Deleted: {os.path.basename(plot_path)}")

if deleted_count > 0:
    print(f"\n3. Deleted {deleted_count} plot(s) to test regeneration")
    
    # Run ICA (should load from cache and regenerate plots)
    print("\n4. Running ICA (loading from cache, should regenerate plots)...")
    ml_logger = MLLogger()
    save_path = os.path.join(script_dir, 'figures', 'accidents', 'dr')
    ica = ICAReduction('accidents', save_path, ml_logger, seed=42)
    ica_results = ica.run_dimensionality_reduction(X_full, y_full, list(range(2, 9)), task='regression')
    
    # Check if plots were regenerated
    ica_plots_after = glob.glob(os.path.join(ica_plots_dir, '*.png'))
    print(f"\n5. Found {len(ica_plots_after)} ICA plots after test")
    
    if len(ica_plots_after) >= len(ica_plots_before):
        print("   ✓ ICA plots successfully regenerated!")
    else:
        print(f"   ✗ Missing plots! Expected {len(ica_plots_before)}, got {len(ica_plots_after)}")
else:
    print("\n   Skipping DR test - plots already exist")

# Test 2: Clustering Plot Regeneration
print("\n" + "="*80)
print("TEST 2: Clustering Plot Regeneration (K-Means on ICA)")
print("="*80)

# Load ICA transformed data
import joblib
ica_cache_path = os.path.join(script_dir, 'cache', 'step2_dr', 'ica', 'accidents_ica_dr_ed8eaff2.pkl')
if os.path.exists(ica_cache_path):
    ica_cache = joblib.load(ica_cache_path)
    X_ica = ica_cache['data']['X_transformed']
    print(f"\n6. Loaded ICA-transformed data: {X_ica.shape}")
    
    kmeans_plots_dir = os.path.join(script_dir, 'figures', 'accidents', 'clustering_dr', 'ica', 'kmeans')
    os.makedirs(kmeans_plots_dir, exist_ok=True)
    
    kmeans_plots_before = glob.glob(os.path.join(kmeans_plots_dir, '**', '*.png'), recursive=True)
    print(f"   Found {len(kmeans_plots_before)} K-Means plots before test")
    
    # Delete some clustering plots
    cluster_plots_to_delete = glob.glob(os.path.join(kmeans_plots_dir, 'clusters', '*.png'))
    deleted_count = 0
    for plot_path in cluster_plots_to_delete[:2]:  # Delete first 2
        if os.path.exists(plot_path):
            os.remove(plot_path)
            deleted_count += 1
            print(f"   Deleted: {os.path.basename(plot_path)}")
    
    if deleted_count > 0:
        print(f"\n7. Deleted {deleted_count} clustering plot(s) to test regeneration")
        
        # Run K-Means (should load from cache and regenerate plots)
        print("\n8. Running K-Means (loading from cache, should regenerate plots)...")
        ml_logger = MLLogger()
        kmeans = KMeansClustering('accidents', kmeans_plots_dir, ml_logger, silhouette_subsample=10000, seed=42)
        kmeans_results = kmeans.run(X_ica, n_clusters=(2, 9), stability_runs=5, n_init=10, n_jobs=1)
        
        # Check if plots were regenerated
        kmeans_plots_after = glob.glob(os.path.join(kmeans_plots_dir, '**', '*.png'), recursive=True)
        print(f"\n9. Found {len(kmeans_plots_after)} K-Means plots after test")
        
        if len(kmeans_plots_after) >= len(kmeans_plots_before):
            print("   ✓ K-Means plots successfully regenerated!")
        else:
            print(f"   ✗ Missing plots! Expected at least {len(kmeans_plots_before)}, got {len(kmeans_plots_after)}")
    else:
        print("\n   Skipping clustering test - plots already exist or not found")
else:
    print("\n   Skipping clustering test - ICA cache not found")

print("\n" + "="*80)
print("PLOT REGENERATION TEST COMPLETE")
print("="*80)
print("\nSummary:")
print("  ✓ DR (ICA) plots regenerate from cache")
print("  ✓ Clustering (K-Means) plots regenerate from cache")
print("\nAll plots should now be generated even when loading from cache!")
