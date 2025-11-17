"""
Generate advanced figures for the Unsupervised Learning report.
Includes scree plots, comparison visualizations, and publication-ready figures.
"""

import os
import re
import sys
import json
import pickle
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from utils.plotter import plot_multiple_y_axes
from utils.data_processing import clean_hotels, clean_accidents, general_clean, winsorize_df, subsample_dataset

# Use non-interactive backend for saving figures
matplotlib.use('Agg')

# Set environment variable for data processing
os.environ['ROOT'] = str(Path(__file__).parent)

# IEEE-style formatting
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.titlesize'] = 10


def parse_pca_report(report_path):
    """Extract PCA explained variance data from execution report."""
    with open(report_path, 'r') as f:
        content = f.read()
    
    # Extract n_components and explained variance
    components = []
    explained_variance = []
    cumulative_variance = []
    reconstruction_errors = []
    
    # Pattern to match each result block including reconstruction error
    pattern = r'N Components: (\d+)\s+Mean Score: ([\d.]+)\s+Explained Variance: \[([\d\s.\[\]]+)\]\s+Reconstruction Error: ([\d.]+)'
    
    for match in re.finditer(pattern, content):
        n_comp = int(match.group(1))
        cum_var = float(match.group(2))
        recon_error = float(match.group(4))
        
        # Parse explained variance array
        var_str = match.group(3).strip()
        var_values = [float(x) for x in re.findall(r'[\d.]+', var_str)]
        
        components.append(n_comp)
        cumulative_variance.append(cum_var)
        reconstruction_errors.append(recon_error)
        
        # Store individual explained variance values
        if n_comp not in [len(ev) for ev in explained_variance]:
            explained_variance.append(var_values)
    
    return {
        'components': components,
        'explained_variance': explained_variance,
        'cumulative_variance': cumulative_variance,
        'reconstruction_errors': reconstruction_errors
    }


def parse_dr_report(report_path, method_name):
    """Extract dimensionality reduction data (PCA, ICA, RP) from execution report."""
    with open(report_path, 'r') as f:
        content = f.read()
    
    components = []
    reconstruction_errors = []
    
    # Pattern for reconstruction error
    if method_name.lower() == 'pca':
        pattern = r'N Components: (\d+)\s+.*?Reconstruction Error: ([\d.]+)'
    elif method_name.lower() == 'ica':
        pattern = r'N Components: (\d+)\s+.*?Reconstruction Error: ([\d.]+)'
    elif method_name.lower() in ['rp', 'random_projection']:
        pattern = r'N Components: (\d+)\s+.*?Reconstruction Error: ([\d.]+)'
    else:
        return {'components': [], 'reconstruction_errors': []}
    
    for match in re.finditer(pattern, content, re.DOTALL):
        n_comp = int(match.group(1))
        recon_error = float(match.group(2))
        
        components.append(n_comp)
        reconstruction_errors.append(recon_error)
    
    return {
        'components': components,
        'reconstruction_errors': reconstruction_errors
    }


def create_scree_plot(data, dataset_name, save_dir):
    """Create a scree plot showing explained variance per component using multiple y-axes."""
    
    # Get the full explained variance (from the largest n_components)
    full_variance = data['explained_variance'][-1] if data['explained_variance'] else []
    
    if not full_variance:
        print(f"No variance data found for {dataset_name}")
        return
    
    n_components = len(full_variance)
    component_numbers = list(range(1, n_components + 1))
    cumulative = np.cumsum(full_variance)
    
    # Use multiple y-axis function to plot both on same graph
    save_path = os.path.join(save_dir, f'{dataset_name}_scree_plot.png')
    
    plot_multiple_y_axes(
        x=component_numbers,
        y_series=[full_variance, cumulative],
        labels=['Explained Variance Ratio', 'Cumulative Variance Ratio'],
        colors=['#2E86AB', '#A23B72'],
        markers=['o', 's'],
        xlabel='Principal Component',
        title=f'{dataset_name.capitalize()} - PCA Scree Plot',
        save_path=save_path
    )
    
    print(f"Saved scree plot: {save_path}")
    
    # Also save as PDF for publication
    save_path_pdf = os.path.join(save_dir, f'{dataset_name}_scree_plot.pdf')
    
    plot_multiple_y_axes(
        x=component_numbers,
        y_series=[full_variance, cumulative],
        labels=['Explained Variance Ratio', 'Cumulative Variance Ratio'],
        colors=['#2E86AB', '#A23B72'],
        markers=['o', 's'],
        xlabel='Principal Component',
        title=f'{dataset_name.capitalize()} - PCA Scree Plot',
        save_path=save_path_pdf
    )
    
    print(f"Saved scree plot (PDF): {save_path_pdf}")


def create_combined_scree_plot(accidents_data, hotels_data, save_dir):
    """Create a combined scree plot comparing both datasets using multiple y-axes."""
    
    acc_variance = accidents_data['explained_variance'][-1] if accidents_data['explained_variance'] else []
    hot_variance = hotels_data['explained_variance'][-1] if hotels_data['explained_variance'] else []
    
    if not acc_variance or not hot_variance:
        print("Missing variance data for combined plot")
        return
    
    # Use minimum length for fair comparison
    min_len = min(len(acc_variance), len(hot_variance))
    component_numbers = list(range(1, min_len + 1))
    
    # Calculate cumulative variance
    acc_cumulative = np.cumsum(acc_variance[:min_len])
    hot_cumulative = np.cumsum(hot_variance[:min_len])
    
    # Create combined plot with 4 series on multiple y-axes
    save_path = os.path.join(save_dir, 'combined_scree_plot.png')
    
    plot_multiple_y_axes(
        x=component_numbers,
        y_series=[
            acc_variance[:min_len],
            hot_variance[:min_len],
            acc_cumulative,
            hot_cumulative
        ], # type: ignore
        labels=[
            'Accidents - Explained Variance',
            'Hotels - Explained Variance',
            'Accidents - Cumulative Variance',
            'Hotels - Cumulative Variance'
        ],
        colors=['#2E86AB', '#F18F01', '#A23B72', '#C73E1D'],
        markers=['o', 's', 'D', '^'],
        xlabel='Principal Component',
        title='PCA Variance Comparison: Accidents vs Hotels',
        save_path=save_path
    )
    
    print(f"Saved combined scree plot: {save_path}")
    
    # PDF version
    save_path_pdf = os.path.join(save_dir, 'combined_scree_plot.pdf')
    
    plot_multiple_y_axes(
        x=component_numbers,
        y_series=[
            acc_variance[:min_len],
            hot_variance[:min_len],
            acc_cumulative,
            hot_cumulative
        ], # type: ignore
        labels=[
            'Accidents - Explained Variance',
            'Hotels - Explained Variance',
            'Accidents - Cumulative Variance',
            'Hotels - Cumulative Variance'
        ],
        colors=['#2E86AB', '#F18F01', '#A23B72', '#C73E1D'],
        markers=['o', 's', 'D', '^'],
        xlabel='Principal Component',
        title='PCA Variance Comparison: Accidents vs Hotels',
        save_path=save_path_pdf
    )
    
    print(f"Saved combined scree plot (PDF): {save_path_pdf}")


def create_elbow_analysis(data, dataset_name, save_dir):
    """Create elbow plot with variance explained and recommended components."""
    
    full_variance = data['explained_variance'][-1] if data['explained_variance'] else []
    
    if not full_variance:
        print(f"No variance data for elbow analysis: {dataset_name}")
        return
    
    n_components = len(full_variance)
    component_numbers = list(range(1, n_components + 1))
    cumulative = np.cumsum(full_variance)
    
    # Calculate "elbow" using second derivative
    if len(cumulative) > 2:
        first_deriv = np.diff(cumulative)
        second_deriv = np.diff(first_deriv)
        
        # Find elbow (max curvature)
        elbow_idx = np.argmax(np.abs(second_deriv)) + 1
    else:
        elbow_idx = 0
    
    # Find components for 90% and 95% variance
    idx_90 = np.argmax(cumulative >= 0.90) + 1 if np.any(cumulative >= 0.90) else n_components
    idx_95 = np.argmax(cumulative >= 0.95) + 1 if np.any(cumulative >= 0.95) else n_components
    
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=300)
    
    # Plot cumulative variance
    ax.plot(component_numbers, cumulative, 'o-', color='#2E86AB', 
            linewidth=2, markersize=5, markerfacecolor='white', 
            markeredgewidth=1.5, markeredgecolor='#2E86AB',
            label='Cumulative Variance')
    
    # Mark important points
    if elbow_idx > 0:
        ax.axvline(x=elbow_idx+1, color='#F18F01', linestyle='--', 
                   linewidth=1.5, alpha=0.7, label=f'Elbow (n={elbow_idx+1})')
    
    ax.axhline(y=0.90, color='#A23B72', linestyle='--', 
               linewidth=1.5, alpha=0.7, label=f'90% (n={idx_90})')
    ax.axhline(y=0.95, color='#C73E1D', linestyle=':', 
               linewidth=1.5, alpha=0.7, label=f'95% (n={idx_95})')
    
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title(f'{dataset_name.capitalize()} - Component Selection Analysis')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='lower right', frameon=True, fancybox=False, framealpha=0.95)
    ax.set_xticks(component_numbers)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'{dataset_name}_elbow_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved elbow analysis: {save_path}")
    plt.close()


def create_reconstruction_error_plot(pca_data, ica_data, rp_data, dataset_name, save_dir):
    """Create reconstruction error comparison plot for all DR methods."""
    
    # Get reconstruction errors
    pca_errors = pca_data.get('reconstruction_errors', [])
    pca_comps = pca_data.get('components', [])
    
    ica_errors = ica_data.get('reconstruction_errors', [])
    ica_comps = ica_data.get('components', [])
    
    rp_errors = rp_data.get('reconstruction_errors', [])
    rp_comps = rp_data.get('components', [])
    
    if not pca_errors and not ica_errors and not rp_errors:
        print(f"No reconstruction error data for {dataset_name}")
        return
    
    # Prepare data for plotting
    y_series = []
    labels = []
    colors = []
    markers = []
    line_styles = []
    
    if pca_errors:
        y_series.append(pca_errors)
        labels.append('PCA Reconstruction Error')
        colors.append('#2E86AB')
        markers.append('o')
        line_styles.append('-')
    
    if ica_errors:
        y_series.append(ica_errors)
        labels.append('ICA Reconstruction Error')
        colors.append('#F18F01')
        markers.append('s')
        line_styles.append('--')
    
    if rp_errors:
        y_series.append(rp_errors)
        labels.append('RP Reconstruction Error')
        colors.append('#A23B72')
        markers.append('D')
        line_styles.append('-')
    
    # Use the maximum component range for x-axis
    max_comps = max(
        max(pca_comps) if pca_comps else 0,
        max(ica_comps) if ica_comps else 0,
        max(rp_comps) if rp_comps else 0
    )
    
    # All methods should have same components, use first available
    x_values = pca_comps if pca_comps else (ica_comps if ica_comps else rp_comps)
    
    save_path = os.path.join(save_dir, f'{dataset_name}_reconstruction_error.png')
    
    plot_multiple_y_axes(
        x=x_values,
        y_series=y_series,
        labels=labels,
        colors=colors,
        line_styles=line_styles,
        markers=markers,
        xlabel='Number of Components',
        title=f'{dataset_name.capitalize()} - Reconstruction Error Comparison',
        save_path=save_path
    )
    
    print(f"Saved reconstruction error plot: {save_path}")
    
    # PDF version
    save_path_pdf = os.path.join(save_dir, f'{dataset_name}_reconstruction_error.pdf')
    
    plot_multiple_y_axes(
        x=x_values,
        y_series=y_series,
        labels=labels,
        colors=colors,
        markers=markers,
        xlabel='Number of Components',
        title=f'{dataset_name.capitalize()} - Reconstruction Error Comparison',
        save_path=save_path_pdf
    )
    
    print(f"Saved reconstruction error plot (PDF): {save_path_pdf}")


def extract_clustering_metrics_from_cache(cache_dir, dataset_name, method_name):
    """Extract clustering metrics from cached pickle files."""
    
    # Find the appropriate cache file
    pattern = os.path.join(cache_dir, f'step1_clustering/{method_name.lower()}', 
                          f'{dataset_name}_{method_name.lower()}_clustering_*.pkl')
    
    pkl_files = glob.glob(pattern)
    
    if not pkl_files:
        print(f"No cache files found for {dataset_name} {method_name}")
        return None
    
    # Try to find the file with selection_results
    for pkl_file in pkl_files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            if 'selection_results' in data and data['selection_results']:
                selection_results = data['selection_results']
                
                # Extract metrics
                k_values = []
                inertias = []
                bics = []
                aics = []
                silhouettes = []
                
                for result in selection_results:
                    k = result.get('k') or result.get('n_components')
                    if k:
                        k_values.append(k)
                        inertias.append(result.get('inertia'))
                        bics.append(result.get('bic'))
                        aics.append(result.get('aic'))
                        silhouettes.append(result.get('silhouette'))
                
                return {
                    'k_values': k_values,
                    'inertias': inertias,
                    'bics': bics,
                    'aics': aics,
                    'silhouettes': silhouettes
                }
        except Exception as e:
            continue
    
    return None


def create_clustering_elbow_plot(dataset_name, save_dir, cache_dir):
    """Create comprehensive elbow/BIC/AIC plot for clustering."""
    
    print(f"\nExtracting clustering metrics for {dataset_name}...")
    
    # Extract K-Means metrics
    kmeans_data = extract_clustering_metrics_from_cache(cache_dir, dataset_name, 'k-means')
    
    # Extract GMM metrics
    gmm_data = extract_clustering_metrics_from_cache(cache_dir, dataset_name, 'gmm')
    
    if not kmeans_data and not gmm_data:
        print(f"No clustering data found for {dataset_name}")
        return
    
    # Prepare y-series for plotting
    y_series = []
    labels = []
    colors = []
    markers = []
    
    # Use K-Means k values as x-axis (should be same for both)
    x_values = kmeans_data['k_values'] if kmeans_data else gmm_data['k_values']
    
    # K-Means Inertia (Elbow)
    if kmeans_data and kmeans_data['inertias'] and any(i is not None for i in kmeans_data['inertias']):
        y_series.append([i if i is not None else 0 for i in kmeans_data['inertias']])
        labels.append('K-Means Inertia')
        colors.append('#2E86AB')
        markers.append('o')
    
    # GMM BIC
    if gmm_data and gmm_data['bics'] and any(b is not None for b in gmm_data['bics']):
        y_series.append([b if b is not None else 0 for b in gmm_data['bics']])
        labels.append('GMM BIC')
        colors.append('#F18F01')
        markers.append('s')
    
    # GMM AIC
    if gmm_data and gmm_data['aics'] and any(a is not None for a in gmm_data['aics']):
        y_series.append([a if a is not None else 0 for a in gmm_data['aics']])
        labels.append('GMM AIC')
        colors.append('#A23B72')
        markers.append('D')
    
    # K-Means Silhouette
    if kmeans_data and kmeans_data['silhouettes'] and any(s is not None for s in kmeans_data['silhouettes']):
        y_series.append([s if s is not None else 0 for s in kmeans_data['silhouettes']])
        labels.append('K-Means Silhouette')
        colors.append('#C73E1D')
        markers.append('^')
    
    # GMM Silhouette
    if gmm_data and gmm_data['silhouettes'] and any(s is not None for s in gmm_data['silhouettes']):
        y_series.append([s if s is not None else 0 for s in gmm_data['silhouettes']])
        labels.append('GMM Silhouette')
        colors.append('#6A994E')
        markers.append('v')
    
    if not y_series:
        print(f"No valid metrics found for {dataset_name}")
        return
    
    save_path = os.path.join(save_dir, f'{dataset_name}_clustering_elbow_bic_aic.png')
    
    plot_multiple_y_axes(
        x=x_values,
        y_series=y_series,
        labels=labels,
        colors=colors,
        markers=markers,
        xlabel='Number of Clusters (k)',
        title=f'{dataset_name.capitalize()} - Clustering Model Selection',
        save_path=save_path
    )
    
    print(f"Saved clustering elbow/BIC/AIC plot: {save_path}")
    
    # PDF version
    save_path_pdf = os.path.join(save_dir, f'{dataset_name}_clustering_elbow_bic_aic.pdf')
    
    plot_multiple_y_axes(
        x=x_values,
        y_series=y_series,
        labels=labels,
        colors=colors,
        markers=markers,
        xlabel='Number of Clusters (k)',
        title=f'{dataset_name.capitalize()} - Clustering Model Selection',
        save_path=save_path_pdf
    )
    
    print(f"Saved clustering elbow/BIC/AIC plot (PDF): {save_path_pdf}")


def load_processed_features(dataset_name, data_dir):
    """Load dataset with cleaning/engineering applied but BEFORE scaling to see actual variance."""
    
    if dataset_name == 'accidents':
        file_path = data_dir / 'US_Accidents_March23_2M_rows.csv'
        target = 'Severity'
        method = 'regression'
    else:  # hotels
        file_path = data_dir / 'hotel_bookings.csv'
        target = 'is_canceled'
        method = 'classification'
    
    print(f"Loading and processing {dataset_name} data (before scaling)...")
    
    # Load raw data
    df = pd.read_csv(file_path)
    print(f"  Raw data: {len(df):,} rows")
    
    # Apply same cleaning as in data_processing.py
    if dataset_name == 'hotels':
        df = clean_hotels(df)
    else:
        df = clean_accidents(df)
    
    df = general_clean(df, target)
    print(f"  After cleaning: {len(df):,} rows")
    
    # Winsorize numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        df = winsorize_df(df, numeric_cols, lower=0.01, upper=0.99)
    
    # Log transform for regression target
    if method == 'regression' and 'Duration_Seconds' in df.columns:
        df['Duration_Seconds'] = np.log1p(df['Duration_Seconds'])
    
    # Separate features and target
    X = df.drop(columns=[target], errors='ignore')
    
    # Keep only numeric features for variance calculation
    numeric_features = X.select_dtypes(include=[np.number])
    
    print(f"  Numeric features: {numeric_features.shape[1]}")
    
    # Calculate variance for each feature
    variances = numeric_features.var()
    
    # Sort by variance (descending)
    variances_sorted = variances.sort_values(ascending=False)
    
    return variances_sorted


def create_feature_variance_plot(dataset_name, save_dir, data_dir):
    """Create bar chart showing variance of all processed features (before scaling)."""
    
    print(f"\nCreating feature variance plot for {dataset_name}...")
    
    variances = load_processed_features(dataset_name, data_dir)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bar chart
    x_pos = np.arange(len(variances))
    bars = ax.bar(x_pos, variances.values, color='#2E86AB', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Customize plot
    ax.set_xlabel('Features', fontsize=11, weight='bold')
    ax.set_ylabel('Variance', fontsize=11, weight='bold')
    ax.set_title(f'{dataset_name.capitalize()} - Feature Variance Distribution', 
                 fontsize=13, weight='bold', pad=15)
    
    # Set x-axis labels (rotated for readability)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(variances.index, rotation=90, ha='right', fontsize=8)
    
    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add stats text box
    stats_text = f'Features: {len(variances)}\n'
    stats_text += f'Max Variance: {variances.max():.2e}\n'
    stats_text += f'Min Variance: {variances.min():.2e}\n'
    stats_text += f'Mean Variance: {variances.mean():.2e}'
    
    ax.text(0.98, 0.97, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout()
    
    # Save PNG
    save_path = os.path.join(save_dir, f'{dataset_name}_feature_variance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved feature variance plot: {save_path}")
    
    # Save PDF
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(x_pos, variances.values, color='#2E86AB', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Features', fontsize=11, weight='bold')
    ax.set_ylabel('Variance', fontsize=11, weight='bold')
    ax.set_title(f'{dataset_name.capitalize()} - Feature Variance Distribution', 
                 fontsize=13, weight='bold', pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(variances.index, rotation=90, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.text(0.98, 0.97, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    plt.tight_layout()
    
    save_path_pdf = os.path.join(save_dir, f'{dataset_name}_feature_variance.pdf')
    plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved feature variance plot (PDF): {save_path_pdf}")
    
    # Also save variance data to CSV for reference
    csv_path = os.path.join(save_dir, f'{dataset_name}_feature_variance.csv')
    variances.to_csv(csv_path, header=['Variance'])
    print(f"Saved variance data: {csv_path}")


def main():
    """Generate all advanced figures."""
    
    # Define paths
    base_dir = Path(__file__).parent
    figures_dir = base_dir / 'figures'
    cache_dir = base_dir / 'cache'
    data_dir = base_dir / 'data'
    
    # PCA reports
    accidents_pca_report = figures_dir / 'accidents' / 'dr' / 'pca' / 'execution_report.txt'
    hotels_pca_report = figures_dir / 'hotels' / 'dr' / 'pca' / 'execution_report.txt'
    
    # ICA reports
    accidents_ica_report = figures_dir / 'accidents' / 'dr' / 'ica' / 'execution_report.txt'
    hotels_ica_report = figures_dir / 'hotels' / 'dr' / 'ica' / 'execution_report.txt'
    
    # RP reports
    accidents_rp_report = figures_dir / 'accidents' / 'dr' / 'random_projection' / 'execution_report.txt'
    hotels_rp_report = figures_dir / 'hotels' / 'dr' / 'random_projection' / 'execution_report.txt'
    
    # Output directory
    output_dir = figures_dir / 'report_plots'
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("Generating Advanced Figures for Unsupervised Learning Report")
    print("=" * 70)
    
    # Parse PCA reports
    print("\nParsing PCA execution reports...")
    accidents_pca_data = parse_pca_report(accidents_pca_report)
    hotels_pca_data = parse_pca_report(hotels_pca_report)
    
    print(f"Accidents PCA: {len(accidents_pca_data['components'])} configurations found")
    print(f"Hotels PCA: {len(hotels_pca_data['components'])} configurations found")
    
    # Parse ICA reports
    print("\nParsing ICA execution reports...")
    accidents_ica_data = parse_dr_report(accidents_ica_report, 'ica')
    hotels_ica_data = parse_dr_report(hotels_ica_report, 'ica')
    
    print(f"Accidents ICA: {len(accidents_ica_data['components'])} configurations found")
    print(f"Hotels ICA: {len(hotels_ica_data['components'])} configurations found")
    
    # Parse RP reports
    print("\nParsing RP execution reports...")
    accidents_rp_data = parse_dr_report(accidents_rp_report, 'rp')
    hotels_rp_data = parse_dr_report(hotels_rp_report, 'rp')
    
    print(f"Accidents RP: {len(accidents_rp_data['components'])} configurations found")
    print(f"Hotels RP: {len(hotels_rp_data['components'])} configurations found")
    
    # Generate scree plots
    print("\n" + "=" * 70)
    print("Generating Scree Plots")
    print("=" * 70)
    
    create_scree_plot(accidents_pca_data, 'accidents', output_dir)
    create_scree_plot(hotels_pca_data, 'hotels', output_dir)
    create_combined_scree_plot(accidents_pca_data, hotels_pca_data, output_dir)
    
    # Generate elbow analysis
    print("\n" + "=" * 70)
    print("Generating Elbow Analysis")
    print("=" * 70)
    
    create_elbow_analysis(accidents_pca_data, 'accidents', output_dir)
    create_elbow_analysis(hotels_pca_data, 'hotels', output_dir)
    
    # Generate reconstruction error plots
    print("\n" + "=" * 70)
    print("Generating Reconstruction Error Plots")
    print("=" * 70)
    
    create_reconstruction_error_plot(
        accidents_pca_data, accidents_ica_data, accidents_rp_data,
        'accidents', output_dir
    )
    create_reconstruction_error_plot(
        hotels_pca_data, hotels_ica_data, hotels_rp_data,
        'hotels', output_dir
    )
    
    # Generate clustering elbow/BIC/AIC plots
    print("\n" + "=" * 70)
    print("Generating Clustering Elbow/BIC/AIC Plots")
    print("=" * 70)
    
    create_clustering_elbow_plot('accidents', output_dir, cache_dir)
    create_clustering_elbow_plot('hotels', output_dir, cache_dir)
    
    # Generate feature variance plots
    print("\n" + "=" * 70)
    print("Generating Feature Variance Plots")
    print("=" * 70)
    
    create_feature_variance_plot('accidents', output_dir, data_dir)
    create_feature_variance_plot('hotels', output_dir, data_dir)
    
    print("\n" + "=" * 70)
    print("All figures generated successfully!")
    print(f"Output directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
