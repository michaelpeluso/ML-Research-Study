import os
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics.pairwise import euclidean_distances

from utils.data_processing import split_processed_data, wrap_into_loaders
from utils.logger import MLLogger
from nerual_networks.models import set_seed, MLP


def extract_cluster_features(
    X: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_centers: Optional[np.ndarray] = None,
    responsibilities: Optional[np.ndarray] = None,
    feature_mode: str = 'all'
) -> np.ndarray:
    """Extract features from clustering results."""
    n_samples = X.shape[0]
    n_clusters = len(np.unique(cluster_labels))
    features_list = []
    
    # One-hot encoded cluster labels (always available)
    if feature_mode in ['labels_only', 'all']:
        one_hot = np.zeros((n_samples, n_clusters))
        one_hot[np.arange(n_samples), cluster_labels] = 1
        features_list.append(one_hot)
    
    # Distances to cluster centroids (K-Means)
    if feature_mode in ['distances', 'all'] and cluster_centers is not None:
        distances = euclidean_distances(X, cluster_centers)
        features_list.append(distances)
    
    # EM responsibilities (soft assignments)
    if feature_mode in ['responsibilities', 'all'] and responsibilities is not None:
        features_list.append(responsibilities)
    
    if not features_list:
        raise ValueError(f"No features extracted with mode={feature_mode}")
    
    return np.concatenate(features_list, axis=1)


def run_neural_networks_with_clusters(
    X_original: np.ndarray,
    y_original: np.ndarray,
    dataset: str,
    method: str,
    save_path: str,
    ml_logger: MLLogger,
    seed: int,
    clustering_results: Dict[str, Any],
    batch_size: int = 1024,
    feature_setup: str = 'additive',  # 'replacement' or 'additive'
    max_updates: int = 5,  # Start with SL optimal
    learning_rate: float = 0.01,  # Start with SL optimal
    weight_decay: float = 0.001,  # Start with SL optimal
    hidden_layers: List[int] = [512, 512],  # Start with SL optimal
    activation: str = 'relu',  # Start with SL optimal
    tune_if_needed: bool = True  # Allow tuning for cluster features
) -> Dict[str, Any]:
    """Train neural networks using cluster-derived features."""
    
    start_time = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    
    # Get clustering results
    kmeans_results = clustering_results['kmeans']
    em_results = clustering_results['em']
    
    # Extract cluster assignments and metadata
    kmeans_labels = kmeans_results['labels']
    kmeans_centers = kmeans_results['centers']  # Changed from cluster_centers
    
    em_labels = em_results['labels']
    em_means = em_results['centers']  # Changed from cluster_centers
    
    # Extract EM responsibilities (soft assignments) from model
    em_model = em_results.get('model', None)
    em_responsibilities = None
    if em_model is not None and hasattr(em_model, 'predict_proba'):
        try:
            em_responsibilities = em_model.predict_proba(X_original)
            print(f"Extracted EM responsibilities: {em_responsibilities.shape}")
        except Exception as e:
            print(f" Could not extract EM responsibilities: {e}")
    
    print(f"\n{'='*80}")
    print(f"Step 5: Neural Networks with Cluster Features ({dataset.upper()})")
    print(f"Feature Setup: {feature_setup.upper()}")
    print(f"K-Means: {len(np.unique(kmeans_labels))} clusters")
    print(f"EM/GMM: {len(np.unique(em_labels))} clusters")
    if em_responsibilities is not None:
        print(f"EM/GMM soft assignments (responsibilities) available")
    print(f"{'='*80}\n")
    
    # Create baseline comparison (original features only)
    print("\n" + "-"*80 + "\nBaseline: NN on Original Features (for comparison)\n" + "-"*80)
    results['baseline'] = train_nn_on_features(
        X_original, y_original, "Original Features", method, seed, device,
        batch_size, max_updates, learning_rate, weight_decay, hidden_layers, activation, ml_logger
    )
    
    # K-Means derived features
    print("\n" + "-"*80 + "\nNeural Network with K-Means Cluster Features\n" + "-"*80)
    
    # Extract K-Means features
    kmeans_cluster_features = extract_cluster_features(
        X_original, kmeans_labels, cluster_centers=kmeans_centers, feature_mode='all'
    )
    
    if feature_setup == 'replacement':
        X_kmeans = kmeans_cluster_features
        feature_desc = "K-Means Cluster Features Only"
    else:  # additive
        X_kmeans = np.concatenate([X_original, kmeans_cluster_features], axis=1)
        feature_desc = "Original + K-Means Cluster Features"
    
    print(f"K-Means feature shape: {X_kmeans.shape} (original: {X_original.shape})")
    
    # Train with K-Means features
    kmeans_max_updates = max_updates
    kmeans_lr = learning_rate
    
    results['kmeans'] = train_nn_on_features(
        X_kmeans, y_original, feature_desc, method, seed, device,
        batch_size, kmeans_max_updates, kmeans_lr, weight_decay, hidden_layers, activation, ml_logger,
        n_cluster_features=kmeans_cluster_features.shape[1]
    )
    
    # Tune if needed (based on convergence)
    if tune_if_needed and results['kmeans']['final_train_loss'] > results['baseline']['final_train_loss'] * 1.5:
        print("\nK-Means features showing poor convergence. Tuning learning rate...")
        kmeans_lr = learning_rate * 0.5  # Reduce learning rate
        kmeans_max_updates = int(max_updates * 1.5)  # Allow more iterations
        print(f"Adjusted: lr={kmeans_lr}, max_updates={kmeans_max_updates}")
        
        results['kmeans_tuned'] = train_nn_on_features(
            X_kmeans, y_original, f"{feature_desc} (Tuned)", method, seed, device,
            batch_size, kmeans_max_updates, kmeans_lr, weight_decay, hidden_layers, activation, ml_logger,
            n_cluster_features=kmeans_cluster_features.shape[1]
        )
    
    # EM/GMM derived features
    print("\n" + "-"*80 + "\nNeural Network with EM/GMM Cluster Features\n" + "-"*80)
    
    # Extract EM features (include responsibilities if available)
    em_cluster_features = extract_cluster_features(
        X_original, em_labels, cluster_centers=em_means, 
        responsibilities=em_responsibilities, feature_mode='all'
    )
    
    if feature_setup == 'replacement':
        X_em = em_cluster_features
        feature_desc = "EM/GMM Cluster Features Only"
    else:  # additive
        X_em = np.concatenate([X_original, em_cluster_features], axis=1)
        feature_desc = "Original + EM/GMM Cluster Features"
    
    print(f"EM/GMM feature shape: {X_em.shape} (original: {X_original.shape})")
    
    # Train with EM features
    em_max_updates = max_updates
    em_lr = learning_rate
    
    results['em'] = train_nn_on_features(
        X_em, y_original, feature_desc, method, seed, device,
        batch_size, em_max_updates, em_lr, weight_decay, hidden_layers, activation, ml_logger,
        n_cluster_features=em_cluster_features.shape[1]
    )
    
    # Tune if needed
    if tune_if_needed and results['em']['final_train_loss'] > results['baseline']['final_train_loss'] * 1.5:
        print("\n EM/GMM features showing poor convergence. Tuning learning rate...")
        em_lr = learning_rate * 0.5
        em_max_updates = int(max_updates * 1.5)
        print(f"Adjusted: lr={em_lr}, max_updates={em_max_updates}")
        
        results['em_tuned'] = train_nn_on_features(
            X_em, y_original, f"{feature_desc} (Tuned)", method, seed, device,
            batch_size, em_max_updates, em_lr, weight_decay, hidden_layers, activation, ml_logger,
            n_cluster_features=em_cluster_features.shape[1]
        )
    
    # Summary comparison
    print("\n" + "="*80)
    print("Step 5 - Neural Network with Cluster Features Summary")
    print("="*80)
    print(f"{'Configuration':<40} {'Test Loss':<12} {'Train Loss':<12} {'Wall Time':<12}")
    print("-" * 80)
    
    for config_name, result in results.items():
        print(f"{config_name:<40} {result['test_loss']:<12.4f} {result['final_train_loss']:<12.4f} {result['wall_time']:<12.2f}")
    
    # Analysis
    print("\n" + "-"*80)
    print("Analysis:")
    print("-"*80)
    
    baseline_test_loss = results['baseline']['test_loss']
    kmeans_key = 'kmeans_tuned' if 'kmeans_tuned' in results else 'kmeans'
    em_key = 'em_tuned' if 'em_tuned' in results else 'em'
    
    kmeans_improvement = ((baseline_test_loss - results[kmeans_key]['test_loss']) / baseline_test_loss) * 100
    em_improvement = ((baseline_test_loss - results[em_key]['test_loss']) / baseline_test_loss) * 100
    
    print(f"K-Means cluster features: {kmeans_improvement:+.2f}% change vs baseline")
    print(f"EM/GMM cluster features: {em_improvement:+.2f}% change vs baseline")
    
    if 'kmeans_tuned' in results:
        print(f"\nK-Means features required tuning (convergence issue)")
    if 'em_tuned' in results:
        print(f"EM/GMM features required tuning (convergence issue)")
    
    total_time = time.perf_counter() - start_time
    print(f"\nStep 5 completed in {total_time:.2f}s")
    
    # Log results
    with ml_logger.log_step(f"Step 5: Neural Networks with Cluster Features ({dataset})") as step_info:
        step_info['feature_setup'] = feature_setup
        step_info['results'] = results
        step_info['total_time'] = total_time
        step_info['kmeans_improvement_pct'] = kmeans_improvement
        step_info['em_improvement_pct'] = em_improvement
    
    return results


def train_nn_on_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_description: str,
    method: str,
    seed: int,
    device: torch.device,
    batch_size: int,
    max_updates: int,
    learning_rate: float,
    weight_decay: float,
    hidden_layers: List[int],
    activation: str,
    ml_logger: MLLogger,
    n_cluster_features: Optional[int] = None
) -> Dict[str, Any]:
    """Train a single neural network on given features."""
    
    # Create consistent splits
    X_train, X_val, X_test, y_train, y_val, y_test = split_processed_data(
        X, y, method=method, test_size=0.2, val_size=0.2, seed=seed, ml_logger=ml_logger
    )
    
    train_loader, val_loader, test_loader = wrap_into_loaders(
        method, X_train, X_val, X_test, y_train, y_val, y_test, batch_size
    )
    
    # Setup model
    set_seed(seed)
    in_dim = X_train.shape[1]
    out_dim = 1 if method == 'regression' else len(np.unique(y))
    
    model = MLP(
        in_dim=in_dim,
        hidden=hidden_layers,
        out_dim=out_dim,
        dropout_p=0.0,
        activation=activation
    ).to(device)
    
    # Use simple SGD optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Loss function
    if method == 'regression':
        loss_fn = nn.MSELoss()
    else:
        loss_fn = nn.CrossEntropyLoss()
    
    if n_cluster_features is not None:
        print(f"Cluster features: {n_cluster_features} dimensions")
    
    # Train
    from nerual_networks.training import train_to_budget, eval_loss
    
    train_start = time.perf_counter()
    curves, steps_to_threshold, wall_time, final_train_loss = train_to_budget(
        model, optimizer, train_loader, val_loader,
        max_updates, float('inf'), loss_fn, device,
        log_interval=500, eval_interval=1  # Changed from 100 to 1 for max_updates=5
    )
    
    # Evaluate on test set
    test_loss = eval_loss(model, test_loader, loss_fn, device)
    
    return {
        'test_loss': test_loss,
        'final_train_loss': final_train_loss,
        'wall_time': wall_time,
        'curves': curves,
        'steps_to_threshold': steps_to_threshold,
        'n_params': sum(p.numel() for p in model.parameters()),
        'n_trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'input_dim': in_dim,
        'n_cluster_features': n_cluster_features
    }
