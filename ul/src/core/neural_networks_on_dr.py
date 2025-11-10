import os
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional

from utils.data_processing import split_processed_data, wrap_into_loaders
from utils.logger import MLLogger, print_t as print
from utils.cache_manager import CacheManager
from utils.plotter import plot_curve
from nerual_networks.training import eval_loss, train_to_budget
from nerual_networks.models import set_seed, MLP
from nerual_networks.optimizers import optimizer_factory


def log_training_curves(
    ml_logger: MLLogger,
    method: str,
    curves: List[float],
    final_train_loss: float,
    test_loss: float,
    n_components: Optional[int] = None
) -> None:
    """Helper function to log training curves consistently."""
    log_data = {
        'method': method,
        'validation_losses': curves,
        'final_val_loss': curves[-1] if curves else None,
        'final_train_loss': final_train_loss,
        'test_loss': test_loss
    }
    if n_components is not None: log_data['n_components'] = n_components
    
    ml_logger.log_learning_curve(log_data)


def print_result_summary(method_name, test_loss, train_loss, wall_time, n_components=None):
    comp_str = f" (n={n_components})" if n_components is not None else ""
    print(f"\n{method_name}{comp_str} - Test Loss: {test_loss:.4f}, "
          f"Train Loss: {train_loss:.4f}, "
          f"Wall Time: {wall_time:.2f}s")


def _log_step_results(
    ml_logger: MLLogger,
    step_name: str,
    results: Dict[str, Any],
    hyperparameters: Dict[str, Any],
    dataset: str
) -> None:
    """Helper function to log comprehensive step results."""
    orig_result = results['original']
    baseline_test_loss = orig_result['test_loss']
    
    with ml_logger.log_step(f"{step_name} ({dataset})") as step_info:
        step_info['hyperparameters'] = hyperparameters
        
        # Log baseline performance
        step_info['baseline'] = {
            'test_loss': orig_result['test_loss'],
            'train_loss': orig_result['final_train_loss'],
            'wall_time': orig_result['wall_time'],
            'n_params': orig_result['n_params'],
            'n_trainable': orig_result['n_trainable']
        }
        
        # Log DR method results and improvements
        step_info['dr_results'] = {}
        dr_methods = ['pca', 'ica', 'rp']
        for dr_method in dr_methods:
            if dr_method in results:
                dr_result = results[dr_method]
                test_loss_change_pct = ((dr_result['test_loss'] - baseline_test_loss) / baseline_test_loss) * 100
                train_loss_change_pct = ((dr_result['final_train_loss'] - orig_result['final_train_loss']) / orig_result['final_train_loss']) * 100
                
                step_info['dr_results'][dr_method] = {
                    'test_loss': dr_result['test_loss'],
                    'train_loss': dr_result['final_train_loss'],
                    'wall_time': dr_result['wall_time'],
                    'n_components': dr_result['n_components'],
                    'n_params': dr_result['n_params'],
                    'test_loss_change_pct': test_loss_change_pct,
                    'train_loss_change_pct': train_loss_change_pct,
                    'improvement': test_loss_change_pct < 0
                }
        
        # Find best DR method
        if dr_methods:
            best_dr_method = min(dr_methods, key=lambda m: results[m]['test_loss'])
            best_dr_result = results[best_dr_method]
            step_info['best_dr_method'] = best_dr_method
            step_info['best_dr_test_loss'] = best_dr_result['test_loss']
            step_info['best_dr_improvement_pct'] = ((best_dr_result['test_loss'] - baseline_test_loss) / baseline_test_loss) * 100
        
        step_info['results'] = results


def _print_performance_comparison(
    results: Dict[str, Any],
    dr_methods: List[str]
) -> None:
    """Helper function to print performance comparison table."""
    print("\n" + "="*80)
    print("STEP 4: Neural Network Performance Summary")
    print("="*80)
    print(f"{'Method':<15} {'Test Loss':<12} {'Train Loss':<12} {'Wall Time':<12} {'N Components':<15}")
    print("-" * 80)

    # Original data
    orig_result = results['original']
    print(f"{'Original':<15} {orig_result['test_loss']:<12.4f} "
          f"{orig_result['final_train_loss']:<12.4f} "
          f"{orig_result['wall_time']:<12.2f} {'N/A':<15}")

    # DR methods
    baseline_test_loss = orig_result['test_loss']
    for dr_method in dr_methods:
        if dr_method in results:
            dr_result = results[dr_method]
            n_comp = dr_result.get('n_components', 'N/A')
            print(f"{dr_method.upper():<15} {dr_result['test_loss']:<12.4f} "
                  f"{dr_result['final_train_loss']:<12.4f} "
                  f"{dr_result['wall_time']:<12.2f} {str(n_comp):<15}")

    # Performance changes
    print("\n" + "-" * 80)
    print("Performance Changes vs Original:")
    print("-" * 80)
    for dr_method in dr_methods:
        if dr_method in results:
            dr_result = results[dr_method]
            test_loss_change = ((dr_result['test_loss'] - baseline_test_loss) / baseline_test_loss) * 100
            time_change = ((dr_result['wall_time'] - orig_result['wall_time']) / orig_result['wall_time']) * 100
            n_comp = dr_result.get('n_components', 'N/A')
            
            improvement_indicator = "↓" if test_loss_change < 0 else "↑"
            time_indicator = "faster" if time_change < 0 else "slower"
            
            print(f"{dr_method.upper():<5} (n={n_comp:<3}): "
                  f"Test loss {improvement_indicator} {abs(test_loss_change):>5.2f}%, "
                  f"{abs(time_change):>5.1f}% {time_indicator}")


def _plot_learning_curves(results, dr_methods, dataset, save_path, eval_interval=100):
    """Plot individual and combined learning curves for Step 4."""
    os.makedirs(save_path, exist_ok=True)
    
    colors = {'original': '#1f77b4', 'pca': '#2ca02c','ica': '#d62728','rp': '#ff7f0e'}
    all_methods = ['original'] + dr_methods
    
    def process_curves(method):
        """Extract and process curves for a method."""
        result = results[method]
        curves = result.get('curves', [])
        if not curves: return None
        
        curves_array = np.array(curves) if isinstance(curves[0], list) else np.array([curves])
        median = np.median(curves_array, axis=0)
        q1 = np.percentile(curves_array, 25, axis=0)
        q3 = np.percentile(curves_array, 75, axis=0)
        updates = np.arange(1, len(median) + 1) * eval_interval
        
        name = method.upper() if method != 'original' else 'Original'
        n_comp = result.get('n_components', '')
        if n_comp: name += f" (n={n_comp})"
        
        return {'median': median, 'q1': q1, 'q3': q3, 'updates': updates, 'name': name}
    
    # Plot individual curves
    for method in all_methods:
        if method not in results: continue
        data = process_curves(method)
        if not data: continue
        
        plot_curve(
            x=data['updates'], y_list=[data['median']], labels=[f"{data['name']} Median"],
            xlabel="Training Updates", ylabel="Validation Loss",
            title=f"Step 4: {data['name']} Learning Curve on {dataset.title()}",
            save_path=f"{save_path}/{method}_learning_curve.png",
            colors=[colors.get(method, '#333333')],
            lower=data['q1'], upper=data['q3'], band_label='IQR (25th-75th percentile)'
        )
    
    # Combined curves
    combined_data = [process_curves(m) for m in all_methods if m in results]
    combined_data = [d for d in combined_data if d is not None]
    
    if combined_data:
        plot_curve(
            x=[d['updates'] for d in combined_data],
            y_list=[d['median'] for d in combined_data],
            labels=[d['name'] for d in combined_data],
            colors=[colors.get(all_methods[i], f'C{i}') for i in range(len(combined_data))],
            linestyles=['--'] + ['-'] * (len(combined_data) - 1),
            xlabel="Training Updates", ylabel="Validation Loss",
            title=f"Step 4: Cumulative NN Learning Curves on {dataset.title()}",
            save_path=f"{save_path}/combined_learning_curves.png"
        )

    print(f"\nSaved learning curves to: {save_path}")


def _train_single_method(
    X: np.ndarray, y: np.ndarray, method_key: str, display_name: str,
    n_components: int, **train_params
) -> Tuple[str, str, Dict[str, Any]]:
    """Train a single method and return results. Used for parallel execution."""
    # Create a separate logger for this method to avoid conflicts
    from utils.logger import MLLogger
    import tempfile
    import os
    
    # Use a temporary log file for this method
    temp_log = tempfile.NamedTemporaryFile(delete=False, suffix='.log')
    temp_log.close()
    
    ml_logger = MLLogger(log_file=temp_log.name)
    ml_logger.log_metric(f'{method_key}_training_start', time.perf_counter())
    
    result = _train_nn_on_data(X=X, y=y, dataset_name=display_name, ml_logger=ml_logger, **train_params)
    
    if n_components is not None:
        result['n_components'] = n_components
    
    # Clean up temp log file
    try:
        os.unlink(temp_log.name)
    except:
        pass
    
    return method_key, display_name, result


def _train_nn_on_data(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    method: str,
    device: torch.device,
    loss_fn: nn.Module,
    ml_logger: MLLogger,
    seed: int,
    batch_size: int,
    optimizer_kind: str,
    hidden_layers: List[int],
    max_updates: int,
    learning_rate: float,
    betas: Tuple[float, float],
    weight_decay: float,
    dropout_p: float,
    l_threshold: float,
    label_smoothing_alpha: float,
    activation: str,
    out_dim: int
) -> Dict[str, Any]:
    """Helper function to train a neural network on given data.
    
    Returns dict with test_loss, final_train_loss, wall_time, curves, steps_to_threshold, n_params, n_trainable.
    """
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_processed_data(
        X, y, method=method, test_size=0.2, val_size=0.2, seed=seed, ml_logger=ml_logger
    )
    train_loader, val_loader, test_loader = wrap_into_loaders(
        method, X_train, X_val, X_test, y_train, y_val, y_test, batch_size
    )
    
    # Setup model
    set_seed(seed)
    in_dim = X_train.shape[1]
    model = MLP(
        in_dim=in_dim,
        hidden=hidden_layers,
        out_dim=out_dim,
        dropout_p=dropout_p,
        activation=activation
    ).to(device)
    
    # Setup optimizer
    optimizer = optimizer_factory(
        model=model,
        kind=optimizer_kind,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=betas,
    )
    
    # Train model
    curves, steps_to_threshold, wall_time, final_train_loss = train_to_budget(
        model, optimizer, train_loader, val_loader, #type:ignore
        max_updates, l_threshold, loss_fn, device,
        log_interval=25, eval_interval=25,
        optimizer_name=optimizer_kind
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
        'n_trainable': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }


def run_neural_networks_on_dr_data(
    X_original: np.ndarray,
    y_original: np.ndarray,
    dataset: str,
    method: str,
    save_path: str,
    ml_logger: MLLogger,
    seed: int,
    dr_results: Dict[str, Any],
    batch_size: int,
    optimizer: str,
    hidden_layers: List[int],
    max_updates: int,
    learning_rate: float,
    betas: Tuple[float, float],
    weight_decay: float,
    dropout_p: float,
    l_threshold: float,
    label_smoothing_alpha: float,
    activation: str
) -> Dict[str, Any]:
    """Retrain neural networks on DR data and compare with original."""
    start_time = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_manager = CacheManager()  # Use centralized cache manager
    results = {}
    
    # Check cache
    cache_params = {
        'seed': seed,
        'method': method,
        'batch_size': batch_size,
        'optimizer': optimizer,
        'hidden_layers': tuple(hidden_layers),  # Convert to tuple for hashing
        'max_updates': max_updates,
        'learning_rate': learning_rate,
        'betas': betas,
        'weight_decay': weight_decay,
        'dropout_p': dropout_p,
        'l_threshold': l_threshold,
        'label_smoothing_alpha': label_smoothing_alpha,
        'activation': activation,
        'n_classes': len(np.unique(y_original)),
        'dr_methods': sorted(list(dr_results.keys()))
    }
    cached_result = cache_manager.load(dataset, 'neural_networks_on_dr', cache_params)
    if cached_result is not None:
        print(f"Loaded cached results for {dataset} neural networks on DR data")
        return cached_result
    
    # Determine output dimension and loss function
    out_dim = 1 if method == 'regression' else len(np.unique(y_original))
    loss_fn = nn.MSELoss() if method == 'regression' else nn.CrossEntropyLoss(label_smoothing=label_smoothing_alpha)
    
    # Shared training parameters
    train_params = {
        'method': method, 'device': device, 'loss_fn': loss_fn,
        'seed': seed, 'batch_size': batch_size, 'optimizer_kind': optimizer,
        'hidden_layers': hidden_layers, 'max_updates': max_updates,
        'learning_rate': learning_rate, 'betas': betas, 'weight_decay': weight_decay,
        'dropout_p': dropout_p, 'l_threshold': l_threshold,
        'label_smoothing_alpha': label_smoothing_alpha, 'activation': activation,
        'out_dim': out_dim
    }
    
    # Prepare all training tasks
    training_tasks = [
        ('original', 'Original Data (Baseline)', X_original, y_original, None),
    ]
    
    dr_methods = ['pca', 'ica', 'rp']
    for dr_method in dr_methods:
        X_dr = dr_results[dr_method]['X_transformed']
        n_components = dr_results[dr_method]['n_components']
        training_tasks.append((
            dr_method, 
            f"{dr_method.upper()}-transformed Data", 
            X_dr, 
            y_original, 
            n_components
        ))
    
    # Train all methods sequentially (Parallel had issues with shared state)
    print("\n" + "="*80)
    print("STEP 4: Training Neural Networks")
    print("="*80)
    
    for method_key, display_name, X_data, y_data, n_comp in training_tasks:
        print(f"\nTraining {method_key.upper()}")
        
        ml_logger.log_metric(f'{method_key}_training_start', time.perf_counter())
        
        result = _train_nn_on_data(X=X_data, y=y_data, dataset_name=display_name, 
                                  ml_logger=ml_logger, **train_params)
        
        if n_comp is not None:
            result['n_components'] = n_comp
        
        log_training_curves(ml_logger, method_key, result['curves'], result['final_train_loss'], result['test_loss'], n_comp) #type:ignore
        results[method_key] = result
        print_result_summary(display_name, result['test_loss'], result['final_train_loss'], result['wall_time'], n_comp)
        print(f"{method_key.upper()} training completed")
    
    print("="*80)

    # 3. Summary and Visualization
    _print_performance_comparison(results, dr_methods)
    
    print("\n" + "="*80)
    print("STEP 4: Generating Learning Curve Visualizations")
    print("="*80)
    _plot_learning_curves(results, dr_methods, dataset, save_path, eval_interval=100)
    
    total_time = time.perf_counter() - start_time
    print(f"\nStep 4 completed in {total_time:.2f}s")
    print("="*80 + "\n")
    
    # 4. Log Final Results
    hyperparameters = {
        'batch_size': batch_size, 'max_updates': max_updates,
        'learning_rate': learning_rate, 'weight_decay': weight_decay,
        'hidden_layers': hidden_layers, 'activation': activation,
        'dropout': dropout_p, 'optimizer': optimizer, 'betas': betas, 'seed': seed
    }
    
    _log_step_results(ml_logger, "Step 4: Neural Networks on DR Data", results, hyperparameters, dataset)
    ml_logger.generate_log_report(output_file=f"{save_path}/execution_report.txt")
    
    # Add total time to results
    results['total_time'] = total_time
    
    # Save to cache
    cache_manager.save(dataset, 'neural_networks_on_dr', cache_params, results)
    
    return results