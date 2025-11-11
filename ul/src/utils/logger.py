import os
import json
import time
import datetime
import psutil
import sys
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

class MLLogger:
    """a logger for machine learning experiments, tracking performance, system info, and step details."""

    def __init__(self, log_file: str = ""):
        """initialize logger with file path and system information."""
        self.log_file = log_file or os.path.join(os.environ['ROOT'], "logs")
        self.experiment_context: Dict[str, Any] = {}
        self.current_logs: List[Dict[str, Any]] = []  # logs for the current experiment run
        self.metrics: Dict[str, Any] = {}
        self.log_interval = 100
        if not log_file: log_file = os.path.join(os.environ['ROOT'], "logs", "experiment_logs.json")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # capture system information
        self.experiment_context['system_info'] = {
            'platform': sys.platform,
            'python_version': sys.version.split()[0],
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': round(psutil.virtual_memory().total / (1024 ** 3), 2),
            'process_id': os.getpid()
        }

    def set_experiment_context(self, **context):
        """update experiment context with provided key-value pairs."""
        self.experiment_context.update(context)

    @contextmanager
    def log_step(self, step_name: str, **initial_info):
        """context manager to log a step with performance metrics."""
        start_time = time.time()
        start_mem = psutil.Process().memory_info().rss / (1024 ** 2)  # convert to mb
        step_info = dict(initial_info)
        error_msg = None
        status = "unknown"

        try:
            yield step_info  # allow updating step_info within the block
            status = "success"
        except Exception as e:
            status = "error"
            error_msg = str(e)
            raise
        finally:
            duration = time.time() - start_time
            end_mem = psutil.Process().memory_info().rss / (1024 ** 2)
            mem_diff = end_mem - start_mem

            log_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "step": step_name,
                "experiment_context": self.experiment_context.copy(),
                "performance": {
                    "duration_seconds": round(duration, 4),
                    "memory_diff_mb": round(mem_diff, 2),
                    "peak_memory_mb": round(end_mem, 2)
                },
                "status": status,
                "step_info": step_info
            }
            if error_msg:
                log_entry["error"] = error_msg

            self.current_logs.append(log_entry)
            self._write_log(log_entry)

    def _write_log(self, log_entry):
        """write a single log entry to the json file."""
        try:
            logs = []
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            logs.append(log_entry)
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2, default=str, ensure_ascii=False)
        except Exception:
            pass  # silent fail on write issues to avoid interrupting experiment

    def log_metric(self, metric_name: str, value: Any):
        """log a single metric to the current logs and metrics dict."""
        if not self.current_logs:
            self.current_logs.append({"step_info": {}})
        self.current_logs[-1]["step_info"][metric_name] = value
        self.metrics[metric_name] = value  # new: store for summary aggregation

    def log_learning_curve(self, curve_data: dict):
        """log learning curve data to the current logs."""
        if not self.current_logs:
            self.current_logs.append({"step_info": {}})
        self.current_logs[-1]["step_info"]["learning_curve_points"] = curve_data

    def _format_step_data(self, f, data, indent=0):
        """Recursively format and write step data with proper indentation."""
        indent_str = "  " * indent
        
        for key, value in data.items():
            # Handle both string and non-string keys
            if isinstance(key, str):
                formatted_key = key.replace('_', ' ').title()
            else:
                formatted_key = str(key)
            
            if value is None:
                continue
            elif isinstance(value, dict):
                f.write(f"{indent_str}{formatted_key}:\n")
                self._format_step_data(f, value, indent + 1)
            elif isinstance(value, list):
                if not value:
                    f.write(f"{indent_str}{formatted_key}: []\n")
                elif len(value) <= 10 and all(isinstance(x, (int, float, str)) for x in value):
                    # Short list of primitives - show inline
                    f.write(f"{indent_str}{formatted_key}: {value}\n")
                else:
                    # Long list or complex items
                    f.write(f"{indent_str}{formatted_key}: [{len(value)} items]\n")
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            f.write(f"{indent_str}  [{i}]:\n")
                            self._format_step_data(f, item, indent + 2)
                        else:
                            f.write(f"{indent_str}  [{i}]: {item}\n")
            elif isinstance(value, tuple):
                f.write(f"{indent_str}{formatted_key}: {value}\n")
            elif isinstance(value, float):
                f.write(f"{indent_str}{formatted_key}: {value:.4f}\n")
            elif isinstance(value, int):
                f.write(f"{indent_str}{formatted_key}: {value:,}\n")
            else:
                f.write(f"{indent_str}{formatted_key}: {value}\n")

    def generate_log_report(self, output_file, part:int=0, start_index:int=0):
        """generate a detailed report from current logs, with analysis summary at top aggregating all data."""
        if not output_file:
            output_file = os.path.join(os.environ['ROOT'], "logs/experiment_report.txt")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        logs_to_report = self.current_logs[start_index:] if start_index > 0 else self.current_logs
        if not logs_to_report:
            print("No logs available for this run.")
            return

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("ML EXPERIMENT DETAILED REPORT\n")
            f.write("=" * 70 + "\n")
            f.write(f"Total Duration: {self.metrics.get('total_duration', 'N/A')}\n\n")

            # system information
            sys_info = self.experiment_context.get('system_info', {})
            f.write("SYSTEM INFORMATION\n")
            f.write("-" * 30 + "\n")
            f.write(f"Platform: {sys_info.get('platform', 'Unknown')}\n")
            f.write(f"Python Version: {sys_info.get('python_version', 'Unknown')}\n")
            f.write(f"CPU Cores: {sys_info.get('cpu_count', 'Unknown')}\n")
            f.write(f"Total Memory: {sys_info.get('memory_total_gb', 0):.2f} GB\n")
            f.write(f"Process ID: {sys_info.get('process_id', 'Unknown')}\n")
            f.write("\n")

            # experiment snapshot
            f.write("EXPERIMENT CONFIGURATION\n")
            f.write("-" * 40 + "\n")
            ctx = self.experiment_context
            f.write(f"Dataset: {ctx.get('dataset', 'N/A')}\n")
            f.write(f"Target: {ctx.get('target', 'N/A')}\n")
            f.write(f"Method: {ctx.get('method', 'N/A')}\n")
            f.write(f"Subsample: {ctx.get('subsample', 'N/A')}\n")
            f.write("\n")

            # all steps - uniform formatting
            for i, log in enumerate(logs_to_report, 1):
                if 'step' not in log:
                    continue
                    
                f.write("=" * 70 + "\n")
                f.write(f"STEP {i}: {log['step'].upper()}\n")
                f.write("=" * 70 + "\n")
                f.write(f"Status: {log['status'].upper()}\n")
                f.write(f"Timestamp: {log['timestamp']}\n")
                f.write(f"Duration: {log['performance']['duration_seconds']:.4f} seconds\n")
                f.write(f"Memory Change: {log['performance']['memory_diff_mb']:+.2f} MB\n")
                f.write(f"Peak Memory: {log['performance']['peak_memory_mb']:.2f} MB\n")
                f.write(f"--\n")
                
                
                if log.get('error'):
                    f.write(f"Error: {log['error']}\n")
                
                f.write("\n")
                
                # print all step data
                if log.get('step_info'):
                    self._format_step_data(f, log['step_info'], indent=0)
                
                f.write("\n")

            # footer
            f.write("=" * 70 + "\n")
            f.write(f"Report Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EDT\n")
            f.write(f"Log File: {self.log_file}\n")
            f.write("=" * 70 + "\n")

        print_t(f"Detailed report saved to: {output_file}")


# print overload to add time since program start and since last print
_program_start_time = time.perf_counter()
def print_t(*args):
    time_str = f"{time.perf_counter() - _program_start_time:.2f}s"
    print(f"{time_str:<10}| ", *args)


def generate_ul_summary(all_results):
    """Generate comprehensive summary for Unsupervised Learning report."""
    import pandas as pd
    
    print("\n" + "="*80)
    print("UNSUPERVISED LEARNING EXPERIMENT SUMMARY")
    print("="*80 + "\n")
    
    for result in all_results:
        dataset = result['dataset']
        clustering_results = result['clustering_results']
        dr_results = result['dr_results']
        clustering_dr_results = result['clustering_dr_results']
        save_path = result['save_path']
        
        print("\n" + "-"*80)
        print(f"DATASET: {dataset.upper()}")
        print("-"*80)
        
        # Step 1: Clustering on Original Data
        print("\n[STEP 1] Clustering on Original Data")
        print("-" * 40)
        
        kmeans_original = clustering_results['kmeans']
        gmm_original = clustering_results['em']
        
        print(f"K-Means:")
        print(f"  Best k: {kmeans_original['chosen_n']}")
        print(f"  Silhouette: {kmeans_original['best_result']['silhouette_score']:.3f}")
        print(f"  Dunn Index: {kmeans_original['best_result']['dunn_index']:.3f}")
        print(f"  Calinski-Harabasz: {kmeans_original['best_result']['calinski_harabasz_score']:.1f}")
        print(f"  Davies-Bouldin: {kmeans_original['best_result']['davies_bouldin_score']:.3f}")
        
        # Handle stability - it might be None or missing
        stability = kmeans_original.get('stability') or {}
        if stability and stability.get('stability_score') is not None:
            print(f"  Stability (ARI): {stability['stability_score']:.3f} ± {stability.get('stability_std', 0):.3f}")
        else:
            print(f"  Stability (ARI): N/A")
        
        print(f"\nGMM:")
        print(f"  Best n_components: {gmm_original['chosen_n']}")
        print(f"  Silhouette: {gmm_original['best_result']['silhouette_score']:.3f}")
        print(f"  Dunn Index: {gmm_original['best_result']['dunn_index']:.3f}")
        print(f"  BIC: {gmm_original['best_result']['bic']:.1f}")
        print(f"  AIC: {gmm_original['best_result']['aic']:.1f}")
        print(f"  Log-Likelihood: {gmm_original['best_result']['log_likelihood']:.1f}")
        
        # Handle stability - it might be None or missing
        stability = gmm_original.get('stability') or {}
        if stability and stability.get('stability_score') is not None:
            print(f"  Stability (ARI): {stability['stability_score']:.3f} ± {stability.get('stability_std', 0):.3f}")
        else:
            print(f"  Stability (ARI): N/A")
        
        # Step 2: Dimensionality Reduction
        print("\n[STEP 2] Dimensionality Reduction")
        print("-" * 40)
        
        for dr_method in ['pca', 'ica', 'rp']:
            dr_result = dr_results[dr_method]
            print(f"\n{dr_method.upper()}:")
            print(f"  Best n_components: {dr_result['n_components']}")
            print(f"  Selection Score: {dr_result['best_result']['mean_score']:.4f}")
            if dr_result['best_result']['reconstruction_error']:
                print(f"  Reconstruction Error: {dr_result['best_result']['reconstruction_error']:.4f}")
            if dr_result['best_result']['cumulative_variance']:
                print(f"  Cumulative Variance: {dr_result['best_result']['cumulative_variance']:.4f}")
            print(f"  Execution Time: {dr_result['total_time']:.2f}s")
        
        # Step 3: Clustering on DR-transformed Data
        print("\n[STEP 3] Clustering on DR-transformed Data")
        print("-" * 40)
        
        for dr_method in ['pca', 'ica', 'rp']:
            print(f"\n{dr_method.upper()} + Clustering:")
            kmeans_dr = clustering_dr_results[dr_method]['kmeans']
            gmm_dr = clustering_dr_results[dr_method]['em']
            
            print(f"  K-Means:")
            print(f"    Best k: {kmeans_dr['chosen_n']}")
            print(f"    Silhouette: {kmeans_dr['best_result']['silhouette_score']:.3f}")
            print(f"    Dunn Index: {kmeans_dr['best_result']['dunn_index']:.3f}")
            
            # Handle stability - it might be None or missing
            stability = kmeans_dr.get('stability') or {}
            if stability and stability.get('stability_score') is not None:
                print(f"    Stability (ARI): {stability['stability_score']:.3f} ± {stability.get('stability_std', 0):.3f}")
            else:
                print(f"    Stability (ARI): N/A")
            
            print(f"  GMM:")
            print(f"    Best n_components: {gmm_dr['chosen_n']}")
            print(f"    Silhouette: {gmm_dr['best_result']['silhouette_score']:.3f}")
            print(f"    Dunn Index: {gmm_dr['best_result']['dunn_index']:.3f}")
            print(f"    BIC: {gmm_dr['best_result']['bic']:.1f}")
            
            # Handle stability - it might be None or missing
            stability = gmm_dr.get('stability') or {}
            if stability and stability.get('stability_score') is not None:
                print(f"    Stability (ARI): {stability['stability_score']:.3f} ± {stability.get('stability_std', 0):.3f}")
            else:
                print(f"    Stability (ARI): N/A")
        
        # Step 4 & 5: Neural Networks (only for accidents dataset)
        nn_original_results = result.get('step_4a_nn_original')
        nn_reduced_results = result.get('step_4b_nn_reduced')
        nn_cluster_results = result.get('step_5_nn_with_clusters')
        
        if nn_original_results is not None or nn_reduced_results is not None:
            print("\n[STEP 4] Neural Networks on Original + DR Data")
            print("-" * 50)
            
            # Original baseline (from Step 4a)
            if nn_original_results is not None and 'original' in nn_original_results:
                orig_nn = nn_original_results['original']
                print(f"Original (Baseline):")
                print(f"  Test Loss: {orig_nn['test_loss']:.4f}")
                print(f"  Train Loss: {orig_nn['final_train_loss']:.4f}")
                print(f"  Wall Time: {orig_nn['wall_time']:.2f}s")
                print(f"  Parameters: {orig_nn['n_params']:,}")
                baseline_loss = orig_nn['test_loss']
            elif nn_reduced_results is not None:
                # If no original, use first DR method as baseline for comparison
                first_dr_method = list(nn_reduced_results.keys())[0]
                baseline_loss = nn_reduced_results[first_dr_method]['test_loss']
                print(f"Using {first_dr_method.upper()} as baseline (no original data available)")
            else:
                baseline_loss = None
            
            # DR methods (from Step 4b)
            if nn_reduced_results is not None:
                for dr_method in ['pca', 'ica', 'rp']:
                    if dr_method in nn_reduced_results:
                        dr_nn = nn_reduced_results[dr_method]
                        if baseline_loss is not None:
                            improvement = ((baseline_loss - dr_nn['test_loss']) / baseline_loss) * 100
                            improvement_str = f" ({improvement:+.2f}%)"
                        else:
                            improvement_str = ""
                        print(f"\n{dr_method.upper()} (n={dr_nn.get('n_components', 'N/A')}):")
                        print(f"  Test Loss: {dr_nn['test_loss']:.4f}{improvement_str}")
                        print(f"  Train Loss: {dr_nn['final_train_loss']:.4f}")
                        print(f"  Wall Time: {dr_nn['wall_time']:.2f}s")
                        print(f"  Parameters: {dr_nn['n_params']:,}")
        
        if nn_cluster_results is not None:
            print("\n[STEP 5] Neural Networks with Cluster Features")
            print("-" * 50)
            
            # Baseline
            baseline_nn = nn_cluster_results['baseline']
            print(f"Baseline (Original Features):")
            print(f"  Test Loss: {baseline_nn['test_loss']:.4f}")
            print(f"  Train Loss: {baseline_nn['final_train_loss']:.4f}")
            print(f"  Wall Time: {baseline_nn['wall_time']:.2f}s")
            
            # K-Means features
            kmeans_key = 'kmeans_tuned' if 'kmeans_tuned' in nn_cluster_results else 'kmeans'
            kmeans_nn = nn_cluster_results[kmeans_key]
            improvement = ((baseline_nn['test_loss'] - kmeans_nn['test_loss']) / baseline_nn['test_loss']) * 100
            print(f"\nK-Means Cluster Features:")
            print(f"  Test Loss: {kmeans_nn['test_loss']:.4f} ({improvement:+.2f}%)")
            print(f"  Train Loss: {kmeans_nn['final_train_loss']:.4f}")
            print(f"  Wall Time: {kmeans_nn['wall_time']:.2f}s")
            print(f"  Input Dim: {kmeans_nn['input_dim']}")
            if kmeans_nn.get('n_cluster_features'):
                print(f"  Cluster Features: {kmeans_nn['n_cluster_features']}")
            
            # EM/GMM features
            em_key = 'em_tuned' if 'em_tuned' in nn_cluster_results else 'em'
            em_nn = nn_cluster_results[em_key]
            improvement = ((baseline_nn['test_loss'] - em_nn['test_loss']) / baseline_nn['test_loss']) * 100
            print(f"\nEM/GMM Cluster Features:")
            print(f"  Test Loss: {em_nn['test_loss']:.4f} ({improvement:+.2f}%)")
            print(f"  Train Loss: {em_nn['final_train_loss']:.4f}")
            print(f"  Wall Time: {em_nn['wall_time']:.2f}s")
            print(f"  Input Dim: {em_nn['input_dim']}")
            if em_nn.get('n_cluster_features'):
                print(f"  Cluster Features: {em_nn['n_cluster_features']}")
        
        # Generate summary CSV (update to include NN results)
        summary_data = []
        
        # Original clustering
        stability = kmeans_original.get('stability') or {}
        summary_data.append({
            'Dataset': dataset,
            'Step': 'Step 1',
            'Method': 'K-Means (Original)',
            'n_components/k': kmeans_original['chosen_n'] or None,
            'Silhouette': kmeans_original['best_result']['silhouette_score'] or None,
            'Dunn': kmeans_original['best_result']['dunn_index'] or None,
            'CH_Score': kmeans_original['best_result']['calinski_harabasz_score'] or None,
            'DB_Score': kmeans_original['best_result']['davies_bouldin_score'] or None,
            'Stability_ARI': stability.get('stability_score') if stability else None
        })
        
        stability = gmm_original.get('stability') or {}
        summary_data.append({
            'Dataset': dataset,
            'Step': 'Step 1',
            'Method': 'GMM (Original)',
            'n_components/k': gmm_original['chosen_n'],
            'Silhouette': gmm_original['best_result']['silhouette_score'],
            'Dunn': gmm_original['best_result']['dunn_index'],
            'CH_Score': gmm_original['best_result']['calinski_harabasz_score'],
            'DB_Score': gmm_original['best_result']['davies_bouldin_score'],
            'Stability_ARI': stability.get('stability_score') if stability else None
        })
        
        # DR + Clustering
        for dr_method in ['pca', 'ica', 'rp']:
            kmeans_dr = clustering_dr_results[dr_method]['kmeans']
            gmm_dr = clustering_dr_results[dr_method]['em']
            
            stability = kmeans_dr.get('stability') or {}
            summary_data.append({
                'Dataset': dataset,
                'Step': 'Step 3',
                'Method': f'K-Means ({dr_method.upper()})',
                'n_components/k': kmeans_dr['chosen_n'],
                'Silhouette': kmeans_dr['best_result']['silhouette_score'],
                'Dunn': kmeans_dr['best_result']['dunn_index'],
                'CH_Score': kmeans_dr['best_result']['calinski_harabasz_score'],
                'DB_Score': kmeans_dr['best_result']['davies_bouldin_score'],
                'Stability_ARI': stability.get('stability_score') if stability else None
            })
            
            stability = gmm_dr.get('stability') or {}
            summary_data.append({
                'Dataset': dataset,
                'Step': 'Step 3',
                'Method': f'GMM ({dr_method.upper()})',
                'n_components/k': gmm_dr['chosen_n'],
                'Silhouette': gmm_dr['best_result']['silhouette_score'],
                'Dunn': gmm_dr['best_result']['dunn_index'],
                'CH_Score': gmm_dr['best_result']['calinski_harabasz_score'],
                'DB_Score': gmm_dr['best_result']['davies_bouldin_score'],
                'Stability_ARI': stability.get('stability_score') if stability else None
            })
        
        # Add NN results to summary CSV
        if nn_original_results is not None or nn_reduced_results is not None:
            # Step 4: NN on original and DR data
            if nn_original_results is not None and 'original' in nn_original_results:
                nn_result = nn_original_results['original']
                summary_data.append({
                    'Dataset': dataset,
                    'Step': 'Step 4',
                    'Method': 'NN (Original)',
                    'n_components/k': '',
                    'Test_Loss': nn_result['test_loss'],
                    'Train_Loss': nn_result['final_train_loss'],
                    'Wall_Time': nn_result['wall_time'],
                    'N_Params': nn_result['n_params']
                })
            
            if nn_reduced_results is not None:
                for method_key in ['pca', 'ica', 'rp']:
                    if method_key in nn_reduced_results:
                        nn_result = nn_reduced_results[method_key]
                        summary_data.append({
                            'Dataset': dataset,
                            'Step': 'Step 4',
                            'Method': f'NN ({method_key.upper()})',
                            'n_components/k': nn_result.get('n_components', ''),
                            'Test_Loss': nn_result['test_loss'],
                            'Train_Loss': nn_result['final_train_loss'],
                            'Wall_Time': nn_result['wall_time'],
                            'N_Params': nn_result['n_params']
                        })
        
        if nn_cluster_results is not None:
            # Step 5: NN with cluster features
            for method_key in ['baseline', 'kmeans', 'em']:
                if method_key in nn_cluster_results:
                    nn_result = nn_cluster_results[method_key]
                    method_name = {
                        'baseline': 'NN (Baseline)',
                        'kmeans': 'NN (K-Means Features)',
                        'em': 'NN (EM/GMM Features)'
                    }[method_key]
                    summary_data.append({
                        'Dataset': dataset,
                        'Step': 'Step 5',
                        'Method': method_name,
                        'n_components/k': '',
                        'Test_Loss': nn_result['test_loss'],
                        'Train_Loss': nn_result['final_train_loss'],
                        'Wall_Time': nn_result['wall_time'],
                        'Input_Dim': nn_result['input_dim'],
                        'Cluster_Features': nn_result.get('n_cluster_features', '')
                    })
        
        df = pd.DataFrame(summary_data)
        summary_path = os.path.join(save_path, "ul_report_summary.csv")
        os.makedirs(save_path, exist_ok=True)  # Ensure directory exists
        df.to_csv(summary_path, index=False)
        print(f"\n\nSummary table saved to: {summary_path}")
        print(f"\n{df.to_string(index=False)}")
    
    print("\n" + "="*80)
    print("END OF SUMMARY")
    print("="*80 + "\n")
