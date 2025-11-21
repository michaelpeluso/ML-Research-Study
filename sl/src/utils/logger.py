# Updated logger.py with robust handling for target_distribution and other potential None values

import time
import datetime
import json
import os
import psutil
import sys
from contextlib import contextmanager

class MLLogger:
    def __init__(self, log_file="logs/experiment_logs.json"):
        self.log_file = log_file
        self.experiment_context = {}
        self.current_logs = []  # store logs for THIS execution
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Add system_info here
        self.experiment_context['system_info'] = {
            'platform': sys.platform,
            'python_version': sys.version.split()[0],
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': round(psutil.virtual_memory().total / (1024 ** 3), 2),
            'process_id': os.getpid()
        }

    def set_experiment_context(self, **context):
        self.experiment_context.update(context)

    @contextmanager
    def log_step(self, step_name: str, **initial_info):
        start_time = time.time()
        start_mem = psutil.Process().memory_info().rss / (1024*1024)
        step_info = dict(initial_info)  # Start with any initial info
        error_msg = None
        try:
            yield step_info  # Yield the step_info dict for user to update inside the block
            status = "success"
        except Exception as e:
            status = "error"
            error_msg = str(e)
            raise
        finally:
            duration = time.time() - start_time
            end_mem = psutil.Process().memory_info().rss / (1024*1024)
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
                "step_info": step_info  # Use the updated step_info
            }
            if error_msg:
                log_entry["error"] = error_msg
            
            # Save to session logs
            self.current_logs.append(log_entry)
            
            # Also write to the persistent JSON if you want
            self._write_log(log_entry)

    def _write_log(self, log_entry):
        try:
            logs = []
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            logs.append(log_entry)
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2, default=str, ensure_ascii=False)
        except Exception as e:
            #print(f"Failed to write log: {e}")
            pass
    
    def generate_log_report(self, output_file="logs/experiment_report.txt"):
        logs = self.current_logs 
        
        if not logs:
            print("No logs for this run.")
            return
        
        if not os.path.exists(self.log_file):
            return

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("ML EXPERIMENT DETAILED REPORT\n")
            f.write("="*70 + "\n\n")
            
            # Experiment Overview
            if logs and 'experiment_context' in logs[0]:
                f.write("EXPERIMENT CONFIGURATION\n")
                f.write("-"*30 + "\n")
                ctx = logs[0]['experiment_context']
                f.write(f"Dataset: {ctx.get('dataset', 'Unknown')}\n")
                f.write(f"Target Variable: {ctx.get('target', 'Unknown')}\n")
                f.write(f"Learning Model: {ctx.get('model', 'Unknown').upper()}\n")
                f.write(f"Learning Method: {ctx.get('method', 'Unknown').title()}\n")
                f.write(f"Data Subsample: {ctx.get('subsample', 1.0)} ({ctx.get('subsample', 1.0)*100:.1f}%)\n")
                f.write(f"Random Seed: {ctx.get('seed', 'Unknown')}\n")
                f.write(f"Hyperparameter Tuning: {'on' if ctx.get('tuning') else 'off'}\n")
                f.write("\n")
            
            # Performance Summary
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-"*30 + "\n")
            
            step_logs = [log for log in logs if 'performance' in log]
            if step_logs:
                total_time = sum(log['performance'].get('duration_seconds', 0) for log in step_logs)
                total_memory = sum(log['performance'].get('memory_diff_mb', 0) for log in step_logs)
                max_memory = max(log['performance'].get('peak_memory_mb', 0) for log in step_logs)
                
                f.write(f"Total Runtime: {total_time:.3f} seconds ({total_time/60:.2f} minutes)\n")
                f.write(f"Total Memory Delta: {total_memory:+.1f} MB\n")
                f.write(f"Peak Memory Usage: {max_memory:.1f} MB\n")
                f.write(f"Number of Steps: {len(step_logs)}\n")
                
                f.write(f"Average Step Time: {total_time/len(step_logs):.3f} seconds\n")
                slowest_step = max(step_logs, key=lambda x: x['performance']['duration_seconds'])
                f.write(f"Slowest Step: {slowest_step['step']} ({slowest_step['performance']['duration_seconds']:.3f}s)\n")
                fastest_step = min(step_logs, key=lambda x: x['performance']['duration_seconds'])
                f.write(f"Fastest Step: {fastest_step['step']} ({fastest_step['performance']['duration_seconds']:.3f}s)\n")
                f.write("\n")

            # Data Analysis
            data_log = next((log for log in logs if log['step'] == 'Load Data' and 'step_info' in log), None)
            if data_log and data_log['step_info']:
                f.write("DATA ANALYSIS\n")
                f.write("-"*30 + "\n")
                step_info = data_log['step_info']
                
                if 'n_rows_initial' in step_info:
                    initial_rows = step_info['n_rows_initial'] if step_info['n_rows_initial'] is not None else 0
                    f.write(f"Initial Rows: {initial_rows}\n")
                if 'n_rows_cleaned' in step_info:
                    f.write(f"Rows After Cleaning: {step_info['n_rows_cleaned']:,}\n")
                if 'train_shape' in step_info:
                    train_samples, train_features = step_info['train_shape']
                    test_samples, test_features = step_info.get('test_shape', (0, 0))
                    f.write(f"Training Set: {train_samples:,} samples x {train_features} features\n")
                    f.write(f"Test Set: {test_samples:,} samples x {test_features} features\n")
                    f.write(f"Train/Test Split: {train_samples/(train_samples+test_samples)*100:.1f}% / {test_samples/(train_samples+test_samples)*100:.1f}%\n")
                
                if 'memory_usage_mb' in step_info:
                    f.write(f"Dataset Memory Usage: {step_info['memory_usage_mb']:.2f} MB\n")
                
                if 'target_distribution' in step_info and step_info['target_distribution'] is not None:
                    f.write("Target Variable Distribution:\n")
                    for class_name, count in step_info['target_distribution'].items():
                        percentage = count / sum(step_info['target_distribution'].values()) * 100
                        f.write(f"  {class_name}: {count:,} ({percentage:.1f}%)\n")
                f.write("\n")
            
            # Model Configuration
            pipeline_log = next((log for log in logs if log['step'] == 'Pipeline Building' and 'step_info' in log), None)
            if pipeline_log and pipeline_log['step_info']:
                f.write("MODEL CONFIGURATION\n")
                f.write("-"*30 + "\n")
                step_info = pipeline_log['step_info']
                
                if 'features' in step_info and 'samples' in step_info:
                    f.write(f"Input Dimensions: {step_info['samples']:,} samples x {step_info['features']} features\n")
                
                feature_counts = {}
                for key in ['numeric_features', 'categorical_features', 'high_cardinality_features']:
                    if key in step_info:
                        feature_counts[key.replace('_features', '')] = step_info[key]
                
                if feature_counts:
                    f.write("Feature Type Breakdown:\n")
                    for feat_type, count in feature_counts.items():
                        f.write(f"  {feat_type.title()}: {count}\n")
                
                if 'model_type' in step_info:
                    f.write(f"Model Type: {step_info['model_type']}\n")
                f.write("\n")
            
            # Model Performance Results
            eval_log = next((log for log in logs if log['step'] == 'Evaluate Model' and 'step_info' in log), None)
            if eval_log and 'step_info' in eval_log:
                f.write("MODEL PERFORMANCE RESULTS\n")
                f.write("-"*30 + "\n")
                step_info = eval_log['step_info']
                
                if 'test_metrics' in step_info:
                    metrics = step_info['test_metrics']
                    f.write("Test Set Performance:\n")
                    for metric_name, metric_value in metrics.items():
                        if isinstance(metric_value, (int, float)):
                            f.write(f"  {metric_name}: {metric_value:.4f}\n")
                        else:
                            f.write(f"  {metric_name}: {metric_value}\n")
                
                if 'learning_curve_points' in step_info:
                    f.write(f"Learning Curve Data Points: {step_info['learning_curve_points']}\n")
                f.write("\n")
            
            # Hyperparameter Tuning Results
            tuning_log = next((log for log in logs if log['step'] == 'Hyperparameter Tuning' and 'step_info' in log), None)
            if tuning_log:
                f.write("HYPERPARAMETER TUNING RESULTS\n")
                f.write("-"*30 + "\n")
                step_info = tuning_log['step_info']
                
                if 'parameter_count' in step_info:
                    f.write(f"Parameters Tuned: {step_info['parameter_count']}\n")
                
                if 'grid_size' in step_info:
                    f.write(f"Total Combinations: {step_info['grid_size']:,}\n")
                
                if 'best_params' in step_info:
                    f.write("Best Parameters Found:\n")
                    best_params = step_info['best_params']
                    for param_name, param_value in best_params.items():
                        if param_name.startswith('estimator__'):
                            clean_name = param_name.replace('estimator__', '')
                            f.write(f"  {clean_name}: {param_value}\n")
                f.write("\n")
            
            # System Information
            sys_info = self.experiment_context.get('system_info', {})
            f.write("SYSTEM INFORMATION\n")
            f.write("-"*30 + "\n")
            f.write(f"Platform: {sys_info.get('platform', 'Unknown')}\n")
            f.write(f"Python Version: {sys_info.get('python_version', 'Unknown')}\n")
            f.write(f"CPU Cores: {sys_info.get('cpu_count', 'Unknown')}\n")
            f.write(f"Total Memory: {sys_info.get('memory_total_gb', 0):.2f} GB\n")
            f.write(f"Process ID: {sys_info.get('process_id', 'Unknown')}\n")
            f.write("\n")
            
            # Detailed Step Analysis
            f.write("DETAILED STEP ANALYSIS\n")
            f.write("-"*30 + "\n")
            for i, log in enumerate(logs, 1):
                f.write(f"{i}. {log['step'].upper()}\n")
                f.write(f"   Status: {log['status'].upper()}\n")
                f.write(f"   Timestamp: {log['timestamp']}\n")
                f.write(f"   Duration: {log['performance']['duration_seconds']:.4f} seconds\n")
                f.write(f"   Memory Change: {log['performance']['memory_diff_mb']:+.2f} MB\n")
                f.write(f"   Peak Memory: {log['performance']['peak_memory_mb']:.2f} MB\n")
                dur = log['performance']['duration_seconds']
                perf = "Very Fast" if dur < 0.1 else "Fast" if dur < 2 else "Moderate" if dur < 10 else "Slow"
                f.write(f"   Performance: {perf}\n")
                f.write("   Step Details:\n")
                if 'step_info' in log and log['step_info']:
                    for k, v in log['step_info'].items():
                        if v is not None:
                            if isinstance(v, int):
                                f.write(f"     {k}: {v:,}\n")
                            elif isinstance(v, float):
                                f.write(f"     {k}: {v:.4f}\n")
                            else:
                                f.write(f"     {k}: {v}\n")
                        else:
                            f.write(f"     {k}: N/A\n")
                else:
                    f.write("     N/A\n")
                f.write("\n")
            
            # Footer
            f.write("="*70 + "\n")
            f.write(f"Report Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Log File: {self.log_file}\n")
            f.write("="*70 + "\n")
        
        print(f"Detailed report saved: {output_file}")

# Global instance
ml_logger = MLLogger()