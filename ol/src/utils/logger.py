import os
import json
import time
import datetime
import psutil
import sys
from contextlib import contextmanager

class MLLogger:
    """A logger for machine learning experiments, tracking performance, system info, and step details."""

    def __init__(self, log_file="logs/experiment_logs.json"):
        """Initialize logger with file path and system information."""
        self.log_file = log_file
        self.experiment_context = {}
        self.current_logs = []  # Logs for the current experiment run
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Capture system information
        self.experiment_context['system_info'] = {
            'platform': sys.platform,
            'python_version': sys.version.split()[0],
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': round(psutil.virtual_memory().total / (1024 ** 3), 2),
            'process_id': os.getpid()
        }

    def set_experiment_context(self, **context):
        """Update experiment context with provided key-value pairs."""
        self.experiment_context.update(context)

    @contextmanager
    def log_step(self, step_name: str, **initial_info):
        """Context manager to log a step with performance metrics."""
        start_time = time.time()
        start_mem = psutil.Process().memory_info().rss / (1024 ** 2)  # Convert to MB
        step_info = dict(initial_info)
        error_msg = None
        status = "unknown"

        try:
            yield step_info  # Allow updating step_info within the block
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
        """Write a single log entry to the JSON file."""
        try:
            logs = []
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            logs.append(log_entry)
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2, default=str, ensure_ascii=False)
        except Exception:
            pass  # Silent fail on write issues to avoid interrupting experiment

    def log_metric(self, metric_name: str, value: float):
        """Log a single metric to the current logs."""
        if not self.current_logs:
            self.current_logs.append({"step_info": {}})
        self.current_logs[-1]["step_info"][metric_name] = value

    def log_learning_curve(self, curve_data: dict):
        """Log learning curve data to the current logs."""
        if not self.current_logs:
            self.current_logs.append({"step_info": {}})
        self.current_logs[-1]["step_info"]["learning_curve_points"] = curve_data

    def generate_log_report(self, output_file="logs/experiment_report.txt"):
        """Generate a detailed report from current logs."""
        if not self.current_logs:
            print("No logs available for this run.")
            return

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("ML EXPERIMENT DETAILED REPORT\n")
            f.write("=" * 70 + "\n\n")

            # Experiment Configuration
            ctx = self.current_logs[0]["experiment_context"]
            f.write("EXPERIMENT CONFIGURATION\n")
            f.write("-" * 30 + "\n")
            f.write(f"Dataset: {ctx.get('dataset', 'Unknown')}\n")
            f.write(f"Target Variable: {ctx.get('target', 'Unknown')}\n")
            f.write(f"Learning Model: {ctx.get('model', 'Unknown').upper()}\n")
            f.write(f"Learning Method: {ctx.get('method', 'Unknown').title()}\n")
            f.write(f"Data Subsample: {ctx.get('subsample', 1.0)} ({ctx.get('subsample', 1.0) * 100:.1f}%)\n")
            f.write(f"Random Seed: {ctx.get('seed', 'Unknown')}\n")
            f.write(f"Hyperparameter Tuning: {'Enabled' if ctx.get('tuning', False) else 'Disabled'}\n")
            f.write("\n")

            # Performance Summary
            step_logs = [log for log in self.current_logs if 'performance' in log]
            if step_logs:
                total_time = sum(log['performance']['duration_seconds'] for log in step_logs)
                total_mem_delta = sum(log['performance']['memory_diff_mb'] for log in step_logs)
                peak_mem = max(log['performance']['peak_memory_mb'] for log in step_logs)
                f.write("PERFORMANCE SUMMARY\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total Runtime: {total_time:.3f} seconds ({total_time / 60:.2f} minutes)\n")
                f.write(f"Total Memory Delta: {total_mem_delta:+.1f} MB\n")
                f.write(f"Peak Memory Usage: {peak_mem:.1f} MB\n")
                f.write(f"Number of Steps: {len(step_logs)}\n")
                f.write(f"Average Step Time: {total_time/len(step_logs):.3f} seconds\n")
                slowest_step = max(step_logs, key=lambda x: x['performance']['duration_seconds'])
                fastest_step = min(step_logs, key=lambda x: x['performance']['duration_seconds'])
                f.write(f"Slowest Step: {slowest_step['step']} ({slowest_step['performance']['duration_seconds']:.3f}s)\n")
                f.write(f"Fastest Step: {fastest_step['step']} ({fastest_step['performance']['duration_seconds']:.3f}s)\n")
                f.write("\n")

            # Data Analysis
            data_log = next((log for log in self.current_logs if log['step'] == 'Load Data' and 'step_info' in log), None)
            if data_log and data_log['step_info']:
                f.write("DATA ANALYSIS\n")
                f.write("-" * 30 + "\n")
                step_info = data_log['step_info']
                f.write(f"Initial Rows: {step_info.get('n_loaded_rows', 0):,}\n")
                f.write(f"Rows After Cleaning: {step_info.get('n_cleaned_rows', 0):,}\n")
                f.write(f"Training Set: {step_info.get('train_shape', '0 x 0')}\n")
                f.write(f"Validation Set: {step_info.get('validation_shape', '0 x 0')}\n")
                f.write(f"Test Set: {step_info.get('test_shape', '0 x 0')}\n")
                f.write(f"Memory Before Cleaning: {step_info.get('memory_before_clean', 'N/A')}\n")
                f.write(f"Memory After Cleaning: {step_info.get('memory_after_clean', 'N/A')}\n")
                if step_info.get('target_distribution'):
                    f.write("Target Distribution:\n")
                    total = sum(step_info['target_distribution'].values())
                    for class_name, count in step_info['target_distribution'].items():
                        f.write(f"  {class_name}: {count:,} ({count/total*100:.1f}%)\n")
                f.write("\n")

            # Model Performance Results
            eval_log = next((log for log in self.current_logs if log['step'] == 'Evaluate Model' and 'step_info' in log), None)
            if eval_log and eval_log['step_info']:
                f.write("MODEL PERFORMANCE RESULTS\n")
                f.write("-" * 30 + "\n")
                step_info = eval_log['step_info']
                f.write("Test Set Performance:\n")
                for metric, value in step_info.get('test_metrics', {}).items():
                    f.write(f"  {metric.replace('_', ' ').title()}: {value:.4f}\n")
                if 'learning_curve_points' in step_info:
                    f.write(f"Learning Curve Points: {len(step_info['learning_curve_points'])}\n")
                f.write("\n")

            # System Information
            sys_info = self.experiment_context.get('system_info', {})
            f.write("SYSTEM INFORMATION\n")
            f.write("-" * 30 + "\n")
            f.write(f"Platform: {sys_info.get('platform', 'Unknown')}\n")
            f.write(f"Python Version: {sys_info.get('python_version', 'Unknown')}\n")
            f.write(f"CPU Cores: {sys_info.get('cpu_count', 'Unknown')}\n")
            f.write(f"Total Memory: {sys_info.get('memory_total_gb', 0):.2f} GB\n")
            f.write(f"Process ID: {sys_info.get('process_id', 'Unknown')}\n")
            f.write("\n")

            # Detailed Step Analysis
            f.write("DETAILED STEP ANALYSIS\n")
            f.write("-" * 30 + "\n")
            for i, log in enumerate(self.current_logs, 1):
                f.write(f"{i}. {log['step'].upper()}\n")
                f.write(f"   Status: {log['status'].upper()}\n")
                f.write(f"   Timestamp: {log['timestamp']}\n")
                f.write(f"   Duration: {log['performance']['duration_seconds']:.4f} seconds\n")
                f.write(f"   Memory Change: {log['performance']['memory_diff_mb']:+.2f} MB\n")
                f.write(f"   Peak Memory: {log['performance']['peak_memory_mb']:.2f} MB\n")
                duration = log['performance']['duration_seconds']
                perf = "Very Fast" if duration < 0.1 else "Fast" if duration < 2 else "Moderate" if duration < 10 else "Slow"
                f.write(f"   Performance: {perf}\n")
                if log['step_info']:
                    f.write("   Step Details:\n")
                    for k, v in log['step_info'].items():
                        if v is not None:
                            if isinstance(v, (int, float)):
                                f.write(f"     {k.replace('_', ' ').title()}: {v:,.4f}\n")
                            elif isinstance(v, dict):
                                f.write(f"     {k.replace('_', ' ').title()}:\n")
                                for sub_k, sub_v in v.items():
                                    f.write(f"       {sub_k}: {sub_v}\n")
                            elif isinstance(v, list):
                                f.write(f"     {k.replace('_', ' ').title()}: {len(v)} items\n")
                            else:
                                f.write(f"     {k.replace('_', ' ').title()}: {v}\n")
                f.write("\n")

            # Footer
            f.write("=" * 70 + "\n")
            f.write(f"Report Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EDT\n")
            f.write(f"Log File: {self.log_file}\n")
            f.write("=" * 70 + "\n")

        print(f"Detailed report saved to: {output_file}")

# Global instance
ml_logger = MLLogger()