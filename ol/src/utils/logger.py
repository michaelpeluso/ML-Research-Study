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

    def generate_log_report(self, output_file, part:int=0):
        """generate a detailed report from current logs, with analysis summary at top aggregating all data."""
        if not output_file:
            output_file = os.path.join(os.environ['ROOT'], "logs/experiment_report.txt")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        if not self.current_logs:
            print("No logs available for this run.")
            return

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("ML EXPERIMENT DETAILED REPORT\n")
            f.write("=" * 70 + "\n\n")

            # Summary for Part 1
            if part == 1:
                f.write("RO ALGORITHMS SUMMARY TABLE\n")
                f.write("-" * 40 + "\n")
                ro_data = []
                for log in self.current_logs:
                    if log['step'] in ["Randomized Hill Climbing", "Simulated Annealing", "Genetic Algorithm"]:
                        info = log['step_info']
                        algo = log['step']
                        best_loss = info.get('best_loss', 'N/A')
                        evals = info.get('evals', 'N/A')
                        wall_time = info.get('wall_clock_time', 'N/A')
                        ro_data.append((algo, best_loss, evals, wall_time))

                if ro_data:
                    # Calculate dynamic column widths
                    max_algo = max(len(algo) for algo, _, _, _ in ro_data)
                    max_loss = max(len(f"{loss:.4f}" if isinstance(loss, (int, float)) else str(loss)) for _, loss, _, _ in ro_data)
                    max_evals = max(len(str(evals)) for _, _, evals, _ in ro_data)
                    max_time = max(len(f"{time:.2f}s" if isinstance(time, (int, float)) else str(time)) for _, _, _, time in ro_data)

                    table_str = f"| {'Algorithm':<{max_algo}} | {'Best Val Loss':>{max_loss}} | {'# Func Evals':>{max_evals}} | {'Wall Time':>{max_time}} |\n"
                    table_str += f"|{'-' * (max_algo + 2)}|{'-' * (max_loss + 2)}|{'-' * (max_evals + 2)}|{'-' * (max_time + 2)}|\n"
                    for algo, loss, evals, time in ro_data:
                        loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else str(loss)
                        time_str = f"{time:.2f}s" if isinstance(time, (int, float)) else str(time)
                        table_str += f"| {algo:<{max_algo}} | {loss_str:>{max_loss}} | {evals:>{max_evals}} | {time_str:>{max_time}} |\n"
                    f.write(table_str + "\n\n")
                else:
                    f.write("No RO data available.\n\n")

            # Optimizer Ablations Summary for Part 2
            if part == 2:
                f.write("OPTIMIZER ABLATIONS SUMMARY TABLE\n")
                f.write("-" * 40 + "\n")
                table_str = self.metrics.get('part2_table', 'No table available')
                f.write(table_str + "\n\n")

            # Key Metrics
            f.write("KEY METRICS\n")
            f.write("-" * 40 + "\n")
            key_metrics = {k: v for k, v in self.metrics.items() if any(sub in k for sub in ['best_loss', 'avg_steps_to_l', 'avg_test_loss', 'test_metric'])}
            for k, v in key_metrics.items():
                f.write(f"{k.replace('_', ' ').title()}: {v}\n")
            f.write("\n")

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

            # data analysis
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
                f.write(f"Memory Reduction: {step_info.get('memory_reduction', 'N/A')}\n")  # changed: added to detailed
                if step_info.get('target_distribution'):
                    f.write("Target Distribution:\n")
                    total = sum(step_info['target_distribution'].values())
                    for class_name, count in step_info['target_distribution'].items():
                        f.write(f"  {class_name}: {count:,} ({count/total*100:.1f}%)\n")
                f.write("\n")

            # detailed step analysis
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
                            if isinstance(v, (int)):
                                f.write(f"     {k.replace('_', ' ').title()}: {v}\n")
                            if isinstance(v, (float)):
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

            # footer
            f.write("=" * 70 + "\n")
            f.write(f"Report Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EDT\n")
            f.write(f"Log File: {self.log_file}\n")
            f.write("=" * 70 + "\n")

        print(f"Detailed report saved to: {output_file}")

# global instance
ml_logger = MLLogger()