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

        print(f"Detailed report saved to: {output_file}")


# print overload to add time since program start and since last print
_program_start_time = time.perf_counter()
def print_t(*args):
    print(f"{time.perf_counter() - _program_start_time:.2f}s | ", *args)