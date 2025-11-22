# AI Use Statement: plotting utilities created with GitHub Copilot assistance
"""generate report-ready figures"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Optional

sns.set_theme(style="whitegrid")


def plot_learning_curve(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    output_path: Optional[Path] = None,
    title: str = "Learning Curve",
    xlabel: str = "Episode",
    ylabel: str = "Return"
) -> None:
    """plot learning curve from dataframe"""
    plt.figure(figsize=(10, 6))
    plt.plot(data[x_col], data[y_col], linewidth=1.5, alpha=0.8)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_convergence_comparison(
    results: dict,
    output_path: Optional[Path] = None,
    title: str = "Convergence Comparison"
) -> None:
    """compare convergence of multiple algorithms"""
    plt.figure(figsize=(10, 6))
    
    for name, data in results.items():
        plt.plot(data['iterations'], data['values'], label=name, marker='o')
    
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_multi_seed_comparison(
    aggregated_data: pd.DataFrame,
    x_col: str,
    y_mean_col: str,
    y_lower_col: str,
    y_upper_col: str,
    output_path: Optional[Path] = None,
    title: str = "Multi-Seed Results",
    xlabel: str = "Episode",
    ylabel: str = "Return"
) -> None:
    """plot mean with confidence interval"""
    plt.figure(figsize=(10, 6))
    
    plt.plot(aggregated_data[x_col], aggregated_data[y_mean_col], 
             linewidth=2, label='Mean')
    plt.fill_between(aggregated_data[x_col],
                     aggregated_data[y_lower_col],
                     aggregated_data[y_upper_col],
                     alpha=0.3, label='IQR')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
