import os, math
from typing import Any, Dict, List, Optional, Union
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import silhouette_samples, silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA

matplotlib.use('Agg')

def plot_curve(x: Union[list, Any], y_list: list, labels: Optional[list[str]] = None, xlabel: str = "", ylabel: str = "", title: str = "", save_path: Optional[str] = None, marker: Optional[Union[str, List[str]]] = None, colors: Optional[list[str]] = None, linestyles: Optional[list[str]] = None, xscale: str = 'linear', std: Optional[List[float]] = None, lower: Optional[List[float]] = None, upper: Optional[List[float]] = None, band_label: str = '± Std Dev'):
    '''
    plot curve with optional labels, colors, linestyles, xscale, grid, and save.
    supports stability bands via std (symmetric) or explicit lower/upper (e.g., for iqr).
    bands applied to first y in y_list only.
    '''
    plt.figure(figsize=(8,5), dpi=300)
    
    # wrap single curve if flat list of numbers
    if isinstance(y_list[0], (int, float)) if y_list else False:  # changed: check if first element is scalar
        y_list = [y_list]  # changed: wrap into list for single curve
    
    colors = colors or ['blue'] * len(y_list)  # default colors if none
    linestyles = linestyles or ['-'] * len(y_list)  # default solid lines
    labels = labels or [""] * len(y_list)  # changed: default to empty string per curve to match length and type
    
    # Handle markers: if str, use for all; if list, use per curve
    if isinstance(marker, list):
        markers = marker
    else:
        markers = [marker] * len(y_list)
    
    # check for per-curve x (list of sequences)
    if isinstance(x, list) and len(x) == len(y_list) and all(hasattr(xi, '__iter__') and not isinstance(xi, str) for xi in x):  # changed: added check for iterable xi (not scalar)
        for i, (x_i, y, lbl) in enumerate(zip(x, y_list, labels)):  # per-curve x
            if any(isinstance(v, str) or v is None for v in x_i):
                x_pos = np.arange(len(x_i))
                plt.xticks(x_pos, [str(v) for v in x_i])
            else:
                x_pos = x_i
            plt.plot(x_pos, y, label=lbl, marker=markers[i], color=colors[i], linestyle=linestyles[i])
    else:  # single x for all
        if any(isinstance(v, str) or v is None for v in x):  # type: ignore
            x_pos = np.arange(len(x))  # type: ignore
            plt.xticks(x_pos, [str(v) for v in x])  # type: ignore
        else:
            x_pos = x  # type: ignore
        for i, (y, lbl) in enumerate(zip(y_list, labels)):
            plt.plot(x_pos, y, label=lbl, marker=markers[i], color=colors[i], linestyle=linestyles[i])
        
    # add stability bands for first y if provided
    if len(y_list) > 0:
        y_for_band = np.atleast_1d(np.array(y_list[0], dtype=float))
        band_color = colors[0] if colors else 'blue'
        if lower is not None and upper is not None:
            plt.fill_between(x_pos, lower, upper, color=band_color, alpha=0.2, label=band_label) # type: ignore
        elif std is not None:
            # Robust handling: accept scalar/numpy scalar/list/ndarray for std
            try:
                std_arr = np.asarray(std, dtype=float)
            except Exception:
                std_arr = None

            if std_arr is None:
                pass
            else:
                # If std_arr is a scalar (0-d), broadcast to length of y_for_band
                if std_arr.ndim == 0:
                    std_arr = np.full_like(y_for_band, float(std_arr), dtype=float)
                # If lengths match, plot the band
                if std_arr.shape[0] == y_for_band.shape[0]:
                    lower_calc = y_for_band - std_arr
                    upper_calc = y_for_band + std_arr
                    plt.fill_between(x_pos, lower_calc, upper_calc, color=band_color, alpha=0.2, label=band_label) # type: ignore
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel if ylabel else "Loss")
    plt.title(title)
    plt.xscale(xscale) # type: ignore
    plt.grid(True)
    
    if any(lbl is not None for lbl in labels):  # changed: show legend only if any non-None labels
        plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def stitch_images(img_dir, title=""):
    filename = "_plots_stitched.png"
    files = sorted(f for f in os.listdir(img_dir) if f.endswith(".png") and f != filename)
    imgs = [Image.open(os.path.join(img_dir, f)).convert("RGB") for f in files]
    
    n = len(imgs)
    if n == 0: return
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    w = min(i.width for i in imgs)
    h = min(i.height for i in imgs)
    imgs = [im.resize((w, h), Image.LANCZOS) for im in imgs] # type: ignore

    
    title_h = 100
    out = Image.new("RGB", (cols * w, rows * h + title_h), "white")
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    font = ImageFont.truetype("arial.ttf", 48)
    bbox = draw.textbbox((0,0), title, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((cols * w - tw) // 2, 40), title, fill="black", font=font)
    
    for idx, im in enumerate(imgs):
        r, c = divmod(idx, cols)
        out.paste(im, (c * w, r * h + title_h))
    
    out_path = os.path.join(img_dir, filename)
    if os.path.exists(out_path): os.remove(out_path)
    out.save(out_path)


def plot_sensitivity_subplots(
    sensitivity_data: List[Dict[str, Any]],
    overall_title: str,
    save_path: str,
    figsize: tuple = (15, 10),
    dpi: int = 150
):
    num_plots = len(sensitivity_data)    
    n_cols = min(3, num_plots)
    n_rows = (num_plots + n_cols - 1) // n_cols  
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Handle single subplot case
    if num_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each sensitivity analysis
    for i, data in enumerate(sensitivity_data):
        ax = axes[i]
        color = data.get('color', f'C{i}')
        
        ax.plot(data['x'], data['y'], 'o-', color=color, linewidth=2, markersize=6)
        ax.set_xlabel(data['xlabel'], fontsize=10)
        ax.set_ylabel('Validation Loss', fontsize=10)
        ax.set_title(data['title'], fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(num_plots, len(axes)):
        axes[i].axis('off')
    
    # Add overall title
    plt.suptitle(overall_title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.99))
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved sensitivity subplots to: {save_path}")




def plot_silhouette(X, labels, title="Silhouette Plot", save_path=None):
    """
    Plot silhouette scores for clustering evaluation.
    """
    silhouette_avg = silhouette_score(X, labels)
    sample_silhouette_values = silhouette_samples(X, labels)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    y_lower = 10
    for i in range(max(labels) + 1):
        ith_cluster_silhouette_values = np.array(sample_silhouette_values[labels == i])  # type:ignore
        ith_cluster_silhouette_values = np.sort(ith_cluster_silhouette_values)
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = plt.cm.nipy_spectral(float(i) / (max(labels) + 1))  # type:ignore
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)  # type: ignore
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    
    ax.set_title(f"{title} (Avg Score: {silhouette_avg:.2f})")
    ax.set_xlabel("Silhouette Coefficient Values")
    ax.set_ylabel("Cluster Label")
    ax.axvline(x=float(silhouette_avg), color="red", linestyle="--")
    ax.set_yticks([])
    ax.set_xlim(-0.1, 1)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_scatter(x_list, y_list, labels=None, xlabel="", ylabel="", title="", save_path=None, colors=None, alpha=0.7, figsize=(8, 6)):
    """
    Generic scatter plot function for multiple series.
    """
    plt.figure(figsize=figsize)
    
    colors = colors or ['blue'] * len(x_list)
    labels = labels or [""] * len(x_list)
    
    for i, (x, y, lbl, col) in enumerate(zip(x_list, y_list, labels, colors)):
        plt.scatter(x, y, label=lbl, color=col, alpha=alpha)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_cluster_scatter(X, labels, method='pca', title="Cluster Scatter Plot", save_path=None):
    """
    Scatter plot of clusters after dimensionality reduction to 2D.
    """
    if X.shape[1] > 2:
        reducer = PCA(n_components=2, random_state=42)
        X_reduced = reducer.fit_transform(X)
    else:
        X_reduced = X
    
    unique_labels = np.unique(labels)
    x_list = []
    y_list = []
    labels_list = []
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'][:len(unique_labels)]
    
    for k, col in zip(unique_labels, colors):
        class_member_mask = (labels == k)
        xy = X_reduced[class_member_mask]
        x_list.append(xy[:, 0])
        y_list.append(xy[:, 1])
        labels_list.append(f'Cluster {k}')
    
    plot_scatter(
        x_list=x_list,
        y_list=y_list,
        labels=labels_list,
        xlabel=f'{method.upper()} Component 1',
        ylabel=f'{method.upper()} Component 2',
        title=title,
        save_path=save_path,
        colors=colors
    )


