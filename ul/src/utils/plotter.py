import os, math
from typing import Any, Dict, List, Optional, Union
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import adjusted_rand_score
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
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
    plt.close()


def plot_dual_axis(x: Union[list, Any], 
                   y_left: Union[list, list[list]], 
                   y_right: Union[list, list[list], None] = None,
                   left_labels: Optional[list[str]] = None,
                   right_labels: Optional[list[str]] = None,
                   left_ylabel: str = "",
                   right_ylabel: str = "",
                   xlabel: str = "",
                   title: str = "",
                   save_path: Optional[str] = None,
                   left_colors: Optional[list[str]] = None,
                   right_colors: Optional[list[str]] = None,
                   left_markers: Optional[list[str]] = None,
                   right_markers: Optional[list[str]] = None,
                   vline_x: Optional[float] = None,
                   vline_label: Optional[str] = None):
    """
    Plot multiple series on left and/or right axes with dual y-axes.
    
    Args:
        vline_x: Optional x-value for vertical line (e.g., chosen_k)
        vline_label: Optional label for the vertical line
    """
    plt.figure(figsize=(8, 5), dpi=300)
    ax_left = plt.gca()
    
    # optional vertical line
    if vline_x is not None:
        ax_left.axvline(x=vline_x, color='red', linewidth=1, 
                       alpha=0.7, zorder=1, label=vline_label or f'x={vline_x}')

    # Normalize y_left to list of lists
    if isinstance(y_left[0], (int, float)) if y_left else False:
        y_left_list = [y_left]
    else:
        y_left_list = y_left

    # Normalize y_right to list of lists
    if y_right is not None:
        if isinstance(y_right[0], (int, float)) if y_right else False:
            y_right_list = [y_right]
        else:
            y_right_list = y_right
    else:
        y_right_list = []

    # Default colors, markers, and linestyles
    left_colors = left_colors or ['blue'] * len(y_left_list)
    right_colors = right_colors or ['orange'] * len(y_right_list)
    left_markers = left_markers or ['o'] * len(y_left_list)
    right_markers = right_markers or ['o'] * len(y_right_list)
    left_linestyles = ['-', '-.', ':', '--'][:len(y_left_list)]
    right_linestyles = ['-', '-.', ':', '--'][:len(y_right_list)]
    left_labels = left_labels or [f"Left {i+1}" for i in range(len(y_left_list))]
    right_labels = right_labels or [f"Right {i+1}" for i in range(len(y_right_list))]

    # Integer ticks for x-axis
    if isinstance(x, (list, np.ndarray)) and all(isinstance(v, int) for v in x):
        ax_left.set_xticks(x)

    # Plot left axis series
    lines_left = []
    for i, (y, color, marker, lbl, ls) in enumerate(zip(y_left_list, left_colors, left_markers, left_labels, left_linestyles)):
        line, = ax_left.plot(x, y, color=color, marker=marker, label=lbl, linewidth=2, linestyle=ls, zorder=3)
        lines_left.append(line)
    ax_left.set_ylabel(left_ylabel or " / ".join(left_labels), color='black', fontweight='bold')
    ax_left.tick_params(axis='y', labelcolor='blue', colors='blue')
    ax_left.spines['left'].set_color('blue')
    ax_left.spines['left'].set_linewidth(2)

    # Plot right axis series
    lines_right = []
    if y_right_list:
        ax_right = ax_left.twinx()
        for i, (y, color, marker, lbl, ls) in enumerate(zip(y_right_list, right_colors, right_markers, right_labels, right_linestyles)):
            line, = ax_right.plot(x, y, color=color, marker=marker, label=lbl, linewidth=2, linestyle=ls, zorder=3)
            lines_right.append(line)
        ax_right.set_ylabel(right_ylabel or " / ".join(right_labels), color='black', fontweight='bold')
        ax_right.tick_params(axis='y', labelcolor='orange', colors='orange')
        ax_right.spines['right'].set_color('orange')
        ax_right.spines['right'].set_linewidth(2)

    # Labels and legend
    ax_left.set_xlabel(xlabel)
    plt.title(title)
    
    all_lines = lines_left + lines_right
    all_labels = left_labels + right_labels
    ax_left.legend(all_lines, all_labels)

    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
    plt.close()


def stitch_specific_images(image_paths, title="", output_path=None):
    """
    Stitch specific images from given paths into a single image.
    """
    if not image_paths:
        return
    
    # Load images
    imgs = []
    for path in image_paths:
        if os.path.exists(path):
            imgs.append(Image.open(path).convert("RGB"))
        else:
            print(f"Warning: Image not found: {path}")
    
    if not imgs:
        return
    
    n = len(imgs)
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
    
    bbox = draw.textbbox((0,0), title)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((cols * w - tw) // 2, 40), title, fill="black")
    
    for idx, im in enumerate(imgs):
        r, c = divmod(idx, cols)
        out.paste(im, (c * w, r * h + title_h))
    
    # Determine output path
    if output_path is None:
        # Save next to the first image
        first_dir = os.path.dirname(image_paths[0])
        output_path = os.path.join(first_dir, "_stitched.png")
    
    # Remove existing file if it exists
    if os.path.exists(output_path):
        os.remove(output_path)
    
    out.save(output_path)

def stitch_images(img_dir, title=""):
    filename = "_stitched.png"
    files = sorted(f for f in os.listdir(img_dir) if f.endswith(".png") and f != filename)
    image_paths = [os.path.join(img_dir, f) for f in files]
    output_path = os.path.join(img_dir, filename)
    stitch_specific_images(image_paths, title, output_path)


def plot_silhouette(X, labels, title="Silhouette Plot", save_path=None, silhouette_avg: float | None = None, sample_silhouette_values: np.ndarray | None = None, sample_indices: Optional[np.ndarray] = None):
    """
    Plot silhouette scores for clustering evaluation.
    """
    # Require precomputed values to avoid hidden expensive computations in the plotter
    if silhouette_avg is None or sample_silhouette_values is None:
        raise ValueError("plot_silhouette requires precomputed `silhouette_avg` and `sample_silhouette_values`.\n"
                         "Compute them in the clustering module (e.g., Clustering.compute_silhouette) and pass them in.")

    # Align labels with the provided sample silhouette values. If `sample_indices` is provided,
    # the silhouette values correspond to labels[sample_indices]. Otherwise they correspond to the
    # full label array.
    if sample_indices is not None:
        labels_arr = np.array(labels)
        try:
            labels_to_plot = labels_arr[sample_indices]
        except Exception:
            # Fallback: if indices incompatible, raise a clear error
            raise ValueError("Provided `sample_indices` cannot be applied to given labels array.")
    else:
        labels_to_plot = np.array(labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    y_lower = 10
    n_clusters = max(labels_to_plot) + 1
    for i in range(n_clusters):
        ith_cluster_silhouette_values = np.array(sample_silhouette_values[labels_to_plot == i])  # type:ignore
        if ith_cluster_silhouette_values.size == 0:
            continue
        ith_cluster_silhouette_values = np.sort(ith_cluster_silhouette_values)
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = plt.cm.nipy_spectral(float(i) / n_clusters)  # type:ignore
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
        plt.savefig(save_path, dpi=300)
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
        plt.savefig(save_path, dpi=300)
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
    n_clusters = len(unique_labels)
    x_list = []
    y_list = []
    labels_list = []
    # Use the same colormap as silhouette plot for consistency
    colors = [plt.cm.nipy_spectral(float(i) / n_clusters) for i in range(n_clusters)]  # type: ignore
    
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


def plot_bar(x_labels, values, xlabel="", ylabel="", title="", save_path=None, color='skyblue'):
    """
    Simple bar plot function.
    """
    plt.figure(figsize=(8, 5))
    plt.bar(x_labels, values, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
    plt.close()


def plot_hist(values, bins: int = 20, xlabel: str = "", ylabel: str = "Count", title: str = "", save_path: Optional[str] = None, color: str = 'tab:blue', figsize=(4, 3), dpi: int = 150):
    """
    Simple histogram plot helper.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize, dpi=dpi)
    plt.hist(values, bins=bins, color=color, alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=dpi)
    else:
        plt.show()
    plt.close()