import os, math
from typing import Any, Dict, List, Optional, Union
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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
            try: std_arr = np.asarray(std, dtype=float)
            except Exception: std_arr = None

            if std_arr is None: pass
            else:
                if std_arr.ndim == 0:
                    std_arr = np.full_like(y_for_band, float(std_arr), dtype=float)
                if std_arr.shape[0] == y_for_band.shape[0]:
                    lower_calc = y_for_band - std_arr
                    upper_calc = y_for_band + std_arr
                    plt.fill_between(x_pos, lower_calc, upper_calc, color=band_color, alpha=0.2, label=band_label) # type: ignore
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel if ylabel else "Loss")
    plt.title(title)
    plt.xscale(xscale) # type: ignore
    plt.grid(True)
    
    if any(lbl is not None for lbl in labels) and len(y_list) >= 1:
        plt.legend()
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

    ax.set_title(f"{title} (Score: {silhouette_avg:.2f})")
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


def plot_multiple_y_axes(x: Union[list, Any],
                         y_series: List[List],
                         labels: List[str],
                         colors: Optional[List[str]] = None,
                         markers: Optional[List[str]] = None,
                         xlabel: str = "",
                         title: str = "",
                         save_path: Optional[str] = None,
                         vline_x: Optional[float] = None,
                         vline_label: Optional[str] = None):
    """
    Plot multiple series, each with its own y-axis on the left side.
    Similar to the example with multiple left-side y-axes.
    """
    n_series = len(y_series)
    if n_series == 0:
        return
    
    fig, host = plt.subplots(figsize=(12, 7), dpi=300)
    
    # Default colors using a distinct color palette
    if colors is None:
        color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        colors = [color_palette[i % len(color_palette)] for i in range(n_series)]
    
    markers = markers or ['o'] * n_series
    
    # Create additional axes for each series (starting from index 1)
    axes = [host]
    spine_offset = 70  # Increased offset for better separation
    for i in range(1, n_series):
        ax = host.twinx()
        # Offset the spine position for visibility
        ax.spines['left'].set_position(('outward', spine_offset * i))
        ax.spines['left'].set_visible(True)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('left')
        axes.append(ax)
    
    # Plot each series on its own axis
    lines = []
    for i, (ax, y, label, color, marker) in enumerate(zip(axes, y_series, labels, colors, markers)):
        line, = ax.plot(x, y, color=color, marker=marker, label=label, 
                       linewidth=2, markersize=7, markeredgewidth=1.5, 
                       markeredgecolor='white', zorder=10 - i)
        ax.set_ylabel(label, color=color, fontweight='bold', fontsize=11)
        ax.tick_params(axis='y', labelcolor=color, colors=color, labelsize=9)
        ax.spines['left'].set_color(color)
        ax.spines['left'].set_linewidth(2)
        lines.append(line)
    
    # Optional vertical line
    if vline_x is not None:
        host.axvline(x=vline_x, color='black', linewidth=2, 
                    linestyle='--', alpha=0.8, zorder=1, 
                    label=vline_label or f'x={vline_x}')
    
    # Set x-axis label and title
    host.set_xlabel(xlabel, fontweight='bold', fontsize=12)
    host.set_title(title, fontweight='bold', fontsize=14, pad=15)
    
    # Integer ticks for x-axis
    if isinstance(x, (list, np.ndarray)) and all(isinstance(v, int) for v in x):
        host.set_xticks(x)
    
    # Style adjustments for better readability
    host.tick_params(axis='x', labelsize=10)
    host.spines['right'].set_visible(False)
    host.spines['top'].set_visible(False)
    
    # Only vertical grid lines (no horizontal)
    host.grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
    host.grid(False, axis='y')  # Disable horizontal grid lines
    
    # Create legend with all lines
    host.legend(lines, labels, loc='upper right', framealpha=0.9, 
               fontsize=10, ncol=min(3, n_series))
    
    # Adjust layout to prevent label cutoff
    plt.subplots_adjust(left=0.05 + (n_series - 1) * 0.05)
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_heatmap(data, title, save_path=None, 
                 xlabel='Column Index', ylabel='Row Index',
                 row_labels=None, col_labels=None,
                 colorbar_label='Value', cmap='RdBu_r',
                 vmin=None, vmax=None, figsize=None):
    """Generic heatmap visualization."""
    n_rows, n_cols = data.shape
    
    # Auto-scale colormap range if not provided (symmetric for diverging colormaps)
    if vmin is None:
        vmin = -max(abs(data.min()), abs(data.max()))
    if vmax is None:
        vmax = max(abs(data.min()), abs(data.max()))
    
    # Auto-scale figure size based on data dimensions
    if figsize is None:
        fig_width = max(10, min(20, n_cols * 0.3))
        fig_height = max(6, min(15, n_rows * 0.4))
        figsize = (fig_width, fig_height)
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    
    ax.set_xlabel(xlabel, fontsize=13, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    
    # Y-axis labels
    if row_labels is not None:
        ax.set_yticks(range(n_rows))
        label_fontsize = max(7, min(10, 120 // n_rows))
        ax.set_yticklabels(row_labels, fontsize=label_fontsize)
    
    # X-axis labels
    if col_labels is not None:
        ax.set_xticks(range(n_cols))
        label_fontsize = max(7, min(9, 150 // n_cols))
        ax.set_xticklabels(col_labels, fontsize=label_fontsize, rotation=45, ha='right')
    else:
        # Auto-scale ticks for readability when no labels provided
        if n_cols > 20:
            tick_step = max(1, n_cols // 10)
            ax.set_xticks(range(0, n_cols, tick_step))
            tick_fontsize = max(7, min(9, 150 // (n_cols // tick_step)))
            ax.set_xticklabels([str(i) for i in range(0, n_cols, tick_step)], 
                              fontsize=tick_fontsize)
        else:
            ax.set_xticks(range(n_cols))
            ax.set_xticklabels([str(i) for i in range(n_cols)], fontsize=9)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label, fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved heatmap to {save_path}")
    plt.close()


def plot_scatter_multicomponent_rgb(X_transformed, y, method_name, dataset_name, save_path=None, max_points=10000, comp_x=0, comp_y=1, comp_red=2, comp_green=3, title=None):
    """
    2D scatter plot with up to 4 components visualized:
    - Components comp_x and comp_y mapped to x/y axes
    - Component comp_red mapped to red channel intensity
    - Component comp_green mapped to green channel intensity
    """
    required_comps = max(comp_x, comp_y, comp_red, comp_green) + 1
    if X_transformed.shape[1] < required_comps:
        print(f"Need at least {required_comps} components for RGB composite. Found {X_transformed.shape[1]}")
        return
    
    # Cap number of points for performance
    if len(X_transformed) > max_points:
        indices = np.random.choice(len(X_transformed), max_points, replace=False)
        X_transformed = X_transformed[indices]
        y = y[indices]
    
    # Extract x, y positions
    x_vals = X_transformed[:, comp_x]
    y_vals = X_transformed[:, comp_y]
    
    # Normalize component values to [0, 1] for RGB channels
    def normalize_to_01(vals):
        vmin, vmax = vals.min(), vals.max()
        if vmax - vmin < 1e-10:
            return np.zeros_like(vals)
        return (vals - vmin) / (vmax - vmin)
    
    red_channel = normalize_to_01(X_transformed[:, comp_red])
    green_channel = normalize_to_01(X_transformed[:, comp_green])
    blue_channel = np.zeros_like(red_channel)  # Keep blue at 0 for now
    
    # Stack into RGB array
    colors = np.stack([red_channel, green_channel, blue_channel], axis=1)
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    ax.scatter(x_vals, y_vals, c=colors, alpha=0.7, s=30, edgecolors='none')
    
    ax.set_xlabel(f'Component {comp_x + 1}', fontsize=13, fontweight='bold')
    ax.set_ylabel(f'Component {comp_y + 1}', fontsize=13, fontweight='bold')
    if title is None: title = f'{method_name} Multi-Component RGB Projection - {dataset_name.title()}\nRed=Comp{comp_red+1}, Green=Comp{comp_green+1}'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    
    # Add a 2D 4-quadrant key as an inset axis
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    key_ax = inset_axes(ax, width="18%", height="18%", loc='upper right', borderpad=2)
    # Create a 2D grid for the key
    key_grid = np.zeros((100, 100, 3))
    for i in range(100):
        for j in range(100):
            r = i / 99.0  # Comp3 (red)
            g = j / 99.0  # Comp4 (green)
            key_grid[i, j, :] = [r, g, 0]
    key_ax.imshow(key_grid, origin='lower', extent=[0, 1, 0, 1])
    key_ax.set_xlabel(f'Comp{comp_red+1}', fontsize=10)
    key_ax.set_ylabel(f'Comp{comp_green+1}', fontsize=10)
    key_ax.set_xticks([0, 1])
    key_ax.set_yticks([0, 1])
    key_ax.tick_params(axis='both', which='both', length=0)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved multi-component RGB projection to {save_path}")
    plt.close()


def plot_scatter(X_transformed, y, method_name, dataset_name, save_path=None, xlabel='Component 1', ylabel='Component 2', components=None, max_points=10000, title=None):
    """2D scatter plot of first 2 components colored by target."""
    if X_transformed.shape[1] < 2:
        print("Need at least 2 components for 2D plot")
        return

    # cap number of points for performance
    if len(X_transformed) > max_points:
        indices = np.random.choice(len(X_transformed), max_points, replace=False)
        X_transformed = X_transformed[indices]
        y = y[indices]
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Determine if classification or regression
    unique_vals = np.unique(y)
    is_classification = len(unique_vals) <= 20
    
    if is_classification:
        n_classes = len(unique_vals)
        if n_classes == 2:
            # Binary classification: simple 2-color scheme
            colors = ['#3498DB', '#E74C3C']  # Blue for 0, Red for 1
            for idx, class_val in enumerate(sorted(unique_vals)):
                mask = (y == class_val)
                ax.scatter(X_transformed[mask, 0], X_transformed[mask, 1],
                          c=colors[idx], label=f'Class {int(class_val)}',
                          alpha=0.5, s=20, edgecolors='none')
            ax.legend(loc='best', fontsize=11, framealpha=0.95, title='Target')
        else:
            # Multi-class: use qualitative colormap
            scatter = ax.scatter(X_transformed[:, 0], X_transformed[:, 1], 
                               c=y, cmap='tab10' if n_classes <= 10 else 'tab20',
                               alpha=0.6, s=20, edgecolors='none')
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Class', fontsize=12, fontweight='bold')
            cbar.ax.tick_params(labelsize=10)
    else:
        # Regression: span values over a continuous colormap
        scatter = ax.scatter(
            X_transformed[:, 0], X_transformed[:, 1], 
            c=y, cmap='viridis', alpha=0.6, s=20, edgecolors='none')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Target Value', fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)
    
    ax.set_xlabel(xlabel, fontsize=13, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
    if title is None: title = f'{method_name} 2D Projection - {dataset_name.title()}'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved 2D projection to {save_path}")
    plt.close()