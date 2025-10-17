import os, math
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from sklearn import tree
from sklearn.base import is_classifier
from sklearn.calibration import LinearSVC, calibration_curve
from sklearn.datasets import make_blobs
from sklearn.metrics import ConfusionMatrixDisplay, auc, precision_recall_curve, roc_curve
from sklearn.model_selection import cross_val_score
from typing import Any, Dict, List, Optional, Union

matplotlib.use('Agg')

def plot_curve(x: Union[list, Any], y_list: list, labels: Optional[list[str]] = None, xlabel: str = "", ylabel: str = "", title: str = "", save_path: Optional[str] = None, marker: Optional[str] = None, colors: Optional[list[str]] = None, linestyles: Optional[list[str]] = None, xscale: str = 'linear', std: Optional[List[float]] = None, lower: Optional[List[float]] = None, upper: Optional[List[float]] = None, band_label: str = '± Std Dev'):
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
    
    # check for per-curve x (list of sequences)
    if isinstance(x, list) and len(x) == len(y_list) and all(hasattr(xi, '__iter__') and not isinstance(xi, str) for xi in x):  # changed: added check for iterable xi (not scalar)
        for i, (x_i, y, lbl) in enumerate(zip(x, y_list, labels)):  # per-curve x
            if any(isinstance(v, str) or v is None for v in x_i):
                x_pos = np.arange(len(x_i))
                plt.xticks(x_pos, [str(v) for v in x_i])
            else:
                x_pos = x_i
            plt.plot(x_pos, y, label=lbl, marker=marker, color=colors[i], linestyle=linestyles[i])
    else:  # single x for all
        if any(isinstance(v, str) or v is None for v in x):  # type: ignore
            x_pos = np.arange(len(x))  # type: ignore
            plt.xticks(x_pos, [str(v) for v in x])  # type: ignore
        else:
            x_pos = x  # type: ignore
        for i, (y, lbl) in enumerate(zip(y_list, labels)):
            plt.plot(x_pos, y, label=lbl, marker=marker, color=colors[i], linestyle=linestyles[i])
        
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
    plt.xscale(xscale)
    plt.grid(True)
    
    if any(lbl is not None for lbl in labels):  # changed: show legend only if any non-None labels
        plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_learning_curve(train_sizes, train_scores, val_scores, scoring, title="Learning Curve", save_path=None):
    scoring_label = "Weighted F1" if scoring == "f1_weighted" else "R2"
    plot_curve(
        x=train_sizes,
        y_list=[train_scores, val_scores],
        labels=[f"Train {scoring_label}", f"Validation {scoring_label}"],
        xlabel="Training Set Size",
        ylabel=f"{scoring_label} Score",
        title=title,
        save_path=f"{save_path}/learning_curve.png" if save_path else None,
        marker="o"
    )


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

def plot_heatmaps(data, labels, x_labels, title, save_path):
    plt.figure(figsize=(10, 6))
    sns.heatmap(data, xticklabels=x_labels, yticklabels=labels, cmap='viridis')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()


def plot_combined_heatmap(
    data_dict: Dict[str, np.ndarray],
    alpha_grid: List[float],
    beta1_grid: List[float],
    beta2_grid: List[float],
    optimizers: List[str],
    title: str = "",
    save_path: Optional[str] = None,
    cmap: str = 'viridis'
):
    '''
    create a single heatmap combining alphas (rows), betas (columns segmented by optimizer).
    flattens beta1 and beta2 into one axis with optimizer-prefixed labels.
    '''
    # Build combined x labels and collect per-optimizer data blocks.
    x_labels = []
    per_opt_blocks = []

    for opt in optimizers:
        # prefer direct key if present
        direct = data_dict.get(opt)
        if direct is not None:
            block = np.asarray(direct, dtype=float)
            # assume block shape matches (len(alpha_grid), Ncols)
            per_opt_blocks.append(block)
            # create generic labels for these columns
            for col_idx in range(block.shape[1]):
                x_labels.append(f"{opt} col{col_idx}")
            continue

        # otherwise look for suffixed keys produced by ablation analysis
        b1_key = f"{opt}_alpha_b1"
        b2_key = f"{opt}_alpha_b2"
        b1 = data_dict.get(b1_key)
        b2 = data_dict.get(b2_key)

        blocks = []
        # beta1 block (may be missing for rmsprop_like)
        if b1 is not None:
            b1_arr = np.asarray(b1, dtype=float)
            blocks.append(b1_arr)
            for b in beta1_grid:
                x_labels.append(f"{opt} β1={b:.3f}")

        # beta2 block
        if b2 is not None:
            b2_arr = np.asarray(b2, dtype=float)
            blocks.append(b2_arr)
            for b in beta2_grid:
                x_labels.append(f"{opt} β2={b:.3f}")

        if blocks:
            # horizontally concatenate available blocks for this optimizer
            try:
                block = np.hstack(blocks)
            except Exception:
                # if shapes don't align, try padding/trimming to alpha_grid length
                aligned = []
                for bb in blocks:
                    bb_arr = np.asarray(bb, dtype=float)
                    if bb_arr.shape[0] != len(alpha_grid):
                        # try to reshape or broadcast
                        try:
                            bb_arr = np.tile(bb_arr.reshape(-1, bb_arr.shape[1]) if bb_arr.ndim == 2 else bb_arr.reshape(-1, 1), (len(alpha_grid), 1))
                        except Exception:
                            # last resort: create zeros
                            bb_arr = np.zeros((len(alpha_grid), bb_arr.shape[1] if bb_arr.ndim == 2 else 1))
                    aligned.append(bb_arr)
                block = np.hstack(aligned)
            per_opt_blocks.append(block)
        else:
            # no data for this optimizer; skip but keep a placeholder label group
            # (we won't add any columns for this optimizer)
            continue

    if not per_opt_blocks:
        raise ValueError("No optimizer data found in data_dict for provided optimizers.")

    # combine data arrays horizontally (columns for betas per optimizer)
    combined_data = np.hstack(per_opt_blocks)

    # plot the combined heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(combined_data, ax=ax, xticklabels=x_labels, yticklabels=[f"{a:.0e}" for a in alpha_grid], cmap=cmap, annot=False, fmt=".2f")
    ax.set_xlabel("Betas by Optimizer")
    ax.set_ylabel("Alphas")
    ax.set_title(title)
    plt.xticks(rotation=45, ha='right')  # rotate labels for readability
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()