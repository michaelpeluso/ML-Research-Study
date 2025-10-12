import os, math
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from sklearn import tree
from sklearn.base import is_classifier
from sklearn.calibration import LinearSVC, calibration_curve
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay, permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, auc, precision_recall_curve, roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.tree import plot_tree
from typing import List, Optional, Union

matplotlib.use('Agg')

def plot_curve(x, y_list, labels=None, xlabel="", ylabel="", title="", save_path=None, marker=None):
    plt.figure(figsize=(8,5))

    if any(isinstance(v, str) or v is None for v in x):
        x_pos = np.arange(len(x))
        plt.xticks(x_pos, [str(v) for v in x])
    else:
        x_pos = x
    
    for y, lbl in zip(y_list, labels): # type: ignore
        plt.plot(x_pos, y, label=lbl, marker=marker)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    
    if any(labels): # type: ignore
        plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_model_evaluation(y_test, y_proba, y_pred, metrics, class_names, train_sizes, train_scores, val_scores, scoring, save_path="./", stitched_title="Stitched Models"):
    if save_path: os.makedirs(save_path, exist_ok=True)

    # Learning curve
    plot_learning_curve(train_sizes, train_scores, val_scores, scoring, save_path=save_path)

    # Confusion matrix - classification only
    if "ConfusionMatrix" in metrics and metrics["ConfusionMatrix"] is not None:
        plot_confusion_matrix(metrics["ConfusionMatrix"], class_names, save_path=save_path)

    # PR/ROC/Calibration curves - classification with proba only
    if y_proba is not None and len(y_proba.shape) > 1:
        if y_proba.shape[1] == 2:  # Binary
            y_test_list = [np.ravel(y_test)]
            y_proba_list = [y_proba[:, 1]]
            labels = [f"Class {class_names[1]}"] if class_names else ["Positive Class"]
        else:
            y_test_list = [np.array(y_test) == cls for cls in class_names]
            y_proba_list = [y_proba[:, i] for i in range(y_proba.shape[1])]
            labels = class_names

        plot_pr_curves(y_test_list, y_proba_list, labels, save_path=save_path)
        plot_roc_curves(y_test_list, y_proba_list, labels, save_path=save_path)
        plot_calibration_curve(y_test_list, y_proba_list, labels, save_path=save_path)

    # Residual - regression only
    if "r2" in metrics and y_pred is not None: 
        plot_residuals(y_test, y_pred, save_path=save_path)

    stitch_images(save_path, stitched_title)


def plot_confusion_matrix(cm: np.ndarray, class_names, normalize=True, title="Confusion Matrix", save_path=None):
    if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap="Blues", ax=ax, values_format=".2f" if normalize else "d")
    plt.title(title)
    if save_path: plt.savefig(f"{save_path}/confusion_matrix.png")
    else: plt.show()
    plt.close()


def plot_pr_curves(y_true_list, y_proba_list, labels, title="Precision-Recall Curve", save_path=None):
    plt.figure(figsize=(8, 6))
    
    for y_true, y_proba, lbl in zip(y_true_list, y_proba_list, labels):
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f"{lbl} (AUC={pr_auc:.2f})")
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    
    if save_path: plt.savefig(f"{save_path}/pr_curve.png")
    else: plt.show()
    plt.close()


def plot_roc_curves(y_true_list, y_proba_list, labels, title="ROC Curve", save_path=None):
    plt.figure(figsize=(8,6))
    
    for y_true, y_proba, lbl in zip(y_true_list, y_proba_list, labels):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc_score_value = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{lbl} (AUC={roc_auc_score_value:.2f})")
    
    # Random baseline
    plt.plot([0,1], [0,1], 'k--', label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    
    if save_path: plt.savefig(f"{save_path}/roc_curve.png")
    else: plt.show()
    plt.close()


def plot_calibration_curve(y_true_list, y_proba_list, class_names, title="Calibration Curve", save_path=None):
    plt.figure(figsize=(8, 6))
    for i, (y_true, y_proba, name) in enumerate(zip(y_true_list, y_proba_list, class_names)):
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
        plt.plot(prob_pred, prob_true, label=f"{name}")
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    if save_path: plt.savefig(f"{save_path}/calibration_curve.png")
    else: plt.show()
    plt.close()

def plot_feature_importance(feature_importances, n_count=None, title="Decision Tree Feature Importance", save_path=None):
    print("Plotting decision tree feature importance.")
    sorted_items = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    features, importance = zip(*sorted_items)
    
    plt.figure(figsize=(8,6))
    plt.bar(features, importance)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Importance")
    plt.title(title + f"(Node Count: {n_count})")
    plt.tight_layout()
    if save_path: plt.savefig(save_path)
    else: plt.show()
    plt.close()

def plot_epoch_curve(
    x: List[Union[int, float]], 
    y: List[float], 
    title: str = "RO Curve", 
    save_path: Optional[str] = None,
    log_x: bool = False,
    caption: Optional[str] = None
) -> None:
    '''
    Plot RO curve with optional log-x and caption.
    '''
    plt.figure(figsize=(8, 6))  # set figure size
    
    # plot data
    plt.plot(x, y, label="Loss")
    
    plt.xlabel("Evals")  # x label
    plt.ylabel("Loss")  # y label
    plt.title(title)  # title
    plt.grid(True)  # add grid
    
    # log x scale
    if log_x:
        plt.xscale('log')
    
    # add caption
    if caption:
        plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment='center', fontsize=8)
    
    plt.legend()  # add legend
    if save_path:
        plt.savefig(save_path)  # save plot
    else:
        plt.show()
    plt.close()

def plot_residuals(y_true, y_pred, title="Residual Plot", save_path=None):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title(title)
    plt.grid(True)
    if save_path: plt.savefig(f"{save_path}/residual_plot.png")
    else: plt.show()
    plt.close()

def plot_complexity_curve(complexity_data, scoring='R2', save_path=None):
    param_name = complexity_data['param_name'].replace('estimator__', '').title()
    param_values = complexity_data['param_values']
    train_mean = complexity_data['train_scores_mean']
    val_mean = complexity_data['val_scores_mean']
    train_std = complexity_data['train_scores_std']
    val_std = complexity_data['val_scores_std']
    
    # Filter out None or non-numeric values
    valid_idx = [i for i, v in enumerate(param_values) if v is not None and isinstance(v, (int, float))]
    if not valid_idx:
        print("No valid numeric param_values for plotting. Skipping plot.")
        return

    x_pos = np.array(valid_idx)
    param_labels = [str(param_values[i]) for i in valid_idx]
    train_mean = np.array(train_mean)[valid_idx]
    val_mean = np.array(val_mean)[valid_idx]
    train_std = np.array(train_std)[valid_idx]
    val_std = np.array(val_std)[valid_idx]
    
    plt.figure(figsize=(8, 6))
    plt.plot(x_pos, train_mean, 'o-', label=f'Training {scoring}', color='#1f77b4')
    plt.plot(x_pos, val_mean, 's-', label=f'Validation {scoring}', color='#ff7f0e')
    plt.fill_between(x_pos, train_mean - train_std, train_mean + train_std, alpha=0.2, color='#1f77b4')
    plt.fill_between(x_pos, val_mean - val_std, val_mean + val_std, alpha=0.2, color='#ff7f0e')

    plt.xlabel(param_name.replace('_', ' ').title())
    plt.ylabel(f'{scoring} Score')
    plt.title("Model Complexity Curve")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(x_pos, param_labels, rotation=45)
    
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


def plot_pruning_path(pipeline, X_train, y_train, save_path=None):
    print('Plotting decision tree pruning path.')
    alphas = [0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    scores = []
    estimator = pipeline.named_steps['estimator']
    scoring = 'average_precision' if is_classifier(estimator) else 'neg_mean_squared_error'
    
    for alpha in alphas:
        pipeline.set_params(estimator__ccp_alpha=alpha)
        score = cross_val_score(pipeline, X_train, y_train, cv=5, scoring=scoring, n_jobs=-1).mean()
        scores.append(score)
    
    plt.figure(figsize=(8, 6))
    plt.plot(alphas, scores, 'o-')
    plt.xlabel('CCP Alpha (pruning parameter)')
    plt.ylabel('Cross-Validation Score')
    plt.title('Decision Tree Pruning Path')
    plt.xscale('log')
    plt.grid(True)
    
    if save_path: plt.savefig(f"{save_path}/pruning_path.png")
    else: plt.show()
    plt.close()

    class_names = [str(c) for c in estimator.classes_] if hasattr(estimator, "classes_") else None
    plt.figure(figsize=(12, 9))
    tree.plot_tree(estimator, feature_names=X_train.columns, class_names=class_names, filled=True, fontsize=1)
    if save_path: plt.savefig(f"{save_path}/decision_tree.png", dpi=600)
    else: plt.show()
    plt.close()
    
    # Reset to best alpha
    best_alpha = alphas[np.argmax(scores)]
    pipeline.set_params(estimator__ccp_alpha=best_alpha)
    return best_alpha

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