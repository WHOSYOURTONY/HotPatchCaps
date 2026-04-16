"""
visualization.py

Plotting utilities for HotPatchCaps evaluation results.

Functions:
    plot_confusion_matrix       – heatmap of a confusion matrix
    plot_per_class_metrics      – grouped bar chart (P/R/F1 per class)
    plot_threshold_distribution – histogram of endpoint/model thresholds
    plot_training_curves        – loss and accuracy vs epoch
    plot_embedding_scatter      – 2-D scatter of reduced-dimension embeddings
                                  (t-SNE, PCA, or UMAP)
"""

import argparse
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Core plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = 'Confusion Matrix',
    save_path: Optional[str] = None,
    figsize: Optional[tuple] = None,
) -> None:
    """
    Plot a labelled confusion matrix heatmap.

    Args:
        cm        : square ndarray, shape [N, N]
        labels    : class name strings (length N)
        title     : figure title
        save_path : file path to save (PNG/PDF); if None, calls plt.show()
        figsize   : (width, height) in inches; auto-scaled if None
    """
    n = cm.shape[0]
    if labels is None:
        labels = [str(i) for i in range(n)]
    if figsize is None:
        figsize = (max(6, n), max(5, n - 1))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    _save_or_show(save_path)


def plot_per_class_metrics(
    class_names: List[str],
    precision: List[float],
    recall: List[float],
    f1: List[float],
    title: str = 'Per-Class Metrics',
    save_path: Optional[str] = None,
) -> None:
    """
    Grouped bar chart showing Precision, Recall, and F1 for each class.

    Args:
        class_names : list of class name strings
        precision   : per-class precision values (same length)
        recall      : per-class recall values
        f1          : per-class F1 values
        title       : figure title
        save_path   : file path to save; if None, calls plt.show()
    """
    df = pd.DataFrame({
        'Class':     class_names,
        'Precision': precision,
        'Recall':    recall,
        'F1 Score':  f1,
    })
    df_m = df.melt(id_vars='Class', var_name='Metric', value_name='Score')

    n  = len(class_names)
    fw = max(10, n * 1.5)
    fig, ax = plt.subplots(figsize=(fw, 6))
    sns.barplot(x='Class', y='Score', hue='Metric', data=df_m, ax=ax)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    _save_or_show(save_path)


def plot_threshold_distribution(
    thresholds: List[float],
    title: str = 'Unknown Detection Threshold Distribution',
    save_path: Optional[str] = None,
) -> None:
    """
    Histogram + KDE of unknown-detection threshold values.

    Args:
        thresholds : list of float threshold values to visualize
        title      : figure title
        save_path  : file path to save; if None, calls plt.show()
    """
    if not thresholds:
        print("[visualization] No thresholds to plot.")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(thresholds, kde=True, bins=20, ax=ax)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Threshold Value (tau)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    _save_or_show(save_path)


def plot_training_curves(
    train_loss: List[float],
    eval_acc: List[float],
    epochs: Optional[List[int]] = None,
    title: str = 'Training Curves',
    save_path: Optional[str] = None,
) -> None:
    """
    Side-by-side plot of training loss and validation accuracy per epoch.

    Args:
        train_loss : list of per-epoch training losses
        eval_acc   : list of per-epoch validation accuracies
        epochs     : optional explicit epoch numbers; defaults to 1-indexed range
        title      : suptitle for the figure
        save_path  : file path to save; if None, calls plt.show()
    """
    if epochs is None:
        epochs = list(range(1, len(train_loss) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14)

    ax1.plot(epochs, train_loss, marker='o', color='steelblue')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, eval_acc, marker='o', color='seagreen')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1.05)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    _save_or_show(save_path)


def plot_embedding_scatter(
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_names: Optional[Dict[int, str]] = None,
    method: str = 'tsne',
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Reduce high-dimensional embeddings to 2-D and visualize as a scatter plot.

    Args:
        embeddings  : [N, D] float array of embedding vectors
        labels      : [N] integer class label array
        label_names : optional mapping from int to display name
        method      : 'tsne' (default), 'pca', or 'umap'
        title       : figure title; auto-generated if None
        save_path   : file path to save; if None, calls plt.show()
    """
    reduced = _reduce_embeddings(embeddings, method)
    if title is None:
        title = f'{method.upper()} Visualization of Embeddings'

    display_labels = ([label_names[int(l)] for l in labels]
                      if label_names else [str(l) for l in labels])
    df = pd.DataFrame({'x': reduced[:, 0], 'y': reduced[:, 1], 'label': display_labels})

    fig, ax = plt.subplots(figsize=(10, 8))
    unique = df['label'].unique()
    palette = dict(zip(unique, sns.color_palette('tab10', len(unique))))
    sns.scatterplot(x='x', y='y', hue='label', data=df, palette=palette,
                    alpha=0.7, s=60, edgecolor='w', linewidth=0.3, ax=ax)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    _save_or_show(save_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _reduce_embeddings(embeddings: np.ndarray, method: str) -> np.ndarray:
    method = method.lower()
    if method == 'tsne':
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, random_state=42).fit_transform(embeddings)
    elif method == 'pca':
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=42).fit_transform(embeddings)
    elif method == 'umap':
        import umap as umap_lib
        return umap_lib.UMAP(n_components=2, random_state=42).fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'tsne', 'pca', or 'umap'.")


def _save_or_show(save_path: Optional[str]) -> None:
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Saved -> {save_path}")
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='HotPatchCaps visualization tools')
    p.add_argument('--results', type=str,
                   help='Path to a results JSON file produced by evaluate()')
    p.add_argument('--output_dir', type=str, default='./figures',
                   help='Directory for saved figures (default: ./figures)')
    p.add_argument('--vis_method', type=str, default='tsne',
                   choices=['tsne', 'pca', 'umap'],
                   help='Dimensionality reduction method for embedding scatter')
    return p


if __name__ == '__main__':
    import json

    args = _build_arg_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.results:
        with open(args.results) as f:
            results = json.load(f)

        cm = np.array(results.get('confusion', []))
        if cm.size:
            plot_confusion_matrix(
                cm,
                labels=results.get('label_names'),
                title='Evaluation Confusion Matrix',
                save_path=os.path.join(args.output_dir, 'confusion_matrix.png'),
            )

        if 'train_loss' in results and 'eval_acc' in results:
            plot_training_curves(
                results['train_loss'],
                results['eval_acc'],
                save_path=os.path.join(args.output_dir, 'training_curves.png'),
            )
