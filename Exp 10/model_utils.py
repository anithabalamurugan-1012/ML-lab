"""
model_utils.py
==============
Model training, evaluation, and inference utilities for Diabetic Retinopathy screening:
- Stratified train/test split with fixed random seed.
- RandomForestClassifier with balanced class weighting.
- Performance evaluation: Hold-out accuracy, macro F1 score, per-class report, confusion matrix.
- Single-image inference and severity score generation.
"""

from typing import Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

CLASS_LABELS: Dict[int, str] = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

CLASS_DESCRIPTIONS: Dict[int, str] = {
    0: "Normal fundus with no visible diabetic microvascular abnormalities.",
    1: "Presence of microaneurysms only (small outpouchings of capillary walls).",
    2: "More than microaneurysms but less than severe: hard exudates, cotton-wool spots, venous caliber changes.",
    3: "Severe non-proliferative changes (>20 intraretinal hemorrhages in each of 4 quadrants, venous beading, or IRMA).",
    4: "Neovascularization (growth of abnormal, fragile new blood vessels) or vitreous/preretinal hemorrhage."
}


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.25,
    random_state: int = 42,
    n_estimators: int = 100
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Trains a balanced Random Forest classifier using a stratified train/test split.
    
    Parameters:
        X: Feature matrix of shape (n_samples, n_features).
        y: Class label vector of shape (n_samples,).
        test_size: Proportion of the dataset to reserve for test split (default 0.25).
        random_state: Seed for reproducibility.
        n_estimators: Number of trees in the forest.
        
    Returns:
        Tuple:
        - model: Fitted RandomForestClassifier
        - eval_results: Dictionary containing accuracy, macro_f1, report, confusion matrix, and split details.
    """
    unique_classes, counts = np.unique(y, return_counts=True)
    class_counts_dict = {int(c): int(cnt) for c, cnt in zip(unique_classes, counts)}
    
    # Check if any class has fewer than 2 samples (required for stratified split)
    can_stratify = all(cnt >= 2 for cnt in counts)
    stratify_param = y if can_stratify else None
    
    # If stratified split is not possible due to ultra-small count, fallback cleanly
    # but document the note in results
    stratify_note = (
        "Stratified split applied (maintains equal class proportions in train & test)."
        if can_stratify
        else "Stratification disabled: at least one class has < 2 samples."
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )

    # Train Random Forest with class_weight='balanced' to handle any class imbalance
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Evaluate on hold-out test set
    y_pred = clf.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    
    # Generate per-class classification report
    report_dict = classification_report(
        y_test, y_pred,
        labels=sorted(list(unique_classes)),
        target_names=[f"{c} ({CLASS_LABELS[c]})" for c in sorted(list(unique_classes))],
        output_dict=True,
        zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=sorted(list(unique_classes)))

    # Train and test class distributions
    train_dist = {int(c): int(np.sum(y_train == c)) for c in unique_classes}
    test_dist = {int(c): int(np.sum(y_test == c)) for c in unique_classes}

    eval_results = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "report_dict": report_dict,
        "confusion_matrix": cm,
        "classes_present": [int(c) for c in unique_classes],
        "class_counts_total": class_counts_dict,
        "train_counts": train_dist,
        "test_counts": test_dist,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "stratify_note": stratify_note,
        "can_stratify": can_stratify,
        "y_test": y_test,
        "y_pred": y_pred,
    }

    return clf, eval_results


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: list,
    title: str = "Diabetic Retinopathy Confusion Matrix"
) -> plt.Figure:
    """
    Renders an aesthetically styled confusion matrix heatmap using Matplotlib.
    """
    fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#1E293B")

    # Create image plot with modern teal/cyan colormap
    cax = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="#94A3B8")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#CBD5E1")

    tick_labels = [f"Class {c}\n({CLASS_LABELS.get(c, str(c))})" for c in classes]
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(tick_labels, rotation=35, ha="right", color="#CBD5E1", fontsize=8)
    ax.set_yticklabels(tick_labels, color="#CBD5E1", fontsize=8)

    ax.set_title(title, color="#F8FAFC", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", color="#94A3B8", fontsize=10, labelpad=8)
    ax.set_ylabel("True Label", color="#94A3B8", fontsize=10, labelpad=8)

    # Annotate numbers in cells
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "#FFFFFF" if val > thresh else "#CBD5E1"
            ax.text(
                j, i, format(val, "d"),
                ha="center", va="center",
                color=color,
                fontsize=11,
                fontweight="bold"
            )

    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    plt.tight_layout()
    return fig


def predict_single_image(
    model: RandomForestClassifier,
    features: np.ndarray,
    classes_present: list
) -> Dict[str, Any]:
    """
    Predicts the severity class and score distribution for a single preprocessed fundus image.
    
    Parameters:
        model: Trained RandomForestClassifier
        features: 1D feature vector of length n_features.
        classes_present: List of class IDs model was trained on.
        
    Returns:
        Dictionary containing predicted class, class name, scores per class, and explanation.
    """
    feat_2d = features.reshape(1, -1)
    predicted_class = int(model.predict(feat_2d)[0])
    
    # Random forest tree vote proportion across classes
    probabilities = model.predict_proba(feat_2d)[0]
    
    # Map probability to class IDs
    class_scores = {}
    for idx, c in enumerate(model.classes_):
        class_scores[int(c)] = float(probabilities[idx])

    # Fill 0.0 for any absent classes
    for c in range(5):
        if c not in class_scores:
            class_scores[c] = 0.0

    top_score = class_scores.get(predicted_class, 0.0)

    return {
        "predicted_class": predicted_class,
        "predicted_name": CLASS_LABELS.get(predicted_class, f"Class {predicted_class}"),
        "predicted_description": CLASS_DESCRIPTIONS.get(predicted_class, ""),
        "predicted_score": top_score,
        "all_scores": class_scores,
    }
