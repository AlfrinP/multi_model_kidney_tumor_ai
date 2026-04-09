import argparse
import json
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from model_utils import load_models, predict_probability_mask


def compute_confusion(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tn, fp, fn, tp


def bce_loss(prob, target, eps=1e-7):
    prob = np.clip(prob, eps, 1.0 - eps)
    return float(-np.mean(target * np.log(prob) + (1.0 - target) * np.log(1.0 - prob)))


def roc_curve_from_scores(y_true, y_score, num_thresholds=101):
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    tprs, fprs = [], []
    for t in thresholds:
        y_pred = (y_score >= t).astype(np.uint8)
        tn, fp, fn, tp = compute_confusion(y_true, y_pred)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tprs.append(tpr)
        fprs.append(fpr)
    return np.array(fprs), np.array(tprs)


def auc_trapezoid(fpr, tpr):
    order = np.argsort(fpr)
    return float(np.trapezoid(tpr[order], fpr[order]))


def _find_mri_layout(dataset_dir):
    # Layout A: dataset/mri/*.png + dataset/mask/*.png (same filenames)
    # Layout B: dataset/mri/image/img_XXXXX.png + dataset/mri/mask/mask_XXXXX.png
    mri_root = os.path.join(dataset_dir, "mri")
    image_subdir = os.path.join(mri_root, "image")
    mask_subdir = os.path.join(mri_root, "mask")
    if os.path.isdir(image_subdir) and os.path.isdir(mask_subdir):
        return image_subdir, mask_subdir, "nested"
    return mri_root, os.path.join(dataset_dir, "mask"), "flat"


def _mri_mask_name(image_name, layout_kind):
    if layout_kind == "nested":
        image_path = Path(image_name)
        if image_path.name.startswith("img_"):
            return str(image_path.with_name("mask_" + image_path.name[4:]))
    return image_name


def _ct_mask_name(image_name, layout_kind):
    if layout_kind == "casewise":
        image_path = Path(image_name)
        parts = list(image_path.parts)
        if "imaging" in parts:
            parts[parts.index("imaging")] = "segmentation"
            return str(Path(*parts))
    return image_name


def _iter_relative_files(root_dir):
    root_path = Path(root_dir)
    if not root_path.is_dir():
        return []

    files = []
    for p in sorted(root_path.rglob("*")):
        if p.is_file():
            files.append(str(p.relative_to(root_path)))
    return files


def _find_ct_only_layout(dataset_dir):
    # Layout A (default): dataset/ct/... + dataset/mask/...
    ct_dir = os.path.join(dataset_dir, "ct")
    mask_dir = os.path.join(dataset_dir, "mask")
    if os.path.isdir(ct_dir) and os.path.isdir(mask_dir):
        return ct_dir, mask_dir, "flat"

    # Layout B: dataset/ct/image/... + dataset/ct/mask/...
    ct_root = os.path.join(dataset_dir, "ct")
    image_subdir = os.path.join(ct_root, "image")
    mask_subdir = os.path.join(ct_root, "mask")
    if os.path.isdir(image_subdir) and os.path.isdir(mask_subdir):
        return image_subdir, mask_subdir, "flat"

    # Layout C: dataset/image/... + dataset/mask/... (dataset points at ct/)
    image_subdir = os.path.join(dataset_dir, "image")
    mask_subdir = os.path.join(dataset_dir, "mask")
    if os.path.isdir(image_subdir) and os.path.isdir(mask_subdir):
        return image_subdir, mask_subdir, "flat"

    # Layout D: dataset/ct/case_x/imaging/*.png + dataset/ct/case_x/segmentation/*.png
    if os.path.isdir(ct_dir):
        case_dirs = [p for p in Path(ct_dir).iterdir() if p.is_dir()]
        if case_dirs and any((p / "imaging").is_dir() and (p / "segmentation").is_dir() for p in case_dirs):
            return ct_dir, ct_dir, "casewise"

    # Layout E: dataset/case_x/imaging/*.png + dataset/case_x/segmentation/*.png
    if os.path.isdir(dataset_dir):
        case_dirs = [p for p in Path(dataset_dir).iterdir() if p.is_dir()]
        if case_dirs and any((p / "imaging").is_dir() and (p / "segmentation").is_dir() for p in case_dirs):
            return dataset_dir, dataset_dir, "casewise"

    return ct_dir, mask_dir, "flat"


def _list_samples(dataset_dir, modality):
    ct_dir = os.path.join(dataset_dir, "ct")
    mri_dir, mri_mask_dir, mri_layout_kind = _find_mri_layout(dataset_dir)
    default_mask_dir = os.path.join(dataset_dir, "mask")

    if modality == "fused":
        mask_dir = default_mask_dir
        if not os.path.isdir(mask_dir):
            raise RuntimeError(f"Missing directory: {mask_dir}")
        if not os.path.isdir(ct_dir):
            raise RuntimeError(f"Missing directory: {ct_dir}")
        if not os.path.isdir(mri_dir):
            raise RuntimeError(f"Missing directory: {mri_dir}")
        names = []
        for rel_path in _iter_relative_files(ct_dir):
            if os.path.isfile(os.path.join(mri_dir, rel_path)) and os.path.isfile(
                os.path.join(mask_dir, rel_path)
            ):
                names.append(rel_path)
        return names, ct_dir, mri_dir, mask_dir, mri_layout_kind

    if modality == "ct":
        ct_only_dir, ct_only_mask_dir, ct_layout_kind = _find_ct_only_layout(dataset_dir)
        if not os.path.isdir(ct_only_mask_dir):
            raise RuntimeError(f"Missing directory: {ct_only_mask_dir}")
        if not os.path.isdir(ct_only_dir):
            raise RuntimeError(f"Missing directory: {ct_only_dir}")
        names = []
        for rel_path in _iter_relative_files(ct_only_dir):
            if ct_layout_kind == "casewise" and "imaging" not in Path(rel_path).parts:
                continue
            mask_name = _ct_mask_name(rel_path, ct_layout_kind)
            if os.path.isfile(os.path.join(ct_only_mask_dir, mask_name)):
                names.append(rel_path)
        return names, ct_only_dir, None, ct_only_mask_dir, ct_layout_kind

    if modality == "mri":
        if not os.path.isdir(mri_dir):
            raise RuntimeError(f"Missing directory: {mri_dir}")
        if not os.path.isdir(mri_mask_dir):
            raise RuntimeError(f"Missing directory: {mri_mask_dir}")
        names = []
        for rel_path in _iter_relative_files(mri_dir):
            mask_name = _mri_mask_name(rel_path, mri_layout_kind)
            if os.path.isfile(os.path.join(mri_mask_dir, mask_name)):
                names.append(rel_path)
        return names, ct_dir, mri_dir, mri_mask_dir, mri_layout_kind

    raise ValueError(f"Unknown modality: {modality}")
def evaluate(dataset_dir, output_dir, threshold=0.5, modality="fused"):
    modality = modality.lower()
    if modality not in ("fused", "ct", "mri"):
        raise ValueError("--modality must be one of: fused, ct, mri")

    os.makedirs(output_dir, exist_ok=True)
    file_names, ct_dir, mri_dir, mask_dir, mri_layout_kind = _list_samples(dataset_dir, modality)

    if not file_names:
        raise RuntimeError(
            f"No valid samples found for modality={modality}. "
            "Check ct/, mri/, mask/ and matching filenames."
        )

    ct_model, mri_model = load_models()

    all_true = []
    all_score = []
    all_pred = []
    losses = []

    for name in file_names:
        if modality == "mri":
            gt_name = _mri_mask_name(name, mri_layout_kind)
        elif modality == "ct":
            gt_name = _ct_mask_name(name, mri_layout_kind)
        else:
            gt_name = name
        gt_path = os.path.join(mask_dir, gt_name)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            continue

        gt = cv2.resize(gt, (256, 256), interpolation=cv2.INTER_NEAREST)
        gt_bin = (gt > 127).astype(np.uint8)

        if modality == "fused":
            ct_path = os.path.join(ct_dir, name)
            mri_path = os.path.join(mri_dir, name)
            ct_img = cv2.imread(ct_path, cv2.IMREAD_GRAYSCALE)
            mri_img = cv2.imread(mri_path, cv2.IMREAD_GRAYSCALE)
            if ct_img is None or mri_img is None:
                continue
            ct_prob = predict_probability_mask(ct_model, ct_img)
            mri_prob = predict_probability_mask(mri_model, mri_img)
            prob = np.clip((ct_prob + mri_prob) / 2.0, 0.0, 1.0)
        elif modality == "ct":
            ct_path = os.path.join(ct_dir, name)
            ct_img = cv2.imread(ct_path, cv2.IMREAD_GRAYSCALE)
            if ct_img is None:
                continue
            prob = predict_probability_mask(ct_model, ct_img)
        else:
            mri_path = os.path.join(mri_dir, name)
            mri_img = cv2.imread(mri_path, cv2.IMREAD_GRAYSCALE)
            if mri_img is None:
                continue
            prob = predict_probability_mask(mri_model, mri_img)

        pred = (prob >= threshold).astype(np.uint8)
        all_true.append(gt_bin.flatten())
        all_score.append(prob.flatten())
        all_pred.append(pred.flatten())
        losses.append(bce_loss(prob, gt_bin))

    if not all_true:
        raise RuntimeError("No samples produced valid predictions.")

    label = {"fused": "Fused CT+MRI", "ct": "CT", "mri": "MRI"}[modality]

    y_true = np.concatenate(all_true)
    y_score = np.concatenate(all_score)
    y_pred = np.concatenate(all_pred)

    tn, fp, fn, tp = compute_confusion(y_true, y_pred)
    accuracy = float((tp + tn) / max(tp + tn + fp + fn, 1))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0
    dice = float((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    mean_loss = float(np.mean(losses))

    # Confusion matrix figure.
    cm = np.array([[tn, fp], [fn, tp]])
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix — {label} (Pixel-level)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["No Tumor", "Tumor"])
    plt.yticks([0, 1], ["No Tumor", "Tumor"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    # ROC figure.
    fpr, tpr = roc_curve_from_scores(y_true, y_score)
    roc_auc = auc_trapezoid(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"ROC Curve — {label} (Pixel-level)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150)
    plt.close()

    # Loss graph (sample-wise BCE loss).
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, len(losses) + 1), losses, marker="o")
    plt.title(f"Loss Graph — {label} (Per-sample BCE)")
    plt.xlabel("Sample Index")
    plt.ylabel("BCE Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_graph.png"), dpi=150)
    plt.close()

    summary = {
        "modality": modality,
        "num_samples": len(losses),
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "dice": dice,
        "mean_bce_loss": mean_loss,
        "roc_auc": roc_auc,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
    with open(os.path.join(output_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Evaluation finished.")
    print(json.dumps(summary, indent=2))
    print(f"Artifacts saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate report metrics and plots.")
    parser.add_argument("--dataset-dir", required=True, help="Folder containing ct/, mri/, and/or mask/")
    parser.add_argument("--output-dir", default="report_evaluation/output")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--modality",
        choices=("fused", "ct", "mri"),
        default="fused",
        help="fused: CT+MRI average prob (needs ct+mri+mask). ct/mri: single-modality + mask only.",
    )
    args = parser.parse_args()
    evaluate(args.dataset_dir, args.output_dir, args.threshold, args.modality)


if __name__ == "__main__":
    main()
