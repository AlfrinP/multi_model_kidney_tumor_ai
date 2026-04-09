# Report Evaluation Kit

This folder provides ready-to-run scripts to generate report artifacts for:

- Accuracy
- Loss graph
- Confusion matrix
- ROC curve
- Time complexity (empirical runtime)
- Resource usage (RAM and GPU memory, if CUDA is available)

## Folder Structure

### Fused evaluation (CT + MRI + mask, same filename per case)

```text
your_dataset/
  ct/
    case_001.png
  mri/
    case_001.png
  mask/
    case_001.png
```

### CT-only or MRI-only (when you do not have paired CT–MRI)

Use **`--modality ct`**: only **`ct/`** and **`mask/`** are required (same filenames). An `mri/` folder may be absent.

Use **`--modality mri`**: only **`mri/`** and **`mask/`** are required. A `ct/` folder may be absent.

File names must match between the **modality folder you use** and **`mask/`**.
Nested case-wise folders are also supported, and matching is done by relative path
(for example: `ct/case_001/slice_014.png` matches `mask/case_001/slice_014.png`).

CT-only also supports case folders that contain both image and label slices:
`ct/case_001/imaging/*.png` is matched to `ct/case_001/segmentation/*.png`.

For MRI-only datasets, this script also supports a nested layout:

```text
your_dataset/
  mri/
    image/
      img_00000.png
    mask/
      mask_00000.png
```

`img_XXXXX` is automatically mapped to `mask_XXXXX`.

## Scripts

- `generate_dummy_dataset.py`  
  Creates a synthetic dataset (useful for testing the pipeline quickly).

- `evaluate_metrics.py`  
  Runs inference and generates:
  - `metrics_summary.json` (accuracy, precision, recall, IoU, Dice, mean loss, ROC-AUC, `modality` field)
  - `loss_graph.png`
  - `confusion_matrix.png`
  - `roc_curve.png`

  **`--modality`**: `fused` (default, average CT+MRI probabilities), `ct`, or `mri`.

- `benchmark_runtime.py`  
  Measures runtime/resource:
  - `runtime_summary.json` (includes `modality`, avg/p95 inference time, max RSS, GPU peak memory)
  - `runtime_per_sample.csv`

  **`--modality`**: `fused` (both models + fusion), `ct`, or `mri` (single forward pass).

## How To Run

From project root (use `.venv/bin/python` if that is where dependencies are installed):

```bash
# Optional: synthetic test data (includes ct/, mri/, mask/)
python report_evaluation/generate_dummy_dataset.py --output-dir report_evaluation/sample_data --samples 20

# Fused metrics (needs ct + mri + mask per name)
python report_evaluation/evaluate_metrics.py \
  --dataset-dir report_evaluation/sample_data \
  --output-dir report_evaluation/output_fused \
  --modality fused \
  --threshold 0.5

# CT-only cohort (ct + mask only)
python report_evaluation/evaluate_metrics.py \
  --dataset-dir /path/to/ct_mask_dataset \
  --output-dir report_evaluation/output_ct \
  --modality ct

# MRI-only cohort (mri + mask only)
python report_evaluation/evaluate_metrics.py \
  --dataset-dir /path/to/mri_mask_dataset \
  --output-dir report_evaluation/output_mri \
  --modality mri

# Runtime: single modality or fused
python report_evaluation/benchmark_runtime.py \
  --dataset-dir report_evaluation/sample_data \
  --output-dir report_evaluation/output_benchmark_fused \
  --modality fused \
  --repeats 3

python report_evaluation/benchmark_runtime.py \
  --dataset-dir /path/to/ct_only_images \
  --output-dir report_evaluation/output_benchmark_ct \
  --modality ct
```

## Notes For Final Report

- This project is segmentation-based; include **IoU and Dice** alongside accuracy.
- `loss_graph.png` is **evaluation BCE per sample**, not training epochs.
- **CT-only** and **MRI-only** metrics describe **separate cohorts**; only use **fused** metrics when CT, MRI, and mask refer to the **same registered case**.
