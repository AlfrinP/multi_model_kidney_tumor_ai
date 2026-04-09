import argparse
import json
import os
import time

import cv2
import numpy as np
import torch

from model_utils import load_models, predict_probability_mask


def max_rss_mb():
    # Linux ru_maxrss is in kilobytes.
    import resource

    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)


def _find_mri_image_dir(dataset_dir):
    mri_root = os.path.join(dataset_dir, "mri")
    nested_image = os.path.join(mri_root, "image")
    if os.path.isdir(nested_image):
        return nested_image
    return mri_root


def list_benchmark_samples(dataset_dir, modality):
    ct_dir = os.path.join(dataset_dir, "ct")
    mri_dir = _find_mri_image_dir(dataset_dir)

    if modality == "fused":
        names = []
        if not os.path.isdir(ct_dir) or not os.path.isdir(mri_dir):
            return names
        for f in sorted(os.listdir(ct_dir)):
            if os.path.isfile(os.path.join(ct_dir, f)) and os.path.isfile(os.path.join(mri_dir, f)):
                names.append(f)
        return names

    if modality == "ct":
        if not os.path.isdir(ct_dir):
            return []
        return sorted(
            f for f in os.listdir(ct_dir) if os.path.isfile(os.path.join(ct_dir, f))
        )

    if modality == "mri":
        if not os.path.isdir(mri_dir):
            return []
        return sorted(
            f for f in os.listdir(mri_dir) if os.path.isfile(os.path.join(mri_dir, f))
        )

    raise ValueError(f"Unknown modality: {modality}")


def run_benchmark(dataset_dir, output_dir, repeats=3, modality="fused"):
    modality = modality.lower()
    if modality not in ("fused", "ct", "mri"):
        raise ValueError("--modality must be one of: fused, ct, mri")

    os.makedirs(output_dir, exist_ok=True)
    ct_model, mri_model = load_models()
    mri_dir = _find_mri_image_dir(dataset_dir)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    names = list_benchmark_samples(dataset_dir, modality)
    if not names:
        raise RuntimeError(f"No input files found for modality={modality}.")

    per_sample = []
    all_times = []

    for name in names:
        times = []
        for _ in range(repeats):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            if modality == "fused":
                ct = cv2.imread(os.path.join(dataset_dir, "ct", name), cv2.IMREAD_GRAYSCALE)
                mri = cv2.imread(os.path.join(mri_dir, name), cv2.IMREAD_GRAYSCALE)
                if ct is None or mri is None:
                    times.clear()
                    break
                ct_prob = predict_probability_mask(ct_model, ct)
                mri_prob = predict_probability_mask(mri_model, mri)
                _ = np.logical_or(ct_prob >= 0.5, mri_prob >= 0.5)
            elif modality == "ct":
                ct = cv2.imread(os.path.join(dataset_dir, "ct", name), cv2.IMREAD_GRAYSCALE)
                if ct is None:
                    times.clear()
                    break
                _ = predict_probability_mask(ct_model, ct)
            else:
                mri = cv2.imread(os.path.join(mri_dir, name), cv2.IMREAD_GRAYSCALE)
                if mri is None:
                    times.clear()
                    break
                _ = predict_probability_mask(mri_model, mri)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

        if not times:
            continue

        sample_ms = float(np.mean(times))
        all_times.extend(times)
        per_sample.append({"sample": name, "avg_inference_ms": sample_ms})

    if not per_sample:
        raise RuntimeError("Benchmark produced no timings (check images readable).")

    gpu_mem_mb = 0.0
    if torch.cuda.is_available():
        gpu_mem_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2))

    summary = {
        "modality": modality,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_samples": len(per_sample),
        "repeats_per_sample": repeats,
        "avg_inference_ms": float(np.mean(all_times)),
        "std_inference_ms": float(np.std(all_times)),
        "p95_inference_ms": float(np.percentile(all_times, 95)),
        "max_rss_mb": max_rss_mb(),
        "peak_gpu_memory_mb": gpu_mem_mb,
    }

    with open(os.path.join(output_dir, "runtime_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(output_dir, "runtime_per_sample.csv"), "w", encoding="utf-8") as f:
        f.write("sample,avg_inference_ms\n")
        for row in per_sample:
            f.write(f"{row['sample']},{row['avg_inference_ms']:.4f}\n")

    print("Benchmark finished.")
    print(json.dumps(summary, indent=2))
    print(f"Artifacts saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference time and resource usage.")
    parser.add_argument("--dataset-dir", required=True, help="Folder containing ct/ and/or mri/")
    parser.add_argument("--output-dir", default="report_evaluation/output")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--modality",
        choices=("fused", "ct", "mri"),
        default="fused",
        help="Which inference path to time: one model (ct/mri) or both fused (fused).",
    )
    args = parser.parse_args()
    run_benchmark(args.dataset_dir, args.output_dir, args.repeats, args.modality)


if __name__ == "__main__":
    main()
