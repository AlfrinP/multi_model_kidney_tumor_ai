import argparse
import os

import cv2
import numpy as np


def _ensure_dirs(base_dir):
    for folder in ("ct", "mri", "mask"):
        os.makedirs(os.path.join(base_dir, folder), exist_ok=True)


def _make_sample(seed):
    rng = np.random.default_rng(seed)
    h, w = 256, 256
    ct = np.zeros((h, w), dtype=np.uint8)
    mri = np.zeros((h, w), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)

    center = (rng.integers(80, 176), rng.integers(80, 176))
    axes = (rng.integers(20, 50), rng.integers(15, 45))
    angle = int(rng.integers(0, 180))
    has_tumor = rng.random() > 0.25

    # Draw synthetic kidney-like background texture.
    cv2.rectangle(ct, (40, 40), (216, 216), 70, thickness=-1)
    cv2.rectangle(mri, (40, 40), (216, 216), 60, thickness=-1)

    if has_tumor:
        cv2.ellipse(mask, center, axes, angle, 0, 360, 255, thickness=-1)
        # Tumor appears brighter in both modalities with different intensities.
        ct = np.clip(ct + (mask > 0) * 90, 0, 255).astype(np.uint8)
        mri = np.clip(mri + (mask > 0) * 120, 0, 255).astype(np.uint8)

    noise_ct = rng.normal(0, 8, size=(h, w))
    noise_mri = rng.normal(0, 10, size=(h, w))
    ct = np.clip(ct + noise_ct, 0, 255).astype(np.uint8)
    mri = np.clip(mri + noise_mri, 0, 255).astype(np.uint8)
    return ct, mri, mask


def main():
    parser = argparse.ArgumentParser(description="Generate a small synthetic CT/MRI/mask dataset.")
    parser.add_argument("--output-dir", default="report_evaluation/sample_data")
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()

    _ensure_dirs(args.output_dir)
    for i in range(args.samples):
        ct, mri, mask = _make_sample(seed=100 + i)
        name = f"sample_{i:03d}.png"
        cv2.imwrite(os.path.join(args.output_dir, "ct", name), ct)
        cv2.imwrite(os.path.join(args.output_dir, "mri", name), mri)
        cv2.imwrite(os.path.join(args.output_dir, "mask", name), mask)

    print(f"Created {args.samples} samples in: {args.output_dir}")


if __name__ == "__main__":
    main()
