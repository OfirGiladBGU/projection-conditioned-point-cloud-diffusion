"""Analyze binary target images for average black pixel counts.

This script scans a target directory of binary black/white images,
counts black pixels per image (with configurable threshold), and
reports summary statistics. It defaults to the target directory
from the local experiments `config.py`.

Usage examples:
    python analyze_black_points.py
    python analyze_black_points.py --threshold 0 --limit 100
    python analyze_black_points.py --target-dir /path/to/targets --output-json outputs_pointdit_v5/black_points_stats.json
"""

import argparse
import json
import os
import sys
from typing import List, Optional, Tuple

import numpy as np


def _load_image_grayscale(path: str) -> np.ndarray:
    """Load an image file and return it as a grayscale numpy array (uint8).

    Tries Pillow first; falls back to OpenCV, then imageio.
    """
    # 1) Pillow
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as img:
            img = img.convert("L")
            arr = np.array(img, dtype=np.uint8)
        return arr
    except Exception:
        pass

    # 2) OpenCV
    try:
        import cv2  # type: ignore

        arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise RuntimeError("cv2.imread returned None")
        return arr.astype(np.uint8)
    except Exception:
        pass

    # 3) imageio
    try:
        import imageio.v3 as iio  # type: ignore

        arr = iio.imread(path)
        # If multi-channel, convert to grayscale via luminance approximation
        if arr.ndim == 3:
            # Assume RGB[A]; ignore alpha if present
            arr = arr[..., :3]
            # Luminance weights: 0.2126 R + 0.7152 G + 0.0722 B
            arr = (
                0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
            ).astype(np.uint8)
        elif arr.ndim == 2:
            arr = arr.astype(np.uint8)
        else:
            raise RuntimeError(f"Unsupported image shape: {arr.shape}")
        return arr
    except Exception as e:
        raise RuntimeError(
            f"Failed to load image {path}: {e}. Install one of Pillow, opencv-python, or imageio."
        )


def _list_images(target_dir: str, recursive: bool = True) -> List[str]:
    """List image files in a directory, optionally recursively, using common extensions."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    image_paths: List[str] = []
    if recursive:
        for root, _, files in os.walk(target_dir):
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                if ext in exts:
                    image_paths.append(os.path.join(root, fn))
    else:
        for fn in os.listdir(target_dir):
            p = os.path.join(target_dir, fn)
            if os.path.isfile(p):
                ext = os.path.splitext(fn)[1].lower()
                if ext in exts:
                    image_paths.append(p)
    image_paths.sort()
    return image_paths


def _count_black_pixels(gray: np.ndarray, threshold: int) -> Tuple[int, int]:
    """Count black pixels in a grayscale image using a threshold.

    Returns:
        black_count: number of pixels where value <= threshold
        total_pixels: total number of pixels in the image
    """
    mask = gray <= threshold
    black_count = int(np.count_nonzero(mask))
    total_pixels = int(gray.size)
    return black_count, total_pixels


def analyze_directory(
    target_dir: str,
    threshold: int = 0,
    limit: Optional[int] = None,
    recursive: bool = True,
) -> dict:
    """Analyze a directory of images and compute summary stats for black pixels."""
    images = _list_images(target_dir, recursive=recursive)
    if not images:
        raise FileNotFoundError(f"No image files found under: {target_dir}")

    if limit is not None:
        images = images[: max(0, int(limit))]

    black_counts: List[int] = []
    totals: List[int] = []
    dims: List[Tuple[int, int]] = []

    for idx, path in enumerate(images):
        try:
            gray = _load_image_grayscale(path)
            bc, tot = _count_black_pixels(gray, threshold)
            black_counts.append(bc)
            totals.append(tot)
            dims.append((int(gray.shape[0]), int(gray.shape[1])))
        except Exception as e:
            print(f"[warn] Skipping {path}: {e}")

    if not black_counts:
        raise RuntimeError("No images were successfully processed.")

    black_arr = np.array(black_counts, dtype=np.int64)
    tot_arr = np.array(totals, dtype=np.int64)
    frac_arr = black_arr / tot_arr

    # Aggregate stats
    stats = {
        "target_dir": target_dir,
        "num_images": int(len(black_counts)),
        "threshold": int(threshold),
        "black_pixels": {
            "mean": float(black_arr.mean()),
            "median": float(np.median(black_arr)),
            "min": int(black_arr.min()),
            "max": int(black_arr.max()),
        },
        "black_fraction": {
            "mean": float(frac_arr.mean()),
            "median": float(np.median(frac_arr)),
            "min": float(frac_arr.min()),
            "max": float(frac_arr.max()),
        },
        "image_dimensions_sample": dims[:10],  # first 10 dims as a quick check
    }

    return stats


def _default_target_dir_from_config() -> Optional[str]:
    """Attempt to load the default target_dir from local experiments config.py."""
    try:
        # The script resides alongside config.py in experiments_pointdit_v5
        from config import Config  # type: ignore

        return Config.default().data.target_dir
    except Exception:
        return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute average black pixels in binary BW target images and summarize."
        )
    )
    default_target = _default_target_dir_from_config()
    parser.add_argument(
        "--target-dir",
        type=str,
        default=default_target,
        help=(
            "Directory containing target images (defaults to experiments config target_dir)."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=0,
        help=(
            "Pixel value threshold for black (<= threshold counts as black). "
            "For strictly binary images, 0 is typical; for noisy images, try 10-20 or 127."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of images processed (useful for quick checks).",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Do not scan subdirectories; only process images in the top-level directory.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs_pointdit_v5/black_points_stats.json",
        help="Optional path to write a JSON summary report.",
    )

    args = parser.parse_args(argv)

    if not args.target_dir:
        print(
            "[error] No target directory provided and config default could not be loaded.",
            file=sys.stderr,
        )
        return 2

    if not os.path.isdir(args.target_dir):
        print(f"[error] Target directory does not exist: {args.target_dir}", file=sys.stderr)
        return 2

    stats = analyze_directory(
        target_dir=args.target_dir,
        threshold=int(args.threshold),
        limit=args.limit,
        recursive=(not args.non_recursive),
    )

    # Print a concise summary
    print("\n=== Black Pixel Analysis Summary ===")
    print(f"Target dir: {stats['target_dir']}")
    print(f"Images processed: {stats['num_images']}")
    print(f"Threshold: {stats['threshold']}")
    bp = stats["black_pixels"]
    bf = stats["black_fraction"]
    print(
        f"Black pixels -> mean: {bp['mean']:.2f}, median: {bp['median']:.2f}, "
        f"min: {bp['min']}, max: {bp['max']}"
    )
    print(
        f"Black fraction -> mean: {bf['mean']:.6f}, median: {bf['median']:.6f}, "
        f"min: {bf['min']:.6f}, max: {bf['max']:.6f}"
    )

    # Optionally write JSON
    out_json = args.output_json
    if out_json:
        out_dir = os.path.dirname(out_json)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"\nSummary written to: {out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
