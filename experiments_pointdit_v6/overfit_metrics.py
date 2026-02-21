"""Overfit-specific stippling metrics and visualization for Point-DiT V5.

All point coordinates are expected in [-1, 1] (the model's native space).
Internally converts to [0, 1] where needed for metric computation.

Provides three metrics:
  1. Grid Capacity  -- CCVT-style: do grid cells have the right point count?
  2. Spacing Quality -- Blue-noise check: are NN distances uniform, no clumping?
  3. Chamfer Distance -- Set-aware point cloud matching to GT

And a combined 3-row visualization:
  Row 0: Source image | GT scatter | Pred scatters ...
  Row 1: (empty)      | GT capacity | Pred capacities ...
  Row 2: (empty)      | GT spacing  | Pred spacings ...
"""

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    plt = None
    HAS_MPL = False


# ── coordinate helpers ───────────────────────────────────────────────

def _to_01(pts):
    """Convert points from [-1, 1] to [0, 1]."""
    return (pts + 1.0) / 2.0


# ── metric computations ─────────────────────────────────────────────

def compute_grid_capacity(points_01, image_01, grid_size=(16, 16)):
    """CCVT-style grid capacity fulfillment.

    Parameters
    ----------
    points_01 : ndarray (N, 2) in [0, 1]
    image_01 : ndarray (H, W) float in [0, 1]  (0=black/dark, 1=white/light)
    grid_size : (rows, cols)

    Returns
    -------
    dict with grid_status (rows, cols), score, underfilled_pct, overfilled_pct
    """
    from scipy.ndimage import zoom

    N = len(points_01)
    H_grid, W_grid = grid_size
    H_img, W_img = image_01.shape

    scale_h = H_grid / H_img
    scale_w = W_grid / W_img
    image_down = zoom(image_01, (scale_h, scale_w), order=1)

    expected_weight = 1.0 - image_down + 0.01
    expected_weight /= expected_weight.sum()
    grid_expected = expected_weight * N

    pts = np.clip(points_01, 0, 1 - 1e-6)
    col_idx = (pts[:, 0] * W_grid).astype(int)
    row_idx = (pts[:, 1] * H_grid).astype(int)

    grid_actual = np.zeros((H_grid, W_grid), dtype=np.float64)
    np.add.at(grid_actual, (row_idx, col_idx), 1)

    grid_ratio = grid_actual / (grid_expected + 1e-6)

    grid_status = np.zeros_like(grid_ratio, dtype=int)
    grid_status[grid_ratio < 0.5] = -1
    grid_status[grid_ratio > 2.0] = 1

    significant = grid_expected > 0.5
    n_sig = significant.sum()
    if n_sig > 0:
        ok = ((grid_status == 0) & significant).sum()
        under = ((grid_status == -1) & significant).sum()
        over = ((grid_status == 1) & significant).sum()
        score = ok / n_sig
        underfilled_pct = 100.0 * under / n_sig
        overfilled_pct = 100.0 * over / n_sig
    else:
        score, underfilled_pct, overfilled_pct = 1.0, 0.0, 0.0

    return {
        "grid_status": grid_status,
        "score": float(score),
        "underfilled_pct": float(underfilled_pct),
        "overfilled_pct": float(overfilled_pct),
    }


def compute_spacing_quality(points_01):
    """Nearest-neighbour spacing quality (blue-noise check).

    Parameters
    ----------
    points_01 : ndarray (N, 2) in [0, 1]

    Returns
    -------
    dict with nn_distances (N,), nn_cv, clumped_pct, spacing_score
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(points_01)
    nn_dists, _ = tree.query(points_01, k=2)
    nn_dists = nn_dists[:, 1]

    nn_mean = nn_dists.mean()
    nn_std = nn_dists.std()
    nn_cv = nn_std / (nn_mean + 1e-8)

    N = len(points_01)
    expected_nn = 0.5 * (1.0 / N) ** 0.5
    clump_threshold = 0.3 * expected_nn
    clumped_mask = nn_dists < clump_threshold
    clumped_pct = 100.0 * clumped_mask.mean()

    cv_penalty = min(nn_cv, 1.0)
    clump_penalty = clumped_pct / 100.0
    spacing_score = max(0.0, 1.0 - 0.5 * cv_penalty - 0.5 * clump_penalty)

    return {
        "nn_distances": nn_dists,
        "nn_mean": float(nn_mean),
        "nn_cv": float(nn_cv),
        "clumped_pct": float(clumped_pct),
        "spacing_score": float(spacing_score),
    }


def compute_chamfer_distance(pts_a, pts_b):
    """Chamfer distance between two point sets (squared L2).

    Parameters
    ----------
    pts_a, pts_b : ndarray (N, 2), (M, 2)

    Returns
    -------
    float : mean squared distance (A->B) + mean squared distance (B->A)
    """
    from scipy.spatial import cKDTree

    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)

    dist_a2b, _ = tree_b.query(pts_a, k=1)
    dist_b2a, _ = tree_a.query(pts_b, k=1)

    return float(np.mean(dist_a2b ** 2) + np.mean(dist_b2a ** 2))


# ── visualization ────────────────────────────────────────────────────

def visualize_overfit_metrics(
    source_img,
    gt_points,
    pred_pointsets,
    save_path,
    step=None,
    image_01=None,
):
    """Create 3-row comparison figure with metrics.

    All points are in [-1, 1].

    Layout (columns: INPUT | GT | Pred0 | Pred1 | ...):
      Row 0  Point clouds (INPUT shows source image, others show scatter)
      Row 1  Grid Capacity  (skip INPUT column)
      Row 2  Spacing Quality (skip INPUT column)

    Parameters
    ----------
    source_img : ndarray (H, W) uint8
    gt_points : ndarray (N, 2) in [-1, 1]
    pred_pointsets : list of ndarray (N, 2) in [-1, 1]
    save_path : str
    step : int or None
    image_01 : ndarray (H, W) float [0, 1], optional (auto from source_img)

    Returns
    -------
    str or None : save_path on success, None if matplotlib unavailable
    """
    if not HAS_MPL:
        return None

    n_preds = min(len(pred_pointsets), 4)
    n_cols = 2 + n_preds
    n_rows = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.5 * n_rows))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    step_label = f" (step {step})" if step is not None else ""

    if image_01 is None:
        image_01 = source_img.astype(np.float64) / 255.0

    gt_01 = _to_01(gt_points)

    # ── Row 0: point clouds ──────────────────────────────────────────
    ax = axes[0, 0]
    ax.imshow(source_img, cmap="gray", vmin=0, vmax=255)
    ax.set_title("Condition (Source)")
    ax.axis("off")

    gt_spa = compute_spacing_quality(gt_01)
    ax = axes[0, 1]
    ax.scatter(gt_points[:, 0], -gt_points[:, 1], c="black", s=0.5, alpha=0.8)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.set_title(f"GT ({len(gt_points)} pts)\nCV: {gt_spa['nn_cv']:.3f}")
    ax.axis("off")

    for i in range(n_preds):
        ax = axes[0, 2 + i]
        pts = pred_pointsets[i]
        pred_01 = _to_01(pts)
        chamfer = compute_chamfer_distance(pred_01, gt_01)
        ax.scatter(pts[:, 0], -pts[:, 1], c="black", s=0.5, alpha=0.8)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal")
        ax.set_facecolor("white")
        ax.set_title(f"Pred {i}{step_label}\nChamfer: {chamfer:.5f}")
        ax.axis("off")

    # ── compute metrics for GT + each prediction ─────────────────────
    all_points_01 = [gt_01] + [_to_01(pred_pointsets[i]) for i in range(n_preds)]
    all_cap = [compute_grid_capacity(p, image_01) for p in all_points_01]
    all_spa = [compute_spacing_quality(p) for p in all_points_01]

    # ── Row 1: grid capacity ─────────────────────────────────────────
    axes[1, 0].axis("off")

    col_labels = ["GT"] + [f"Pred {i}{step_label}" for i in range(n_preds)]
    for j, (cap, label) in enumerate(zip(all_cap, col_labels)):
        ax = axes[1, 1 + j]
        status = cap["grid_status"]
        H_g, W_g = status.shape
        rgb = np.zeros((H_g, W_g, 3), dtype=np.float32)
        rgb[status == 0, 1] = 1.0    # green = ok
        rgb[status == -1, 0] = 1.0   # red = underfilled
        rgb[status == 1, 2] = 1.0    # blue = overfilled

        ax.imshow(rgb, origin="lower", aspect="equal")
        ok_pct = 100.0 - cap["underfilled_pct"] - cap["overfilled_pct"]
        ax.set_title(
            f"{label} Capacity\n"
            f"OK:{ok_pct:.0f}% Under:{cap['underfilled_pct']:.0f}% "
            f"Over:{cap['overfilled_pct']:.0f}%\n"
            f"Score: {cap['score']:.3f}",
            fontsize=9,
        )
        ax.axis("off")

    # ── Row 2: spacing quality ───────────────────────────────────────
    axes[2, 0].axis("off")

    all_nn = [s["nn_distances"] for s in all_spa]
    vmin = min(d.min() for d in all_nn)
    vmax = max(d.max() for d in all_nn)

    for j, (spa, pts_01, label) in enumerate(
        zip(all_spa, all_points_01, col_labels)
    ):
        ax = axes[2, 1 + j]
        nn = spa["nn_distances"]
        pts_display = pts_01 * 2 - 1  # back to [-1, 1] for display
        sc = ax.scatter(
            pts_display[:, 0], -pts_display[:, 1],
            c=nn, cmap="RdYlBu", s=1.5, alpha=0.8, vmin=vmin, vmax=vmax,
        )
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal")
        ax.set_facecolor("white")
        ax.set_title(
            f"{label} Spacing\n"
            f"CV:{spa['nn_cv']:.3f}  Clumped:{spa['clumped_pct']:.1f}%\n"
            f"Score: {spa['spacing_score']:.3f}",
            fontsize=9,
        )
        ax.axis("off")
        plt.colorbar(sc, ax=ax, shrink=0.7, label="NN dist")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def collect_metrics_dict(gt_points, pred_points, image_01):
    """Compute all scalar metrics for a single GT/pred pair.

    Parameters
    ----------
    gt_points : ndarray (N, 2) in [-1, 1]
    pred_points : ndarray (N, 2) in [-1, 1]
    image_01 : ndarray (H, W) float in [0, 1]

    Returns
    -------
    dict with scalar metrics
    """
    gt_01 = _to_01(gt_points)
    pred_01 = _to_01(pred_points)

    cap_gt = compute_grid_capacity(gt_01, image_01)
    cap_pred = compute_grid_capacity(pred_01, image_01)
    spa_gt = compute_spacing_quality(gt_01)
    spa_pred = compute_spacing_quality(pred_01)
    chamfer = compute_chamfer_distance(pred_01, gt_01)

    return {
        "chamfer": chamfer,
        "gt_capacity_score": cap_gt["score"],
        "pred_capacity_score": cap_pred["score"],
        "pred_underfilled_pct": cap_pred["underfilled_pct"],
        "pred_overfilled_pct": cap_pred["overfilled_pct"],
        "gt_spacing_cv": spa_gt["nn_cv"],
        "pred_spacing_cv": spa_pred["nn_cv"],
        "gt_spacing_score": spa_gt["spacing_score"],
        "pred_spacing_score": spa_pred["spacing_score"],
        "pred_clumped_pct": spa_pred["clumped_pct"],
        "pred_nn_mean": spa_pred["nn_mean"],
    }
