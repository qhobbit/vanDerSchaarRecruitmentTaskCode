import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

import train_tune_eval as tte


# Motif class IDs (kept as "motif class" on purpose)
# 0 = linearly increasing, 1 = linearly decreasing, 2 = constant
CLS_LIN_UP = 0
CLS_LIN_DN = 1
CLS_CONST = 2


def make_single_token(cls: int, u0: float, u1: float, t0: float = 0.0, t1: float = 1.0):
    """
    Build a "token" containing a single semantic tuple.

    The model interface expects a list of tuples (even when there's only one).
    Tuple format is:
        (motif_class, t_start, t_end, value_start, value_end)
    """
    return [(int(cls), float(t0), float(t1), float(u0), float(u1))]


def u_from_token_linear(t: np.ndarray, token):
    """
    Reconstruct u(t) implied by a single semantic tuple.

    For these demos the tuple is always linear, so this is just endpoint
    interpolation over the provided time grid.
    """
    _, t0, t1, u0, u1 = token[0]
    t0 = float(t0)
    t1 = float(t1)
    return u0 + (u1 - u0) * (t - t0) / (t1 - t0)


def gt_dynamic_sine_trans(t: np.ndarray, x_static: float, alpha: float, u_t: np.ndarray):
    """
    Ground-truth mapping for the Dynamic Sine (transformed) dataset.

    y(t) = sin(2π t / x + α u(t))
    """
    return np.sin(2.0 * np.pi * t / float(x_static) + float(alpha) * u_t)


def predict_tf_from_token(lit_tf, static_vec: np.ndarray, t: np.ndarray, token, y_gt: np.ndarray):
    """
    Forward pass through the Lightning module using a single semantic tuple.
    """
    batch = {
        "static": torch.tensor(static_vec, dtype=torch.float32).unsqueeze(0).to(tte.device),
        "semantics": [token],
        "t": torch.tensor(t, dtype=torch.float32).unsqueeze(0).to(tte.device),
        "y": torch.tensor(y_gt, dtype=torch.float32).unsqueeze(0).to(tte.device),
    }
    with torch.no_grad():
        y_hat, _target = lit_tf._forward_from_dict_batch(batch)
    return y_hat.squeeze(0).detach().cpu().numpy()


def global_ylim_from_curves(curves, pad=0.05):
    """
    Shared y-limits across a set of curves.

    This avoids plots looking "better" or "worse" just because autoscaling
    picked different limits in each panel.
    """
    ymin = float(min(np.min(c) for c in curves))
    ymax = float(max(np.max(c) for c in curves))
    span = ymax - ymin
    if span < 1e-8:
        span = 1.0
    ymin -= pad * span
    ymax += pad * span
    return ymin, ymax


def add_semantic_tuple_label_box(ax, label: str):
    """
    No axis titles: keep a small label inside the plot instead.

    This reads cleaner in the final figure while still making the semantics obvious.
    """
    ax.text(
        0.5,
        0.92,
        label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, linewidth=0.6),
    )


def main():
    # Load the stored tuning/training output to find the checkpoint and split.
    results_json = Path("results_bias_off/DynamicSineTransDataset.json")
    if not results_json.exists():
        results_json = Path("results_bias_off/DynamicSineDataset.json")

    out_json = json.loads(results_json.read_text())
    seed = int(out_json.get("seed", 0))

    # Recreate the dataset to recover the time grid + dataset constants.
    # (The model itself is loaded from the checkpoint.)
    ds = tte.DynamicSineTransDataset(n_samples=2000, n_timesteps=60)
    t = np.asarray(ds.ts[0], dtype=float)

    split = out_json["split_indices"]
    train_idx = list(split["train"])
    tf_cfg = tte.build_tf_cfg_from_best_params(out_json["tf_best_params"], ds, train_idx, seed)

    lit_tf = tte.LitSemanticTransformer.load_from_checkpoint(
        out_json["tf_best_ckpt_path"],
        config=tf_cfg,
    ).to(tte.device)
    lit_tf.eval()

    # Fixed static covariates across all runs (Sine dataset has one static: x).
    x_static = 3.0
    static_vec = np.array([x_static], dtype=np.float32)

    alpha = float(getattr(ds, "alpha", 0.5))

    # ------------------------------------------------------------------
    # Level 1: change motif class by swapping the semantic tuple class.
    # ------------------------------------------------------------------
    cases_lvl1 = [
        (make_single_token(CLS_CONST, 1.0, 1.0), "Constant"),
        (make_single_token(CLS_LIN_UP, 0.0, 1.5), "Linearly increasing"),
        (make_single_token(CLS_LIN_DN, 0.0, -1.5), "Linearly decreasing"),
    ]

    # Precompute everything once so limits can be shared fairly across panels.
    lvl1_cache = []
    lvl1_outputs = []
    for token, label in cases_lvl1:
        u_t = u_from_token_linear(t, token)
        y_gt = gt_dynamic_sine_trans(t, x_static=x_static, alpha=alpha, u_t=u_t)
        y_hat = predict_tf_from_token(lit_tf, static_vec, t, token, y_gt=y_gt)
        lvl1_cache.append((token, label, u_t, y_gt, y_hat))
        lvl1_outputs.extend([y_gt, y_hat])

    y1min, y1max = global_ylim_from_curves(lvl1_outputs, pad=0.05)

    fig1, axes1 = plt.subplots(
        2,
        3,
        figsize=(12, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.3]},
    )

    for j, (token, label, u_t, y_gt, y_hat) in enumerate(lvl1_cache):
        ax_u = axes1[0, j]
        ax_y = axes1[1, j]

        # Input panel: u(t) implied by the semantic tuple (no plot titles).
        ax_u.plot(t, u_t, lw=2)
        add_semantic_tuple_label_box(ax_u, label)
        if j == 0:
            ax_u.set_ylabel("u(t)")

        # Output panel: GT and prediction overlaid on the same axes.
        ax_y.plot(t, y_gt, "k", lw=2, label="GT $y(t)$")
        ax_y.plot(t, y_hat, lw=2, label="TF $\\hat{y}(t)$")
        ax_y.set_ylim(y1min, y1max)
        ax_y.set_xlabel("t")
        if j == 0:
            ax_y.set_ylabel("y(t)")
            ax_y.legend(fontsize=8)

    fig1.suptitle(
        "Bilevel transparency (single semantic tuple) — Level 1: motif class controls output shape\n"
        f"Static fixed: x={x_static:.3f} | Output y-limits shared across cases",
        y=1.02,
    )
    fig1.tight_layout()
    fig1.savefig("bilevel_token_demo_level1_overlay_global_ylim.png", dpi=220, bbox_inches="tight")
    plt.close(fig1)

    # ------------------------------------------------------------------
    # Level 2: keep motif class fixed; change the semantic tuple endpoints.
    # ------------------------------------------------------------------
    cases_lvl2 = [
        (make_single_token(CLS_LIN_UP, -1.5, 1.5), "Linearly increasing"),
        (make_single_token(CLS_LIN_UP, -0.2, 0.2), "Linearly increasing"),
    ]

    lvl2_cache = []
    lvl2_outputs = []
    for token, label in cases_lvl2:
        u_t = u_from_token_linear(t, token)
        y_gt = gt_dynamic_sine_trans(t, x_static=x_static, alpha=alpha, u_t=u_t)
        y_hat = predict_tf_from_token(lit_tf, static_vec, t, token, y_gt=y_gt)
        lvl2_cache.append((token, label, u_t, y_gt, y_hat))
        lvl2_outputs.extend([y_gt, y_hat])

    y2min, y2max = global_ylim_from_curves(lvl2_outputs, pad=0.05)

    fig2, axes2 = plt.subplots(
        2,
        2,
        figsize=(10, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.3]},
    )

    for j, (token, label, u_t, y_gt, y_hat) in enumerate(lvl2_cache):
        ax_u = axes2[0, j]
        ax_y = axes2[1, j]

        ax_u.plot(t, u_t, lw=2)
        add_semantic_tuple_label_box(ax_u, label)
        if j == 0:
            ax_u.set_ylabel("u(t)")

        ax_y.plot(t, y_gt, "k", lw=2, label="GT $y(t)$")
        ax_y.plot(t, y_hat, lw=2, label="TF $\\hat{y}(t)$")
        ax_y.set_ylim(y2min, y2max)
        ax_y.set_xlabel("t")
        if j == 0:
            ax_y.set_ylabel("y(t)")
            ax_y.legend(fontsize=8)

    fig2.suptitle(
        "Bilevel transparency (single semantic tuple) — Level 2: tuple endpoints (slope) control output quantitatively\n"
        f"Static fixed: x={x_static:.3f}, alpha={alpha:.3f} | Output y-limits shared across cases",
        y=1.02,
    )
    fig2.tight_layout()
    fig2.savefig("bilevel_token_demo_level2_overlay_global_ylim.png", dpi=220, bbox_inches="tight")
    plt.close(fig2)

    print(
        "Saved:",
        "bilevel_token_demo_level1_overlay_global_ylim.png",
        "bilevel_token_demo_level2_overlay_global_ylim.png",
    )


if __name__ == "__main__":
    main()
