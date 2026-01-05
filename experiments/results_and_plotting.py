from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

import train_tune_eval as tte


# ============================================================
# Types
# ============================================================
# Semantics: (cls, t0, t1, u0, u1)
SemanticTuple = Tuple[Any, float, float, float, float]


# ============================================================
# Helpers: selection
# ============================================================
def pick_examples_from_set(test_set: List[int], n: int, seed: int) -> List[int]:
    rng = np.random.default_rng(seed)
    test_set = list(test_set)
    if len(test_set) == 0:
        return []
    return list(rng.choice(test_set, size=min(n, len(test_set)), replace=False))


# ============================================================
# Plot helper: input u(t) + semantic transitions + state labels
# ============================================================
def plot_input_with_transition_points_and_states(
    ax: plt.Axes,
    t: np.ndarray,
    u: np.ndarray,
    semantics: List[SemanticTuple],
    *,
    lw: float = 2.0,
    annotate_coords: bool = True,
):
    """
    Plot u(t) with semantic segmentation overlays:
      - the input curve
      - dashed vertical lines at segment boundaries
      - dots at boundary points
      - optional (t, u) coordinate annotations
      - "STATE <cls>" label for each segment near its midpoint
    """
    ax.plot(t, u, linewidth=lw, label="Input u(t)")

    if not semantics:
        return

    # Build transition point polyline from segment endpoints.
    tp = [float(semantics[0][1])]
    utp = [float(semantics[0][3])]
    for cls, t0, t1, u0, u1 in semantics:
        tp.append(float(t1))
        utp.append(float(u1))

    tp = np.asarray(tp, dtype=float)
    utp = np.asarray(utp, dtype=float)

    for x in tp:
        ax.axvline(x, color="black", linestyle="--", linewidth=1.0, alpha=0.6)

    ax.scatter(tp, utp, color="r", s=45, zorder=5, label="Transitions")

    if annotate_coords:
        for x, yu in zip(tp, utp):
            ax.annotate(
                f"({x:.3f}, {yu:.3f})",
                (x, yu),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
                color="r",
            )

    u_range = float(np.ptp(u)) if np.ptp(u) > 0 else 1.0
    y_offset = 0.05 * u_range

    for cls, t0, t1, u0, u1 in semantics:
        tm = 0.5 * (float(t0) + float(t1))
        um = 0.5 * (float(u0) + float(u1))
        cls_str = str(int(cls)) if str(cls).isdigit() else str(cls)
        ax.text(
            tm,
            um + y_offset,
            f"STATE {cls_str}",
            ha="center",
            va="bottom",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                alpha=0.85,
                linewidth=0.5,
            ),
        )


# ============================================================
# Transformer prediction (dict-batch path)
# ============================================================
def predict_tf(lit_tf, ds, idx: int) -> np.ndarray:
    """
    Run a single-example forward pass for the semantic transformer.

    This assumes the model path expects the batch key "semantics" and that the
    dataset exposes ds.get_semantics(i). If those were renamed elsewhere, this
    stays consistent.
    """
    t = np.asarray(ds.ts[idx], dtype=float)
    y = np.asarray(ds.ys[idx], dtype=float)

    semantics = ds.get_semantics(idx)

    batch = {
        "static": torch.tensor(ds.X.iloc[idx].values, dtype=torch.float32)
        .unsqueeze(0)
        .to(tte.device),
        "semantics": [semantics],
        "t": torch.tensor(t, dtype=torch.float32).unsqueeze(0).to(tte.device),
        "y": torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(tte.device),
    }

    with torch.no_grad():
        y_hat, _ = lit_tf._forward_from_dict_batch(batch)

    return y_hat.squeeze(0).detach().cpu().numpy()


# ============================================================
# RNN predictions (reuse rebuild + loaders)
# ============================================================
def get_rnn_predictions(ds, out_json: Dict[str, Any], seed: int) -> Dict[int, np.ndarray]:
    split = out_json["split_indices"]
    train_idx = split["train"]
    val_idx = split["val"]
    test_idx = split["test"]

    rnn_cfg = tte.build_rnn_cfg_from_best_params(
        out_json["rnn_best_params"],
        ds,
        train_idx=train_idx,
        seed=seed,
    )

    _, _, test_loader, _ = tte.create_rnn_loaders_interleaved_fixed_split(
        rnn_cfg,
        ds,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        n_basis=rnn_cfg.n_basis,
        internal_knots=rnn_cfg.internal_knots,
    )

    lit_rnn = tte.LitTTSDynamic.load_from_checkpoint(
        out_json["rnn_best_ckpt_path"],
        config=rnn_cfg,
    )

    trainer = tte.pl.Trainer(
        accelerator=tte.device,
        devices="auto",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        deterministic=True,
    )

    pred_batches = trainer.predict(lit_rnn, dataloaders=test_loader)

    flat_preds: List[np.ndarray] = []
    for batch_preds in pred_batches:
        for p in batch_preds:
            flat_preds.append(p.detach().cpu().numpy())

    assert len(flat_preds) == len(test_idx), (len(flat_preds), len(test_idx))
    return {orig_idx: flat_preds[i] for i, orig_idx in enumerate(test_idx)}


# ============================================================
# Build TF lit module from checkpoint + best params
# ============================================================
def build_tf_lit(ds, out_json: Dict[str, Any], seed: int):
    split = out_json["split_indices"]
    cfg = tte.build_tf_cfg_from_best_params(
        out_json["tf_best_params"],
        ds,
        split["train"],
        seed,
    )
    lit = tte.LitSemanticTransformer.load_from_checkpoint(
        out_json["tf_best_ckpt_path"],
        config=cfg,
    ).to(tte.device)
    lit.eval()
    return lit


# ============================================================
# Plot: one grid figure (rows = examples, cols = views)
# ============================================================
def plot_four_column_grid_one_figure(
    ds_tumor,
    ds_sine,
    ds_beta,
    out_tumor: Dict[str, Any],
    out_sine: Dict[str, Any],
    out_beta: Dict[str, Any],
    fig_dir: Path,
    seed: int,
    n_examples: int,
    annotate_input_coords: bool = True,
):
    # Use the intersection of test sets so the same indices exist across datasets.
    test_t = set(out_tumor["split_indices"]["test"])
    test_s = set(out_sine["split_indices"]["test"])
    test_b = set(out_beta["split_indices"]["test"])
    common_test = sorted(list(test_t & test_s & test_b))
    if len(common_test) == 0:
        raise RuntimeError("No common indices in test splits across Tumour/Sine/Beta.")

    chosen = pick_examples_from_set(common_test, n_examples, seed)

    print(f"[plot] seed={seed} n_examples={n_examples}")
    print(f"[plot] test sizes: tumour={len(test_t)} sine={len(test_s)} beta={len(test_b)}")
    print(f"[plot] common_test size={len(common_test)}")
    print(f"[plot] chosen indices: {chosen}")

    # Load TF checkpoints + RNN predictions.
    lit_tf_t = build_tf_lit(ds_tumor, out_tumor, seed)
    lit_tf_s = build_tf_lit(ds_sine, out_sine, seed)
    lit_tf_b = build_tf_lit(ds_beta, out_beta, seed)

    rnn_pred_t = get_rnn_predictions(ds_tumor, out_tumor, seed)
    rnn_pred_s = get_rnn_predictions(ds_sine, out_sine, seed)
    rnn_pred_b = get_rnn_predictions(ds_beta, out_beta, seed)

    fig, axes = plt.subplots(
        n_examples,
        4,
        figsize=(16, 3.4 * n_examples),
        squeeze=False,
    )

    for r, idx in enumerate(chosen):
        ax_in, ax_t, ax_s, ax_b = axes[r]

        # INPUT: plotted from the tumour dataset's dynamic input
        t_in = np.asarray(ds_tumor.ts[idx], dtype=float)
        u_in = np.asarray(ds_tumor.X_dynamic.loc[idx, "x_dynamic"], dtype=float)
        semantics_in = ds_tumor.get_semantics(idx)

        plot_input_with_transition_points_and_states(
            ax_in,
            t_in,
            u_in,
            semantics_in,
            annotate_coords=annotate_input_coords,
        )
        ax_in.set_xlabel("t")
        ax_in.set_ylabel("u(t)")
        ax_in.set_title(f"Input (idx={idx})")
        ax_in.legend(fontsize=8, loc="best")

        # TUMOUR
        t = np.asarray(ds_tumor.ts[idx], dtype=float)
        y = np.asarray(ds_tumor.ys[idx], dtype=float)
        y_tf = predict_tf(lit_tf_t, ds_tumor, idx)
        y_rnn = rnn_pred_t[idx]

        ax_t.plot(t, y, "k", lw=2, label="GT")
        ax_t.plot(t, y_tf, lw=2, label="TF")
        ax_t.plot(t, y_rnn, lw=2, ls="--", label="RNN")
        ax_t.set_xlabel("t")
        ax_t.set_ylabel("y(t)")
        ax_t.set_title("Tumour")
        ax_t.legend(fontsize=8, loc="best")

        # SINE
        t = np.asarray(ds_sine.ts[idx], dtype=float)
        y = np.asarray(ds_sine.ys[idx], dtype=float)
        y_tf = predict_tf(lit_tf_s, ds_sine, idx)
        y_rnn = rnn_pred_s[idx]

        ax_s.plot(t, y, "k", lw=2, label="GT")
        ax_s.plot(t, y_tf, lw=2, label="TF")
        ax_s.plot(t, y_rnn, lw=2, ls="--", label="RNN")
        ax_s.set_xlabel("t")
        ax_s.set_ylabel("y(t)")
        ax_s.set_title("Sine")
        ax_s.legend(fontsize=8, loc="best")

        # BETA
        t = np.asarray(ds_beta.ts[idx], dtype=float)
        y = np.asarray(ds_beta.ys[idx], dtype=float)
        y_tf = predict_tf(lit_tf_b, ds_beta, idx)
        y_rnn = rnn_pred_b[idx]

        ax_b.plot(t, y, "k", lw=2, label="GT")
        ax_b.plot(t, y_tf, lw=2, label="TF")
        ax_b.plot(t, y_rnn, lw=2, ls="--", label="RNN")
        ax_b.set_xlabel("t")
        ax_b.set_ylabel("y(t)")
        ax_b.set_title("Beta")
        ax_b.legend(fontsize=8, loc="best")

    fig.suptitle(
        "Test-set examples (rows) — Columns: Input | Tumour | Sine | Beta",
        y=0.995,
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_path = fig_dir / f"fourcol_grid_n{n_examples}_seed{seed}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[saved] {out_path}")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    tte.patch_dynamic_encoder_no_permute()

    seed = int(os.environ.get("SEED", "0"))
    n_examples = int(os.environ.get("N_EXAMPLES", "5"))
    annotate_input_coords = bool(int(os.environ.get("ANNOTATE_INPUT_COORDS", "1")))

    results_dir = Path("results_bias_off")
    fig_dir = Path("report_figs")
    fig_dir.mkdir(exist_ok=True)

    # Dataset construction mirrors the training/eval script.
    ds_tumor = tte.DynamicTumorDataset(2000, 60)
    ds_sine = tte.DynamicSineTransDataset(2000, 60)
    ds_beta = tte.DynamicBetaDataset(2000, 60)

    out_tumor = json.loads((results_dir / "DynamicTumorDataset.json").read_text())
    out_sine = json.loads((results_dir / "DynamicSineDataset.json").read_text())
    out_beta = json.loads((results_dir / "DynamicBetaDataset.json").read_text())

    plot_four_column_grid_one_figure(
        ds_tumor=ds_tumor,
        ds_sine=ds_sine,
        ds_beta=ds_beta,
        out_tumor=out_tumor,
        out_sine=out_sine,
        out_beta=out_beta,
        fig_dir=fig_dir,
        seed=seed,
        n_examples=n_examples,
        annotate_input_coords=annotate_input_coords,
    )


if __name__ == "__main__":
    main()
