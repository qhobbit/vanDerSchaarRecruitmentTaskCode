from __future__ import annotations

import os
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Callable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ---- project imports ----
from timeview.lit_module import LitTTSDynamic, LitSemanticTransformer
from timeview.config import DynamicTuningConfig, SemanticTransformerTuningConfig
from timeview.knot_selection import calculate_knot_placement
from timeview.basis import BSplineBasis

# ---- datasets ----
from datasets import DynamicTumorDataset, DynamicBetaDataset, DynamicSineTransDataset


device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# 0) Patch: DynamicEncoder.forward to accept (B, T, F) directly
# ============================================================
def patch_dynamic_encoder_no_permute():
    import timeview.model as tv_model

    DynamicEncoder = tv_model.DynamicEncoder
    orig_forward = DynamicEncoder.forward

    def forward_no_permute(self, x):
        # RNNs are configured batch_first=True, so x is expected as (B, T, F).
        # The older forward path permuted dims, which breaks the interleaved setup.
        if getattr(self, "rnn_type", "lstm").lower() == "lstm":
            _, (h_n, _) = self.rnn(x)
        else:
            _, h_n = self.rnn(x)

        # Last layer hidden state -> latent
        last_hidden_state = h_n[-1]
        latent = self.fc(last_hidden_state)
        latent = self.batch_norm(latent)
        latent = self.activation(latent)
        latent = self.dropout(latent)
        return latent

    # Patch only if the existing forward likely contains a permute.
    try:
        if "permute" in set(orig_forward.__code__.co_names):
            DynamicEncoder.forward = forward_no_permute
    except Exception:
        # If introspection fails, patch anyway to avoid silent shape issues.
        DynamicEncoder.forward = forward_no_permute


# ============================================================
# 1) Small utilities
# ============================================================
def _seed_worker(worker_id: int):
    # Dataloader worker seeding: keep numpy/random aligned with torch.
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def infer_static_dim(ds: Any) -> int:
    # Static covariates are stored in a pandas frame.
    return int(ds.X.shape[1])


def count_params(module: torch.nn.Module) -> int:
    # Trainable params only (used for a rough TF vs RNN comparison).
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def split_indices_fixed(n_total: int, train_frac: float, val_frac: float, seed: int):
    """
    Fixed train/val/test split computed once per dataset and reused across trials.

    Note: this bypasses any split controls exposed via config classes. It is
    done intentionally here to keep Optuna trials comparable.
    """
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n_total, generator=g).tolist()
    n_train = int(train_frac * n_total)
    n_val = int(val_frac * n_total)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    return train_idx, val_idx, test_idx


def internal_knots_from_train_indices(
    ds: Any,
    train_idx: List[int],
    n_basis: int,
    T: float,
    seed: int,
) -> List[float]:
    """
    Compute internal knot placement from the train split only.
    """
    ts_train = [ds.ts[i] for i in train_idx]
    ys_train = [ds.ys[i] for i in train_idx]
    return calculate_knot_placement(
        ts_train,
        ys_train,
        n_internal_knots=int(n_basis - 2),
        T=float(T),
        seed=int(seed),
    )


def safe_json_dump(obj: Any, path: Path):
    # Convenience helper for writing tuning summaries and loss curves.
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _find_best_ckpt(ckpt_dir: Path) -> Path:
    # Expect ModelCheckpoint naming like best-{epoch}-{val_loss}.ckpt
    best = sorted(ckpt_dir.glob("best-*.ckpt"))
    if not best:
        last = ckpt_dir / "last.ckpt"
        if last.exists():
            return last
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    return best[0]


class TrialShim:
    """
    Minimal Optuna Trial adapter backed by a params dict.
    Used to rebuild configs after tuning without re-running Optuna.
    """
    def __init__(self, params):
        self.params = params

    def suggest_int(self, name, low, high):
        return int(self.params[name]) if name in self.params else int(low)

    def suggest_float(self, name, low, high, log=False):
        return float(self.params[name]) if name in self.params else float(low)

    def suggest_categorical(self, name, choices):
        return self.params[name] if name in self.params else choices[0]


def evaluate_checkpoint_on_test(
    lit_cls,
    cfg,
    ckpt_path: Path,
    test_loader: DataLoader,
) -> Tuple[float, int, Dict[str, Any]]:
    """
    Load a checkpoint, run evaluation on the test set, and return:
      (test_loss, parameter_count, raw_metrics_dict)

    If test_step does not log `test_loss`, validate() is used as a fallback and
    `val_loss` (or `loss`) is treated as a proxy.
    """
    try:
        model = lit_cls.load_from_checkpoint(str(ckpt_path), config=cfg)
    except TypeError:
        model = lit_cls.load_from_checkpoint(str(ckpt_path))

    n_params = count_params(model.model)

    trainer = pl.Trainer(
        accelerator=device,
        devices="auto",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        deterministic=True,
    )

    raw_metrics: Dict[str, Any] = {}
    test_loss: Optional[float] = None

    try:
        res = trainer.test(model, dataloaders=test_loader, verbose=False)
        if isinstance(res, list) and len(res) > 0 and isinstance(res[0], dict):
            raw_metrics = dict(res[0])
            if "test_loss" in raw_metrics:
                test_loss = float(raw_metrics["test_loss"])
    except Exception:
        pass

    if test_loss is None:
        res = trainer.validate(model, dataloaders=test_loader, verbose=False)
        if isinstance(res, list) and len(res) > 0 and isinstance(res[0], dict):
            raw_metrics = dict(res[0])
            if "val_loss" in raw_metrics:
                test_loss = float(raw_metrics["val_loss"])
            elif "loss" in raw_metrics:
                test_loss = float(raw_metrics["loss"])

    if test_loss is None:
        raise RuntimeError(
            "Could not extract test loss. Ensure the LightningModule logs "
            "`test_loss` in test_step, or `val_loss` in validation_step."
        )

    return test_loss, n_params, raw_metrics


# ============================================================
# 2) Callback: write per-epoch train/val losses to JSON
# ============================================================
class LossHistoryCallback(Callback):
    def __init__(self, out_path: Path):
        super().__init__()
        self.out_path = out_path
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.epochs: List[int] = []

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        m = trainer.callback_metrics
        epoch = int(trainer.current_epoch)
        v = m.get("train_loss", m.get("loss", None))
        if v is not None:
            self.epochs.append(epoch)
            self.train_losses.append(float(v.detach().cpu().item()))

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        m = trainer.callback_metrics
        v = m.get("val_loss", None)
        if v is not None:
            self.val_losses.append(float(v.detach().cpu().item()))

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        payload = {
            "epochs_recorded": self.epochs,
            "train_loss": self.train_losses,
            "val_loss": self.val_losses,
            "stopped_epoch": int(trainer.current_epoch),
        }
        safe_json_dump(payload, self.out_path)


def make_trainer(max_epochs: int, ckpt_dir: Path, history_path: Path) -> pl.Trainer:
    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best-{epoch}-{val_loss:.6f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    es_cb = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=10,
        min_delta=0.0,
    )
    hist_cb = LossHistoryCallback(history_path)

    return pl.Trainer(
        max_epochs=max_epochs,
        accelerator=device,
        devices="auto",
        enable_checkpointing=True,
        logger=False,
        enable_progress_bar=False,
        deterministic=True,
        callbacks=[ckpt_cb, es_cb, hist_cb],
    )


class DynamicInterleavedPaddedDataset(Dataset):
    """
    Wraps the base dataset to produce the RNN "interleaved" representation.

    Each item returns (X_static, X_dynamic_padded, Phi, y):
      - X_static:         (F_static,)
      - X_dynamic_padded: (Lmax, 1)   interleaved vector treated as a sequence
      - Phi:              (N_i, n_basis) spline basis evaluated at this sample's times
      - y:                (N_i,)        targets on irregular times

    Note: padding to a dataset-wide Lmax is a convenience choice here. With more
    time, it could be made cleaner (e.g., packed sequences or per-batch padding).
    """

    def __init__(
        self,
        base_ds: Any,
        n_basis: int,
        T: float,
        internal_knots: List[float],
        append_last_endpoint_property: bool = False,
        pad_value: float = 0.0,
        device_for_phi: str = device,
    ):
        self.ds = base_ds
        self.n_basis = int(n_basis)
        self.T = float(T)
        self.internal_knots = internal_knots
        self.append_last_endpoint_property = bool(append_last_endpoint_property)
        self.pad_value = float(pad_value)
        self.device_for_phi = device_for_phi

        # Precompute interleaved vectors and a fixed max length for padding.
        self._flat: List[np.ndarray] = []
        lens: List[int] = []
        for i in range(len(self.ds)):
            v = self.ds.encode_dynamic_interleaved_irregular_for_sample(
                i, append_last_endpoint_property=self.append_last_endpoint_property
            )
            v = np.asarray(v, dtype=np.float32).reshape(-1)
            self._flat.append(v)
            lens.append(int(v.shape[0]))
        self.Lmax = int(max(lens) if lens else 0)

        self._basis = BSplineBasis(self.n_basis, (0.0, self.T), internal_knots=self.internal_knots)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int):
        x_static = torch.tensor(self.ds.X.iloc[i].values, dtype=torch.float32)

        flat = self._flat[i]
        L = flat.shape[0]
        x_dyn = torch.from_numpy(flat).float().view(L, 1)  # (L,1)
        if L < self.Lmax:
            pad = torch.full((self.Lmax - L, 1), self.pad_value, dtype=torch.float32)
            x_dyn = torch.cat([x_dyn, pad], dim=0)          # (Lmax,1)

        t = np.asarray(self.ds.ts[i], dtype=np.float32).reshape(-1)
        y = np.asarray(self.ds.ys[i], dtype=np.float32).reshape(-1)

        Phi_np = self._basis.get_matrix(t)                 # (N_i, n_basis)
        Phi = torch.from_numpy(Phi_np).float().to(self.device_for_phi)
        y_t = torch.from_numpy(y).float().to(self.device_for_phi)

        return x_static, x_dyn, Phi, y_t


def dynamic_iterative_collate(batch):
    # Phi and y remain as lists since they can have per-sample lengths.
    batch_X = torch.stack([b[0] for b in batch], dim=0)     # (B, F_static)
    batch_Xdyn = torch.stack([b[1] for b in batch], dim=0)  # (B, Lmax, 1)
    batch_Phis = [b[2] for b in batch]                      # list length B
    batch_ys = [b[3] for b in batch]                        # list length B
    return batch_X, batch_Xdyn, batch_Phis, batch_ys


def create_rnn_loaders_interleaved_fixed_split(
    cfg,
    base_ds: Any,
    train_idx: List[int],
    val_idx: List[int],
    test_idx: List[int],
    n_basis: int,
    internal_knots: List[float],
):
    """
    Build dataloaders for the RNN baseline using fixed split indices.

    Note: split indices come from split_indices_fixed(), so config-level split
    controls are effectively bypassed.
    """
    seed = int(getattr(cfg, "seed", 0))
    g = torch.Generator().manual_seed(seed)

    wrapped = DynamicInterleavedPaddedDataset(
        base_ds=base_ds,
        append_last_endpoint_property=False,
        T=1.0,
        pad_value=0.0,
        n_basis=n_basis,
        internal_knots=internal_knots
    )

    train_ds = Subset(wrapped, train_idx)
    val_ds   = Subset(wrapped, val_idx)
    test_ds  = Subset(wrapped, test_idx)

    bs = int(getattr(getattr(cfg, "training", object()), "batch_size", 32))
    num_workers = int(getattr(cfg, "num_workers", 0))

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, generator=g,
        worker_init_fn=_seed_worker, collate_fn=dynamic_iterative_collate,
        num_workers=num_workers, persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, generator=g,
        worker_init_fn=_seed_worker, collate_fn=dynamic_iterative_collate,
        num_workers=num_workers, persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False, generator=g,
        worker_init_fn=_seed_worker, collate_fn=dynamic_iterative_collate,
        num_workers=num_workers, persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader, test_loader, wrapped.Lmax


# ============================================================
# 4) Transformer data: keep the semantics structure as-is
# ============================================================
class SemanticDataset(Dataset):
    # Thin wrapper so Lightning modules get a consistent dict batch.
    def __init__(self, base_dataset: Any):
        self.ds = base_dataset

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int):
        return {
            "static": torch.tensor(self.ds.X.iloc[i].values, dtype=torch.float32),
            "semantics": self.ds.get_semantics(i),
            "t": torch.tensor(self.ds.ts[i], dtype=torch.float32),
            "y": torch.tensor(self.ds.ys[i], dtype=torch.float32),
        }


def semantic_collate_fn(batch):
    # Semantics stays as a list since it is a nested per-sample structure.
    static = torch.stack([b["static"] for b in batch], dim=0)
    semantics = [b["semantics"] for b in batch]
    t = torch.stack([b["t"] for b in batch], dim=0)
    y = torch.stack([b["y"] for b in batch], dim=0)
    return {"static": static, "semantics": semantics, "t": t, "y": y}


def create_tf_loaders_fixed_split(cfg, base_ds: Any, train_idx: List[int], val_idx: List[int], test_idx: List[int]):
    """
    Build dataloaders for the semantic transformer using fixed split indices.

    Note: split indices come from split_indices_fixed(), so config-level split
    controls are effectively bypassed.
    """
    seed = int(getattr(cfg, "seed", 0))
    g = torch.Generator().manual_seed(seed)

    wrapped = SemanticDataset(base_ds)
    train_ds = Subset(wrapped, train_idx)
    val_ds   = Subset(wrapped, val_idx)
    test_ds  = Subset(wrapped, test_idx)

    bs = int(getattr(getattr(cfg, "training", object()), "batch_size", 32))
    num_workers = int(getattr(cfg, "num_workers", 0))

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, generator=g,
        worker_init_fn=_seed_worker, collate_fn=semantic_collate_fn,
        num_workers=num_workers, persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, generator=g,
        worker_init_fn=_seed_worker, collate_fn=semantic_collate_fn,
        num_workers=num_workers, persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False, generator=g,
        worker_init_fn=_seed_worker, collate_fn=semantic_collate_fn,
        num_workers=num_workers, persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader, test_loader


# ============================================================
# 5) Optuna objectives
# ============================================================
def objective_rnn(
    trial: optuna.Trial,
    ds: Any,
    ds_name: str,
    seed: int,
    ckpt_root: Path,
    train_idx: List[int],
    val_idx: List[int],
    test_idx: List[int],
) -> float:
    # Seed is reset per trial for determinism.
    # Split indices are fixed outside the configs.
    # basis is tuned outside config so that internal knots can be set after.
    seed_everything(seed, workers=True)

    rnn_n_basis_out = trial.suggest_int("rnn_n_basis_out", 5, 16)

    internal_knots = internal_knots_from_train_indices(
        ds=ds,
        train_idx=train_idx,
        n_basis=rnn_n_basis_out,
        T=1.0,
        seed=seed,
    )

    cfg = DynamicTuningConfig(
        trial=trial,
        n_features=infer_static_dim(ds),
        n_features_dynamic=1,
        n_basis=rnn_n_basis_out,
        T=1.0,
        seed=seed,
        device=device,
        num_epochs=100,
        internal_knots=internal_knots,
        n_basis_tunable=False,
        dynamic_bias=False,
        dataloader_type="iterative",
    )

    train_loader, val_loader, _test_loader, _Lmax = create_rnn_loaders_interleaved_fixed_split(
        cfg, ds, train_idx, val_idx, test_idx, n_basis=rnn_n_basis_out, internal_knots=internal_knots
    )

    ckpt_dir = ckpt_root / "rnn_interleaved" / ds_name / f"trial_{trial.number}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history_path = ckpt_dir / "loss_history.json"
    trainer = make_trainer(max_epochs=int(cfg.num_epochs), ckpt_dir=ckpt_dir, history_path=history_path)

    lit = LitTTSDynamic(cfg)
    trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader)

    val = trainer.callback_metrics.get("val_loss", None)
    if val is None:
        raise RuntimeError("val_loss not found for RNN.")
    return float(val.detach().cpu().item())


def objective_tf(
    trial: optuna.Trial,
    ds: Any,
    ds_name: str,
    seed: int,
    ckpt_root: Path,
    train_idx: List[int],
    val_idx: List[int],
    test_idx: List[int],
) -> float:
    seed_everything(seed, workers=True)

    # basis is tuned outside config so that internal knots can be set after.
    tf_n_basis = trial.suggest_int("tf_n_basis", 5, 16)

    internal_knots = internal_knots_from_train_indices(
        ds=ds,
        train_idx=train_idx,
        n_basis=tf_n_basis,
        T=1.0,
        seed=seed,
    )

    cfg = SemanticTransformerTuningConfig(
        trial=trial,
        n_features=infer_static_dim(ds),
        n_basis=tf_n_basis,
        T=1.0,
        seed=seed,
        device=device,
        num_epochs=100,
        internal_knots=internal_knots,
        dynamic_bias=False,
        n_basis_tunable=False,
    )

    train_loader, val_loader, _test_loader = create_tf_loaders_fixed_split(cfg, ds, train_idx, val_idx, test_idx)

    ckpt_dir = ckpt_root / "transformer" / ds_name / f"trial_{trial.number}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history_path = ckpt_dir / "loss_history.json"
    trainer = make_trainer(max_epochs=int(cfg.num_epochs), ckpt_dir=ckpt_dir, history_path=history_path)

    lit = LitSemanticTransformer(cfg)
    trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader)

    val = trainer.callback_metrics.get("val_loss", None)
    if val is None:
        raise RuntimeError("val_loss not found for Transformer.")
    return float(val.detach().cpu().item())


# ============================================================
# 6) Rebuild configs from best params (post-tuning) + test eval
# ============================================================
def build_rnn_cfg_from_best_params(
    best_params: Dict[str, Any],
    ds: Any,
    train_idx: List[int],
    seed: int,
) -> DynamicTuningConfig:
    rnn_n_basis_out = int(best_params["rnn_n_basis_out"])

    internal_knots = internal_knots_from_train_indices(
        ds=ds,
        train_idx=train_idx,
        n_basis=rnn_n_basis_out,
        T=1.0,
        seed=seed,
    )
    shim = TrialShim(best_params)

    cfg = DynamicTuningConfig(
        trial=shim,
        n_features=infer_static_dim(ds),
        n_features_dynamic=1,
        n_basis=rnn_n_basis_out,
        T=1.0,
        seed=seed,
        device=device,
        num_epochs=100,
        internal_knots=internal_knots,
        n_basis_tunable=False,
        dynamic_bias=False,
        dataloader_type="iterative",
    )

    for k, v in best_params.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def build_tf_cfg_from_best_params(
    best_params: Dict[str, Any],
    ds: Any,
    train_idx: List[int],
    seed: int,
) -> SemanticTransformerTuningConfig:
    tf_n_basis = int(best_params["tf_n_basis"])

    internal_knots = internal_knots_from_train_indices(
        ds=ds,
        train_idx=train_idx,
        n_basis=tf_n_basis,
        T=1.0,
        seed=seed,
    )
    shim = TrialShim(best_params)

    cfg = SemanticTransformerTuningConfig(
        trial=shim,
        n_features=infer_static_dim(ds),
        n_basis=tf_n_basis,
        T=1.0,
        seed=seed,
        device=device,
        num_epochs=100,
        internal_knots=internal_knots,
        dynamic_bias=False,
        n_basis_tunable=False,
    )

    for k, v in best_params.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


# ============================================================
# 7) Run tuning + evaluation for one dataset
# ============================================================
@dataclass
class DatasetSpec:
    name: str
    builder: Callable[[], Any]


def run_one_dataset(spec: DatasetSpec, seed: int, n_trials: int, ckpt_root: Path, results_dir: Path) -> Dict[str, Any]:
    ds = spec.builder()
    out: Dict[str, Any] = {"dataset": spec.name}

    # Split computed once and reused for all trials (both models).
    train_frac = 0.8
    val_frac = 0.1
    train_idx, val_idx, test_idx = split_indices_fixed(len(ds), train_frac, val_frac, seed)

    out["split_sizes"] = {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)}
    out["split_indices"] = {"train": train_idx, "val": val_idx, "test": test_idx}

    sampler = TPESampler(seed=seed)
    pruner = MedianPruner()

    # ---- RNN tuning ----
    study_rnn = optuna.create_study(
        direction="minimize",
        study_name=f"{spec.name}_rnn_interleaved",
        sampler=sampler,
        pruner=pruner,
    )
    study_rnn.optimize(
        lambda tr: objective_rnn(tr, ds, spec.name, seed, ckpt_root, train_idx, val_idx, test_idx),
        n_trials=n_trials,
        n_jobs=1,
    )
    out["rnn_best_val"] = float(study_rnn.best_value)
    out["rnn_best_params"] = dict(study_rnn.best_trial.params)
    out["rnn_best_trial_number"] = int(study_rnn.best_trial.number)

    # Rebuild cfg + loaders for the best RNN trial, then evaluate test loss.
    rnn_cfg = build_rnn_cfg_from_best_params(out["rnn_best_params"], ds, train_idx, seed)
    _rnn_train_loader, _rnn_val_loader, rnn_test_loader, _Lmax = create_rnn_loaders_interleaved_fixed_split(
        rnn_cfg, ds, train_idx, val_idx, test_idx, n_basis=rnn_cfg.n_basis, internal_knots=rnn_cfg.internal_knots
    )
    rnn_ckpt_dir = ckpt_root / "rnn_interleaved" / spec.name / f"trial_{out['rnn_best_trial_number']}"
    rnn_best_ckpt = _find_best_ckpt(rnn_ckpt_dir)
    out["rnn_best_ckpt_path"] = str(rnn_best_ckpt)

    rnn_test_loss, rnn_params, rnn_test_metrics = evaluate_checkpoint_on_test(
        LitTTSDynamic, rnn_cfg, rnn_best_ckpt, rnn_test_loader
    )
    out["rnn_test_loss"] = float(rnn_test_loss)
    out["rnn_param_count"] = int(rnn_params)
    out["rnn_test_metrics_raw"] = rnn_test_metrics

    # ---- Transformer tuning ----
    study_tf = optuna.create_study(
        direction="minimize",
        study_name=f"{spec.name}_transformer",
        sampler=sampler,
        pruner=pruner,
    )
    study_tf.optimize(
        lambda tr: objective_tf(tr, ds, spec.name, seed, ckpt_root, train_idx, val_idx, test_idx),
        n_trials=n_trials,
        n_jobs=1,
    )
    out["tf_best_val"] = float(study_tf.best_value)
    out["tf_best_params"] = dict(study_tf.best_trial.params)
    out["tf_best_trial_number"] = int(study_tf.best_trial.number)

    # ---- test eval for best TF ----
    tf_cfg = build_tf_cfg_from_best_params(out["tf_best_params"], ds, train_idx, seed)
    _tf_train_loader, _tf_val_loader, tf_test_loader = create_tf_loaders_fixed_split(
        tf_cfg, ds, train_idx, val_idx, test_idx
    )
    tf_ckpt_dir = ckpt_root / "transformer" / spec.name / f"trial_{out['tf_best_trial_number']}"
    tf_best_ckpt = _find_best_ckpt(tf_ckpt_dir)
    out["tf_best_ckpt_path"] = str(tf_best_ckpt)

    tf_test_loss, tf_params, tf_test_metrics = evaluate_checkpoint_on_test(
        LitSemanticTransformer, tf_cfg, tf_best_ckpt, tf_test_loader
    )
    out["tf_test_loss"] = float(tf_test_loss)
    out["tf_param_count"] = int(tf_params)
    out["tf_test_metrics_raw"] = tf_test_metrics

    # ---- Parameter ratios ----
    if rnn_params > 0:
        out["param_ratio_tf_over_rnn"] = float(tf_params) / float(rnn_params)
        out["param_ratio_rnn_over_tf"] = float(rnn_params) / float(tf_params) if tf_params > 0 else None
    else:
        out["param_ratio_tf_over_rnn"] = None
        out["param_ratio_rnn_over_tf"] = None

    results_dir.mkdir(parents=True, exist_ok=True)
    safe_json_dump(out, results_dir / f"{spec.name}.json")
    return out


# ============================================================
# 8) Main entry point
# ============================================================
def main():
    patch_dynamic_encoder_no_permute()

    # Run settings are read from env vars in this script.
    # This means seed/split choices are controlled here, not purely via configs.
    seed = int(os.environ.get("SEED", "0"))
    n_trials = int(os.environ.get("N_TRIALS", "25"))

    ckpt_root = Path(os.environ.get("CKPT_DIR", "checkpoints_bias_off"))
    results_dir = Path(os.environ.get("RESULTS_DIR", "results_bias_off"))

    ckpt_root.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        DatasetSpec("DynamicTumorDataset", lambda: DynamicTumorDataset(n_samples=2000, n_timesteps=60)),
        DatasetSpec("DynamicSineDataset",  lambda: DynamicSineTransDataset(n_samples=2000, n_timesteps=60)),
        DatasetSpec("DynamicBetaDataset",  lambda: DynamicBetaDataset(n_samples=2000, n_timesteps=60)),
    ]

    all_results: List[Dict[str, Any]] = []
    for spec in datasets:
        print(f"\n=== {spec.name}: tune Transformer + RNN (n_trials={n_trials}) ===")
        res = run_one_dataset(spec, seed=seed, n_trials=n_trials, ckpt_root=ckpt_root, results_dir=results_dir)
        all_results.append(res)

        print(f"{spec.name}")
        print(f"  TF  best val:  {res['tf_best_val']:.6f} | test: {res['tf_test_loss']:.6f} | params: {res['tf_param_count']}")
        print(f"  RNN best val:  {res['rnn_best_val']:.6f} | test: {res['rnn_test_loss']:.6f} | params: {res['rnn_param_count']}")
        print(f"  Param ratio (TF/RNN): {res['param_ratio_tf_over_rnn']:.4f}")

    safe_json_dump(all_results, results_dir / "ALL_RESULTS.json")
    print(f"\nSaved results JSON to: {results_dir.resolve()}")
    print(f"Saved checkpoints under: {ckpt_root.resolve()}")


if __name__ == "__main__":
    main()
