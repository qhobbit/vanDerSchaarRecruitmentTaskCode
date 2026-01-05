import pytorch_lightning as pl
import torch
from .config import Config, OPTIMIZERS
from .model import TTS, TTSDynamic
import glob
import os
import pickle
from .model import SemanticTransformer
from typing import Any, Dict, List, Optional, Tuple



def _get_seed_number(path):
    seeds = [os.path.basename(path).split("_")[1] for path in glob.glob(os.path.join(path, '*'))]
    seed = seeds[0]
    return seed

def _get_logs_seed_path(benchmarks_folder, timestamp, final=True, seed=None):

    # Create path
    if final:
        path = os.path.join(benchmarks_folder, timestamp, 'TTS', 'final', 'logs')
    else:
        path = os.path.join(benchmarks_folder, timestamp, 'TTS', 'tuning', 'logs')
    
    if seed is None:
        seed = _get_seed_number(path)

    logs_path = os.path.join(path, f'seed_{seed}')
    return logs_path

def _get_checkpoint_path_from_logs_seed_path(path):
    checkpoint_path = os.path.join(path, 'lightning_logs', 'version_0', 'checkpoints', 'best_val.ckpt')
    return checkpoint_path

def _load_config_from_logs_seed_path(path):
    config_path = os.path.join(path, 'config.pkl')
    # load config from a pickle
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    return config


def load_config(benchmarks_folder, timestamp, final=True, seed=None):

    logs_seed_path = _get_logs_seed_path(benchmarks_folder, timestamp, final=final, seed=seed)
    return _load_config_from_logs_seed_path(logs_seed_path)


def load_model(timestamp, benchmarks_folder='benchmarks', final=True, seed=None):

    logs_seed_path = _get_logs_seed_path(benchmarks_folder, timestamp, final=final, seed=seed)
    config = _load_config_from_logs_seed_path(logs_seed_path)
    checkpoint_path = _get_checkpoint_path_from_logs_seed_path(logs_seed_path)
    model = LitTTS.load_from_checkpoint(checkpoint_path, config=config)
    return model


class LitTTS(pl.LightningModule):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.model = TTS(config)
        self.loss_fn = torch.nn.MSELoss()
        self.lr = self.config.training.lr

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # def forward(self, batch, batch_idx, dataloader_idx=0):
        if self.config.dataloader_type == 'iterative':
            batch_X, batch_Phis, batch_ys = batch
            preds = self.model(batch_X, batch_Phis)  # list of tensors
            return preds

        elif self.config.dataloader_type == 'tensor':
            batch_X, batch_Phi, batch_y, batch_N = batch
            pred = self.model(batch_X, batch_Phi)
            return pred  # 2D tensor

    def training_step(self, batch, batch_idx):

        if self.config.dataloader_type == 'iterative':
            batch_X, batch_Phis, batch_ys = batch
            preds = self.model(batch_X, batch_Phis)
            losses = [self.loss_fn(pred, y)
                      for pred, y in zip(preds, batch_ys)]
            loss = torch.mean(torch.stack(losses))

        elif self.config.dataloader_type == 'tensor':
            batch_X, batch_Phi, batch_y, batch_N = batch
            pred = self.model(batch_X, batch_Phi)
            loss = torch.sum(torch.sum(((pred - batch_y) ** 2),
                             dim=1) / batch_N) / batch_X.shape[0]

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):

        if self.config.dataloader_type == 'iterative':
            batch_X, batch_Phis, batch_ys = batch
            preds = self.model(batch_X, batch_Phis)
            losses = [self.loss_fn(pred, y)
                      for pred, y in zip(preds, batch_ys)]
            loss = torch.mean(torch.stack(losses))

        elif self.config.dataloader_type == 'tensor':
            batch_X, batch_Phi, batch_y, batch_N = batch
            pred = self.model(batch_X, batch_Phi)
            loss = torch.sum(torch.sum(((pred - batch_y) ** 2),
                             dim=1) / batch_N) / batch_X.shape[0]

        self.log('val_loss', loss)

        return loss

    def test_step(self, batch, batch_idx):

        if self.config.dataloader_type == 'iterative':
            batch_X, batch_Phis, batch_ys = batch
            preds = self.model(batch_X, batch_Phis)
            losses = [self.loss_fn(pred, y)
                      for pred, y in zip(preds, batch_ys)]
            loss = torch.mean(torch.stack(losses))

        elif self.config.dataloader_type == 'tensor':
            batch_X, batch_Phi, batch_y, batch_N = batch
            pred = self.model(batch_X, batch_Phi)
            loss = torch.sum(torch.sum(((pred - batch_y) ** 2),
                             dim=1) / batch_N) / batch_X.shape[0]

        self.log('test_loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = OPTIMIZERS[self.config.training.optimizer](self.model.parameters(
        ), lr=self.config.training.lr, weight_decay=self.config.training.weight_decay)
        return optimizer

class LitTTSDynamic(pl.LightningModule):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.model = TTSDynamic(config)
        self.loss_fn = torch.nn.MSELoss()
        self.lr = self.config.training.lr
        self.contrastive_loss_weight = 0.1

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # def forward(self, batch, batch_idx, dataloader_idx=0):
        if self.config.dataloader_type == 'iterative':
            batch_X, batch_X_dynamic, batch_Phis, batch_ys = batch
            preds, _ = self.model(batch_X, batch_X_dynamic, batch_Phis)  # list of tensors
            return preds

        elif self.config.dataloader_type == 'tensor':
            batch_X, batch_X_dynamic, batch_Phi, batch_y, batch_N = batch
            pred = self.model(batch_X, batch_X_dynamic, batch_Phi)
            return pred  # 2D tensor

    def training_step(self, batch, batch_idx):

        if self.config.dataloader_type == 'iterative':
            batch_X, batch_X_dynamic, batch_Phis, batch_ys = batch
            preds, contrastive_loss = self.model(batch_X, batch_X_dynamic, batch_Phis)
            losses = [self.loss_fn(pred, y)
                      for pred, y in zip(preds, batch_ys)]
            loss = torch.mean(torch.stack(losses))
            total_loss = loss + self.contrastive_loss_weight * contrastive_loss

        elif self.config.dataloader_type == 'tensor':
            batch_X, batch_X_dynamic, batch_Phi, batch_y, batch_N = batch
            pred, contrastive_loss = self.model(batch_X, batch_X_dynamic, batch_Phi)
            loss = torch.sum(torch.sum(((pred - batch_y) ** 2),
                             dim=1) / batch_N) / batch_X.shape[0]
            total_loss = loss + self.contrastive_loss_weight * contrastive_loss

        self.log('train_loss', total_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):

        if self.config.dataloader_type == 'iterative':
            batch_X, batch_X_dynamic, batch_Phis, batch_ys = batch
            preds, _ = self.model(batch_X, batch_X_dynamic, batch_Phis)
            losses = [self.loss_fn(pred, y)
                      for pred, y in zip(preds, batch_ys)]
            loss = torch.mean(torch.stack(losses))

        elif self.config.dataloader_type == 'tensor':
            batch_X, batch_X_dynamic, batch_Phi, batch_y, batch_N = batch
            pred, _ = self.model(batch_X, batch_X_dynamic, batch_Phi)
            loss = torch.sum(torch.sum(((pred - batch_y) ** 2),
                             dim=1) / batch_N) / batch_X.shape[0]

        self.log('val_loss', loss)

        return loss

    def test_step(self, batch, batch_idx):

        if self.config.dataloader_type == 'iterative':
            batch_X, batch_X_dynamic, batch_Phis, batch_ys = batch
            preds, _ = self.model(batch_X, batch_X_dynamic, batch_Phis)
            losses = [self.loss_fn(pred, y)
                      for pred, y in zip(preds, batch_ys)]
            loss = torch.mean(torch.stack(losses))

        elif self.config.dataloader_type == 'tensor':
            batch_X, batch_X_dynamic, batch_Phi, batch_y, batch_N = batch
            pred, _ = self.model(batch_X, batch_X_dynamic, batch_Phi)
            loss = torch.sum(torch.sum(((pred - batch_y) ** 2),
                             dim=1) / batch_N) / batch_X.shape[0]

        self.log('test_loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = OPTIMIZERS[self.config.training.optimizer](self.model.parameters(
        ), lr=self.config.training.lr, weight_decay=self.config.training.weight_decay)
        return optimizer

class LitSemanticTransformer(pl.LightningModule):
    """
    Lightning wrapper for SemanticTransformer.

    Supports two batch formats:
      1) Dict batches (preferred):
         {
           "static": Tensor(B, n_features),
           "semantics": list[B] of list[(cls, t0, t1, y0, y1)],
           "t":      Tensor(B, T) or Tensor(T,),
           "y":      Tensor(B, T),
         }

      2) Legacy tuple batches (iterative / tensor), kept for backward compatibility.
    """

    def __init__(self, config: "Config", model: Optional[torch.nn.Module] = None):
        super().__init__()
        self.config = config
        self.model = model if model is not None else SemanticTransformer(config)
        self.loss_fn = torch.nn.MSELoss()
        self.lr = float(self.config.training.lr)

    # -------------------------
    # Legacy tuple-batch helpers
    # -------------------------
    def _unpack_iterative(self, batch):
        # (X, semantics, Phis, ys) OR (X, motif_class, semantic_cont, Phis, ys)
        if len(batch) == 4:
            batch_X, batch_semantics, batch_Phis, batch_ys = batch
            return batch_X, batch_semantics, None, batch_Phis, batch_ys
        if len(batch) == 5:
            batch_X, batch_motif_class, batch_semantic_cont, batch_Phis, batch_ys = batch
            return batch_X, batch_motif_class, batch_semantic_cont, batch_Phis, batch_ys
        raise ValueError(f"Iterative batch must have length 4 or 5, got {len(batch)}.")

    def _unpack_tensor(self, batch):
        # (X, semantics, Phi, y, N) OR (X, motif_class, semantic_cont, Phi, y, N)
        if len(batch) == 5:
            batch_X, batch_semantics, batch_Phi, batch_y, batch_N = batch
            return batch_X, batch_semantics, None, batch_Phi, batch_y, batch_N
        if len(batch) == 6:
            batch_X, batch_motif_class, batch_semantic_cont, batch_Phi, batch_y, batch_N = batch
            return batch_X, batch_motif_class, batch_semantic_cont, batch_Phi, batch_y, batch_N
        raise ValueError(f"Tensor batch must have length 5 or 6, got {len(batch)}.")

    # -------------------------
    # Dict-batch helpers
    # -------------------------
    @staticmethod
    def _semantics_to_tensors(
        semantics: List[List[Tuple[int, float, float, float, float]]],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pads ragged semantic lists into fixed tensors.

        Returns:
          motif_class: (B, Kmax) long
          semantic_vals:  (B, Kmax, 4) float   [t0, t1, y0, y1]
          pad_mask:    (B, Kmax) bool      True means padding
        """
        B = len(semantics)
        Kmax = max((len(s) for s in semantics), default=0)

        motif_class = torch.zeros(B, Kmax, dtype=torch.long, device=device)
        semantic_vals = torch.zeros(B, Kmax, 4, dtype=torch.float32, device=device)
        pad_mask = torch.ones(B, Kmax, dtype=torch.bool, device=device)

        for i, mi in enumerate(semantics):
            k = len(mi)
            if k == 0:
                continue
            motif_class[i, :k] = torch.tensor([m[0] for m in mi], dtype=torch.long, device=device)
            semantic_vals[i, :k] = torch.tensor([m[1:5] for m in mi], dtype=torch.float32, device=device)
            pad_mask[i, :k] = False

        return motif_class, semantic_vals, pad_mask

    def _forward_from_dict_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        x_static = batch["static"].to(self.device)
        semantics = batch["semantics"]
        t = batch["t"].to(self.device)
        target = batch["y"].to(self.device)

        motif_class, semantic_vals, pad_mask = self._semantics_to_tensors(semantics, x_static.device)

        pred = self.model(
            x_static=x_static,
            motif_class=motif_class,
            semantic_vals=semantic_vals,
            t=t,
            semantic_key_padding_mask=pad_mask,
        )
        return pred, target

    # -------------------------
    # Lightning hooks
    # -------------------------
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, dict):
            pred, _ = self._forward_from_dict_batch(batch)
            return pred

        if self.config.dataloader_type == "iterative":
            batch_X, a, b, batch_Phis, _ = self._unpack_iterative(batch)
            return self.model(batch_X, a, batch_Phis) if b is None else self.model(batch_X, a, b, batch_Phis)

        if self.config.dataloader_type == "tensor":
            batch_X, a, b, batch_Phi, _, _ = self._unpack_tensor(batch)
            return self.model(batch_X, a, batch_Phi) if b is None else self.model(batch_X, a, b, batch_Phi)

        raise ValueError("dataloader_type must be one of ['iterative','tensor'].")

    def training_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            pred, target = self._forward_from_dict_batch(batch)
            loss = self.loss_fn(pred, target)
            self.log("train_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
            return loss

        if self.config.dataloader_type == "iterative":
            batch_X, a, b, batch_Phis, batch_ys = self._unpack_iterative(batch)
            preds = self.model(batch_X, a, batch_Phis) if b is None else self.model(batch_X, a, b, batch_Phis)
            loss = torch.mean(torch.stack([self.loss_fn(pred, y) for pred, y in zip(preds, batch_ys)]))
            self.log("train_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
            return loss

        if self.config.dataloader_type == "tensor":
            batch_X, a, b, batch_Phi, batch_y, batch_N = self._unpack_tensor(batch)
            pred = self.model(batch_X, a, batch_Phi) if b is None else self.model(batch_X, a, b, batch_Phi)
            loss = torch.sum(torch.sum(((pred - batch_y) ** 2), dim=1) / batch_N) / batch_X.shape[0]
            self.log("train_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
            return loss

        raise ValueError("dataloader_type must be one of ['iterative','tensor'].")

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            pred, target = self._forward_from_dict_batch(batch)
            loss = self.loss_fn(pred, target)
            self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
            return loss

        if self.config.dataloader_type == "iterative":
            batch_X, a, b, batch_Phis, batch_ys = self._unpack_iterative(batch)
            preds = self.model(batch_X, a, batch_Phis) if b is None else self.model(batch_X, a, b, batch_Phis)
            loss = torch.mean(torch.stack([self.loss_fn(pred, y) for pred, y in zip(preds, batch_ys)]))
            self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
            return loss

        if self.config.dataloader_type == "tensor":
            batch_X, a, b, batch_Phi, batch_y, batch_N = self._unpack_tensor(batch)
            pred = self.model(batch_X, a, batch_Phi) if b is None else self.model(batch_X, a, b, batch_Phi)
            loss = torch.sum(torch.sum(((pred - batch_y) ** 2), dim=1) / batch_N) / batch_X.shape[0]
            self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
            return loss

        raise ValueError("dataloader_type must be one of ['iterative','tensor'].")

    def test_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            pred, target = self._forward_from_dict_batch(batch)
            loss = self.loss_fn(pred, target)
            self.log("test_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
            return loss

        if self.config.dataloader_type == "iterative":
            batch_X, a, b, batch_Phis, batch_ys = self._unpack_iterative(batch)
            preds = self.model(batch_X, a, batch_Phis) if b is None else self.model(batch_X, a, b, batch_Phis)
            loss = torch.mean(torch.stack([self.loss_fn(pred, y) for pred, y in zip(preds, batch_ys)]))
            self.log("test_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
            return loss

        if self.config.dataloader_type == "tensor":
            batch_X, a, b, batch_Phi, batch_y, batch_N = self._unpack_tensor(batch)
            pred = self.model(batch_X, a, batch_Phi) if b is None else self.model(batch_X, a, b, batch_Phi)
            loss = torch.sum(torch.sum(((pred - batch_y) ** 2), dim=1) / batch_N) / batch_X.shape[0]
            self.log("test_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
            return loss

        raise ValueError("dataloader_type must be one of ['iterative','tensor'].")

    def configure_optimizers(self):
        optimizer = OPTIMIZERS[self.config.training.optimizer](
            self.model.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
        )
        return optimizer
