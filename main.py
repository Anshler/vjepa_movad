"""
Entry point: V-JEPA 2.1 + MOVAD training / evaluation.

Supports four temporal model variants via config key ``temporal_model``:
  - ``lstm`` (default)
  - ``mamba``
  - ``slotssm``
  - ``sparse_slotssm``

Usage
-----
Train:
    python main.py --config cfgs/vjepa_v1.yaml --phase train --epochs 200

Test:
    python main.py --config cfgs/vjepa_v1.yaml --phase test --epoch 190

Resume:
    python main.py --config cfgs/vjepa_v1.yaml --phase train --epoch 50
"""
from __future__ import annotations

import argparse
import datetime
import os
import random
import sys
from contextlib import nullcontext

import numpy as np
import torch
import yaml
from easydict import EasyDict
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Ensure the repo root is on sys.path for sibling imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# Compatibility: torchvision >=0.20 moved functional_tensor to _functional_tensor
# pytorchvideo 0.1.5 still imports the old path.
try:
    import torchvision.transforms._functional_tensor as _ft
    sys.modules.setdefault("torchvision.transforms.functional_tensor", _ft)
except ImportError:
    pass

from movad_core.dota import Dota, gt_cls_target, setup_dota
from movad_core.losses import build_loss
from movad_core.metrics import evaluation, print_results
from movad_core.optim import build_optimizer
from movad_core import utils as movad_utils
from model import build_cls_vjepa


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_configs():
    parser = argparse.ArgumentParser(description="V-JEPA 2.1 + MOVAD anomaly detection")
    parser.add_argument("--config", default="cfgs/vjepa_v1.yaml", help="YAML config file")
    parser.add_argument("--phase", default="train", choices=["train", "test"], help="train or test")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--snapshot_interval", type=int, default=10)
    parser.add_argument("--epoch", type=int, default=-1, help="Resume from (train) or eval (test)")
    parser.add_argument("--output", default="./output/vjepa_v1", help="Output directory")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = EasyDict(yaml.safe_load(f))
    cfg.update(vars(args))

    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return cfg


def set_deterministic(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(cfg, model, traindata_loader, optimizer, lr_scheduler, begin_epoch):
    writer = SummaryWriter(
        os.path.join(cfg.output, "tensorboard", f"train_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}")
    )
    os.makedirs(os.path.join(cfg.output, "checkpoints"), exist_ok=True)
    with open(os.path.join(cfg.output, "cfg.yml"), "w") as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)

    criterion = build_loss(cfg)
    fb = cfg.NF

    index_guess = 0
    index_loss = 0

    _amp_cfg = cfg.get("amp_dtype", "fp32")
    _amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(_amp_cfg)
    autocast_ctx = torch.amp.autocast("cuda", dtype=_amp_dtype) if _amp_dtype else nullcontext()

    # -------------------------------------------------------------------
    # Per-video temporal loop (extracted so pre-batched + standard paths
    # can share it without duplication).
    # -------------------------------------------------------------------
    def _run_temporal_loop(
        model, patches_clips, data_info,
        fb, v_len,
        cfg, criterion, optimizer, autocast_ctx,
    ):
        """Run stride-1 sliding-window temporal loop over one video batch.

        ``patches_clips``:  ``[B, n_clips, embed_dim]``  (standard, mean-pooled)
                          or ``[B, n_clips, N_patches, embed_dim]``  (slot-based).

        Returns (avg_loss, frame_count, slot_diag, targets, outputs, index_loss).
        """
        toa_batch = data_info[:, 2]
        tea_batch = data_info[:, 3]
        video_len_orig = data_info[:, 0]

        t_shape = (data_info.shape[0], v_len - fb)
        targets = torch.full(t_shape, -100).to(data_info.device)
        outputs_t = torch.full(t_shape, -100, dtype=torch.float).to(data_info.device)

        state = None
        slot_diag = {}
        running_loss = 0.0
        frame_count = 0
        _idx_loss = 0

        for i in range(fb, v_len):
            target = gt_cls_target(i, toa_batch, tea_batch).long()

            with autocast_ctx:
                feat = patches_clips[:, i - fb, ...]                 # [B, embed_dim] or [B, N, embed_dim]
                output, state = model.forward_temporal_step(feat, state)

            flt = i >= video_len_orig
            target[flt] = -100
            output[flt] = -100

            if cfg.get("apply_softmax", True):
                output = output.softmax(dim=1)

            optimizer.zero_grad()
            loss = criterion(output, target)

            # Entropy penalty for sparse SlotSSM
            entropy_weight = cfg.get("entropy_weight", 0.0)
            if entropy_weight > 0 and hasattr(model, "temporal") and hasattr(model.temporal, "_entropy"):
                ent = model.temporal._entropy
                if isinstance(ent, torch.Tensor) and ent.item() > 0:
                    loss = loss + entropy_weight * (-ent)

            # Slot cross-attn diagnostics (inverted only)
            if hasattr(model, "temporal") and hasattr(model.temporal, "_slot_mass_min"):
                _mass_min = model.temporal._slot_mass_min
                if not (isinstance(_mass_min, torch.Tensor) and torch.isnan(_mass_min)):
                    slot_diag["mass_min"] = min(slot_diag.get("mass_min", 999.0), float(_mass_min))
                    slot_diag["mass_mean"] = slot_diag.get("mass_mean", 0.0) + float(model.temporal._slot_mass_mean)
                    slot_diag["usage_frac"] = min(slot_diag.get("usage_frac", 999.0), float(model.temporal._slot_usage_frac))
                    slot_diag["_count"] = slot_diag.get("_count", 0) + 1

            loss.backward()
            optimizer.step()

            _idx_loss += 1
            running_loss += loss.item()
            frame_count += 1
            targets[:, i - fb] = target.clone()
            out = output.max(1)[1]
            out[target == -100] = -100
            outputs_t[:, i - fb] = out

        return running_loss, frame_count, slot_diag, targets, outputs_t, _idx_loss

    model.train(True)
    is_slot_based = model._slot_based

    for e in range(begin_epoch, cfg.epochs):
        epoch_total_loss = 0.0
        epoch_frames = 0

        pbar = tqdm(enumerate(traindata_loader), total=len(traindata_loader), desc=f"Epoch {e+1}/{cfg.epochs}")
        for j, (video_data, data_info) in pbar:  # noqa: B007
            video_data = video_data.to(cfg.device, non_blocking=True)
            data_info = data_info.to(cfg.device, non_blocking=True)
            video_data = torch.swapaxes(video_data, 1, 2)            # [B, C, F, H, W]
            v_len = video_data.shape[2]

            # Pre-compute frozen encoder: [B, n_clips, N_patches, embed_dim]
            patches = model.encode_video_clips(video_data, fb)

            if not is_slot_based:
                # Standard path: spatially mean-pool → [B, n_clips, embed_dim]
                patches = patches.mean(dim=2)
            # Slot path: keep full patches [B, n_clips, N_patches, embed_dim]

            # --- Temporal loop -----------------------------------------
            running_loss, f_count, slot_diag, targets, outputs, dl = _run_temporal_loop(
                model, patches, data_info,
                fb, v_len,
                cfg, criterion, optimizer, autocast_ctx,
            )
            index_loss += dl

            # --- Per-video logging -------------------------------------
            avg_loss = running_loss / max(f_count, 1)
            writer.add_scalar("train/loss", avg_loss, index_guess)

            postfix_parts = [f"loss={avg_loss:.4f}"]
            if slot_diag:
                n = slot_diag.pop("_count", 1)
                writer.add_scalar("slots/mass_min", slot_diag["mass_min"], index_guess)
                writer.add_scalar("slots/mass_mean", slot_diag["mass_mean"] / n, index_guess)
                writer.add_scalar("slots/usage_frac", slot_diag["usage_frac"], index_guess)
                postfix_parts.extend([
                    f"mass_min={slot_diag['mass_min']:.4f}",
                    f"mass_avg={slot_diag['mass_mean']/n:.4f}",
                    f"use={slot_diag['usage_frac']:.2f}",
                ])
            pbar.set_postfix_str(" ".join(postfix_parts))

            epoch_total_loss += running_loss
            epoch_frames += max(f_count, 1)

            outputs = outputs[outputs != -100]
            targets = targets[targets != -100]
            index_guess += 1

        epoch_avg_loss = epoch_total_loss / max(epoch_frames, 1)
        writer.add_scalar("train/epoch_loss", epoch_avg_loss, e)
        print(f"  Epoch {e+1}/{cfg.epochs} avg loss: {epoch_avg_loss:.4f}  ({epoch_frames} frames in {epoch_total_loss:.1f} total)")

        if lr_scheduler is not None:
            lr_scheduler.step()

        if (e + 1) % cfg.snapshot_interval == 0:
            ckpt_path = os.path.join(cfg.output, "checkpoints", f"model-{e+1:02d}.pt")
            torch.save(
                {
                    "epoch": e,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler else None,
                    "index_guess": index_guess,
                    "index_loss": index_loss,
                },
                ckpt_path,
            )
            print(f"  checkpoint saved → {ckpt_path}")

    writer.close()


# ---------------------------------------------------------------------------
# Testing loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def test(cfg, model, testdata_loader, epoch, filename):
    fb = cfg.NF
    _amp_cfg = cfg.get("amp_dtype", "fp32")
    _amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(_amp_cfg)
    autocast_ctx = torch.amp.autocast("cuda", dtype=_amp_dtype) if _amp_dtype else nullcontext()

    targets_all, outputs_all = [], []
    toas_all, teas_all, idxs_all, info_all, frames_counter = [], [], [], [], []

    model.eval()
    for video_data, data_info in tqdm(testdata_loader, desc=f"Test epoch {epoch}"):
        video_data = video_data.to(cfg.device, non_blocking=True)
        data_info = data_info.to(cfg.device, non_blocking=True)

        video_data = torch.swapaxes(video_data, 1, 2)

        t_shape = (video_data.shape[0], video_data.shape[2] - fb)
        targets = torch.full(t_shape, -100).to(video_data.device)
        outputs = torch.full(t_shape, -100, dtype=torch.float).to(video_data.device)

        idx_batch = data_info[:, 1]
        toa_batch = data_info[:, 2]
        tea_batch = data_info[:, 3]
        info_batch = data_info[:, 7:11]

        # Pre-compute frozen encoder: [1, n_clips, N_patches, embed_dim]
        patches = model.encode_video_clips(video_data, fb)
        if not model._slot_based:
            patches = patches.mean(dim=2)  # spatial pool

        state = None
        for i in range(fb, video_data.shape[2]):
            target = gt_cls_target(i, toa_batch, tea_batch).long()

            with autocast_ctx:
                feat = patches[:, i - fb, ...]
                output, state = model.forward_temporal_step(feat, state)

            if cfg.get("apply_softmax", True):
                output = output.softmax(dim=1)

            targets[:, i - fb] = target.clone()
            outputs[:, i - fb] = output[:, 1].clone()

        targets_all.append(targets.view(-1).tolist())
        outputs_all.append(outputs.view(-1).tolist())
        toas_all.append(toa_batch.tolist())
        teas_all.append(tea_batch.tolist())
        idxs_all.append(idx_batch.tolist())
        info_all.append(info_batch.tolist())
        frames_counter.append(video_data.shape[2])

    import pickle

    print(f"  saving results → {filename}")
    with open(filename, "wb") as f:
        pickle.dump(
            {
                "targets": targets_all,
                "outputs": outputs_all,
                "toas": np.array(toas_all).reshape(-1),
                "teas": np.array(teas_all).reshape(-1),
                "idxs": np.array(idxs_all).reshape(-1),
                "info": np.array(info_all).reshape(-1, 4),
                "frames_counter": np.array(frames_counter).reshape(-1),
            },
            f,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = parse_configs()
    set_deterministic(cfg.seed)

    # Derive NF from num_frames so they can't drift apart
    if "NF" not in cfg:
        cfg.NF = cfg.num_frames

    # --- Data --------------------------------------------------------------
    traindata_loader, testdata_loader = setup_dota(
        Dota,
        cfg,
        num_workers=cfg.num_workers,
        VCL=cfg.get("VCL", None),
        phase=cfg.phase,
    )

    # --- Model -------------------------------------------------------------
    checkpoint = None
    epoch = 0

    model = build_cls_vjepa(cfg)
    print(f"Temporal model: {cfg.get('temporal_model', 'lstm')}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    if cfg.epoch != -1:
        try:
            checkpoint = movad_utils.load_checkpoint(cfg)
            model.load_state_dict(checkpoint["model_state_dict"])
            epoch = checkpoint["epoch"] + 1
            print(f"Resumed from epoch {epoch}")
        except FileNotFoundError:
            print(f"No checkpoint found at epoch {cfg.epoch} — starting fresh")
            epoch = cfg.epoch if cfg.epoch > 0 else 0

    if cfg.phase == "train":
        optimizer, lr_scheduler = build_optimizer(cfg, model, checkpoint)
        train(cfg, model, traindata_loader, optimizer, lr_scheduler, epoch)

    elif cfg.phase == "test":
        filename = movad_utils.get_result_filename(cfg, epoch)
        if not os.path.exists(filename):
            test(cfg, model, testdata_loader, epoch, filename)

        content = movad_utils.load_results(filename)
        print_results(cfg, *evaluation(FPS=cfg.FPS, **content))
