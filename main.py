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
import glob
import json
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
from model import build_multi_head_vjepa


# ---------------------------------------------------------------------------
# Precomputed validation dataset — loads cached encoder embeddings from disk
# to skip the frozen ViT forward pass during validation.
# ---------------------------------------------------------------------------
def pad_collate_embeddings(batch):
    """Collate precomputed embeddings — pad ``patches_full`` to max clips.

    Each sample is a dict with:
        ``patches_full`` — ``[n_clips_i, N, D]``  (fp16)
        ``data_info``   — ``[11]``                (fp32)
        ``v_len``       — int

    Returns a dict with tensors stacked into ``[B, ...]``.

    IMPORTANT: The padded tensor stays in fp16 to keep CPU→GPU transfer
    memory in check.  ``patches.mean(dim=2)`` and the temporal model's
    LayerNorm+Mamba blocks handle fp16 input without issues.
    """
    max_clips = max(b["patches_full"].shape[0] for b in batch)
    N = batch[0]["patches_full"].shape[1]
    D = batch[0]["patches_full"].shape[2]
    B = len(batch)
    padded = torch.zeros(B, max_clips, N, D, dtype=torch.float16)
    data_infos = torch.zeros(B, 11)
    v_lens = torch.zeros(B, dtype=torch.long)
    for i, item in enumerate(batch):
        n = item["patches_full"].shape[0]
        padded[i, :n] = item["patches_full"]
        data_infos[i] = item["data_info"]
        v_lens[i] = item["v_len"]
    return {"patches_full": padded, "data_info": data_infos, "v_len": v_lens}


class PrecomputedValDataset(torch.utils.data.Dataset):
    """Loads precomputed V-JEPA encoder embeddings (``.pt`` files), sorted by
    video length so adjacent samples in a DataLoader batch have similar
    durations — minimising padding waste during bucket batching.

    Sorted **descending** (longest videos first): the first batch allocates
    the largest CUDA memory block; subsequent smaller batches reuse that
    block without allocator fragmentation.

    Each ``.pt`` file is a dict with keys:
        ``patches_full`` — fp16 ``[n_clips, N_patches, embed_dim]``
        ``data_info``   — fp32 ``[11]``  (same format as raw DoTA)
        ``v_len``       — int, total video frames
    """

    def __init__(self, embed_dir: str, data_path: str):

        self.files = sorted(glob.glob(os.path.join(embed_dir, "*.pt")))
        if not self.files:
            raise FileNotFoundError(f"No *.pt files found in {embed_dir}")

        # Load metadata to sort files by video length — this groups
        # similar-length videos together for efficient batched padding,
        # without needing to open every .pt file.
        metadata_path = os.path.join(data_path, "metadata", "metadata_val.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Sort DESCENDING: longest videos first → CUDA allocator grabs the
        # biggest block upfront and reuses it for all smaller later batches.
        self.files.sort(key=lambda fp: metadata.get(
            os.path.splitext(os.path.basename(fp))[0], {}
        ).get("num_frames", 0), reverse=True)

        self.is_precomputed = True   # flag checked by _evaluate_model

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx], weights_only=True)


def _try_build_precomputed_val_loader(data_path: str, batch_size: int, num_workers: int = 0):
    """Return a DataLoader over precomputed embeddings, or ``None`` if missing."""
    embed_dir = os.path.join(data_path, "embedding_val")
    if not os.path.isdir(embed_dir) or not os.listdir(embed_dir):
        return None
    dataset = PrecomputedValDataset(embed_dir, data_path)
    pin = os.name != "nt"
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin, collate_fn=pad_collate_embeddings,
    )


# ---------------------------------------------------------------------------
# CLI
#
# Single-model (backward-compatible):
#   python main.py --config cfgs/vjepa_v1.yaml --phase train
#
# Multi-head (one encoder → multiple temporal models):
#   python main.py --config cfgs/vjepa_v1.yaml cfgs/vjepa_mamba.yaml cfgs/vjepa_slotssm.yaml --phase train
#
# The first config is the *master* — its encoder, data, training, augmentation,
# and output sections are shared.  Each config contributes its temporal-model
# variant as an independent head.  Head name = config basename minus ``.yaml``.
# ---------------------------------------------------------------------------
def parse_configs():
    parser = argparse.ArgumentParser(description="V-JEPA 2.1 + MOVAD anomaly detection")
    _DEFAULT_CONFIGS = [
        #"cfgs/vjepa_v1.yaml",
        "cfgs/vjepa_mamba.yaml",
        "cfgs/vjepa_slotssm.yaml",
        #"cfgs/vjepa_sparse_slotssm.yaml",
        #"cfgs/vjepa_slotssm_inv.yaml",
        #"cfgs/vjepa_sparse_slotssm_inv.yaml",
    ]
    parser.add_argument("--config", nargs="+", default=_DEFAULT_CONFIGS,
                        help="YAML config(s). First = master (encoder/data/training/...); "
                             "rest = temporal-model heads reusing the master's shared settings.")
    parser.add_argument("--phase", default="train", choices=["train", "test"], help="train or test")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--snapshot_interval", type=int, default=10)
    parser.add_argument("--epoch", type=int, default=-1, help="Resume from (train) or eval (test)")
    parser.add_argument("--enable_validation", action="store_true", default=None,
                        help="Run validation during training (overrides config)")
    parser.add_argument("--no-enable_validation", action="store_false", dest="enable_validation",
                        help="Disable validation during training")
    parser.add_argument("--validation_epoch_step", type=int, default=None,
                        help="Validate every N epochs (overrides config)")
    parser.add_argument("--output", default=None, help="Output directory (default: from first config)")
    args = parser.parse_args()

    # Load all configs
    all_cfgs = []
    for path in args.config:
        with open(path, "r") as f:
            all_cfgs.append(EasyDict(yaml.safe_load(f)))

    # Master config = first one
    cfg = all_cfgs[0]

    # Derive output from master config before CLI args clobber it
    _yaml_output = cfg.get("output", "./output")
    _yaml_workers = cfg.get("num_workers", 0)
    _yaml_enable_val = cfg.get("enable_validation", True)
    _yaml_val_step = cfg.get("validation_epoch_step", 10)

    cfg.update(vars(args))

    if cfg.output is None:
        cfg.output = _yaml_output
    # Keep YAML num_workers unless explicitly overridden via CLI
    if args.num_workers == 0 and _yaml_workers > 0:
        cfg.num_workers = _yaml_workers
    # Keep YAML validation settings unless explicitly overridden via CLI
    if args.enable_validation is None:
        cfg.enable_validation = _yaml_enable_val
    if args.validation_epoch_step is None:
        cfg.validation_epoch_step = _yaml_val_step

    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build head configs — always, even for 1 head
    import pathlib

    cfg._head_names = [pathlib.Path(p).stem for p in args.config]
    head_cfgs_flat = []
    for name, hc in zip(cfg._head_names, all_cfgs):
        entry = dict(hc)
        entry["name"] = name
        head_cfgs_flat.append(entry)
    cfg._head_cfgs_flat = head_cfgs_flat

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
# Checkpoint helpers — strip frozen encoder weights (saves ~600MB per ckpt)
# ---------------------------------------------------------------------------
def _head_state_dict_without_encoder(head):
    """Return ``head.state_dict()`` excluding the shared frozen encoder weights."""
    return {k: v for k, v in head.state_dict().items() if not k.startswith("encoder.")}


def _load_head_state_dict(head, state_dict):
    """Load state dict into a head, skipping encoder keys (frozen, loaded separately)."""
    head.load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# Training loop — one encode → N independent temporal trainings
#
# Works for 1 head or N heads — same code path either way.
# Each head has its own optimizer, checkpoint directory, and TensorBoard
# writer.  Losses are never summed — ``.backward()`` flows only through that
# head's temporal + classifier parameters.  The shared encoder is frozen.
# ---------------------------------------------------------------------------
def train(cfg, model, traindata_loader, begin_epoch,
          testdata_loader=None, validation_epoch_step=10):
    """Train a MultiHeadVJEPA: encode once per batch, then loop heads sequentially.

    Heads are trained **independently** — different losses, different
    optimizers, different checkpoints.  Each head's ``.backward()`` only
    touches that head's parameters.

    If ``testdata_loader`` is provided, validation runs every
    ``validation_epoch_step`` epochs and metrics are logged to tensorboard.
    """
    head_names = list(model.heads.keys())

    # -------------------------------------------------------------------
    # Per-head infrastructure
    # -------------------------------------------------------------------
    head_cfgs: dict[str, dict] = {}
    criterion: dict[str, torch.nn.Module] = {}
    optimizer: dict[str, torch.optim.Optimizer] = {}
    writers: dict[str, SummaryWriter] = {}
    output_dirs: dict[str, str] = {}
    accum_steps: dict[str, int] = {}
    autocast_ctx: dict[str, object] = {}

    for name in head_names:
        hc = model.head_configs.get(name, {})
        head_cfgs[name] = dict(cfg)
        for k in ("dim_latent", "dropout", "rnn_state_size", "rnn_cell_num",
                   "mamba_d_state", "mamba_d_conv", "mamba_expand", "mamba_version",
                   "num_slots", "slot_dim", "num_ssm_blocks", "top_k", "eps_random",
                   "use_inverted_attention", "entropy_weight"):
            if k in hc:
                head_cfgs[name][k] = hc[k]

        output_dir = os.path.join(cfg.output, name)
        output_dirs[name] = output_dir
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
        with open(os.path.join(output_dir, "cfg.yml"), "w") as f:
            yaml.dump(head_cfgs[name], f, default_flow_style=False)

        writer = SummaryWriter(
            os.path.join(output_dir, "tensorboard", f"train_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}")
        )
        writers[name] = writer

        # Build head-specific loss
        head_easy = EasyDict(head_cfgs[name])
        head_easy.device = cfg.device
        criterion[name] = build_loss(head_easy)

        # Build head-specific optimizer (only head params, not shared encoder)
        opt, _ = build_optimizer(
            EasyDict({"lr": cfg.lr}),
            model.heads[name],
            None,
        )
        optimizer[name] = opt

        _amp_cfg = head_cfgs[name].get("amp_dtype", "fp32")
        _amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(_amp_cfg)
        autocast_ctx[name] = torch.amp.autocast("cuda", dtype=_amp_dtype) if _amp_dtype else nullcontext()
        accum_steps[name] = head_cfgs[name].get("grad_accum", 1)

    # -------------------------------------------------------------------
    # Per-video temporal loop — identical to single-model version but
    # parameterised by head name so slot diagnostics route to the right
    # TensorBoard writer.
    # -------------------------------------------------------------------
    def _run_temporal_loop(head, patches_clips, data_info, fb, v_len, head_name):
        toa_batch = data_info[:, 2]
        tea_batch = data_info[:, 3]
        video_len_orig = data_info[:, 0]

        t_shape = (data_info.shape[0], v_len - fb)
        targets = torch.full(t_shape, -100).to(data_info.device)
        outputs_t = torch.full(t_shape, -100, dtype=torch.float).to(data_info.device)

        state = None
        slot_diag = {}
        total_loss = torch.tensor(0.0, device=data_info.device)
        frame_count = 0

        for i in range(fb, v_len):
            target = gt_cls_target(i, toa_batch, tea_batch).long()

            with autocast_ctx[head_name]:
                feat = patches_clips[:, i - fb, ...]
                output, state = head.forward_temporal_step(feat, state)

            flt = i >= video_len_orig
            target[flt] = -100
            output[flt] = -100

            if head_cfgs[head_name].get("apply_softmax", True):
                output = output.softmax(dim=1)

            loss = criterion[head_name](output, target)

            entropy_weight = head_cfgs[head_name].get("entropy_weight", 0.0)
            if entropy_weight > 0 and hasattr(head, "temporal") and hasattr(head.temporal, "_entropy"):
                ent = head.temporal._entropy
                if isinstance(ent, torch.Tensor) and ent.item() > 0:
                    loss = loss + entropy_weight * (-ent)

            if hasattr(head, "temporal") and hasattr(head.temporal, "_slot_mass_min"):
                _mass_min = head.temporal._slot_mass_min
                if not (isinstance(_mass_min, torch.Tensor) and torch.isnan(_mass_min)):
                    slot_diag["mass_min"] = min(slot_diag.get("mass_min", 999.0), float(_mass_min))
                    slot_diag["mass_mean"] = slot_diag.get("mass_mean", 0.0) + float(head.temporal._slot_mass_mean)
                    slot_diag["usage_frac"] = min(slot_diag.get("usage_frac", 999.0), float(head.temporal._slot_usage_frac))
                    slot_diag["_count"] = slot_diag.get("_count", 0) + 1

            total_loss = total_loss + loss
            frame_count += 1
            targets[:, i - fb] = target.clone()
            out = output.max(1)[1]
            out[target == -100] = -100
            outputs_t[:, i - fb] = out

        return total_loss, frame_count, slot_diag, targets, outputs_t

    # -------------------------------------------------------------------
    # Epoch loop
    # -------------------------------------------------------------------
    model.train(True)
    accum_losses: dict[str, torch.Tensor] = {}
    for name in head_names:
        accum_losses[name] = torch.tensor(0.0, device=cfg.device)

    for e in range(begin_epoch, cfg.epochs):
        epoch_losses = {n: 0.0 for n in head_names}
        epoch_frames = {n: 0 for n in head_names}

        loader_iter = iter(traindata_loader)

        video_data, data_info = next(loader_iter)
        video_data = video_data.to(cfg.device, non_blocking=True)
        data_info = data_info.to(cfg.device, non_blocking=True)

        n_batches = len(traindata_loader)
        pbar = tqdm(range(n_batches), desc=f"Epoch {e+1}/{cfg.epochs}")
        global_idx = 0

        for j in pbar:
            video_data = torch.swapaxes(video_data, 1, 2)
            v_len = video_data.shape[2]

            if j < n_batches - 1:
                next_video, next_info = next(loader_iter)
                next_video = next_video.to(cfg.device, non_blocking=True)
                next_info = next_info.to(cfg.device, non_blocking=True)

            # Encoder runs ONCE for all heads
            patches = model.encode_video_clips(video_data, cfg.NF)

            # Train each head sequentially — independent forward + backward
            postfix_parts = []
            for name in head_names:
                head = model.heads[name]
                is_slot = head._slot_based

                patches_in = patches if is_slot else patches.mean(dim=2)

                total_loss, f_count, slot_diag, _, _ = _run_temporal_loop(
                    head, patches_in, data_info, cfg.NF, v_len, name,
                )

                epoch_losses[name] += total_loss.item()
                epoch_frames[name] += max(f_count, 1)

                accum_losses[name] = accum_losses[name] + (total_loss / accum_steps[name])

                step_now = ((j + 1) % accum_steps[name] == 0) or (j == n_batches - 1)
                if step_now:
                    optimizer[name].zero_grad()
                    accum_losses[name].backward()
                    optimizer[name].step()
                    accum_losses[name] = torch.tensor(0.0, device=cfg.device)

                avg_loss = total_loss.item() / max(f_count, 1)
                writers[name].add_scalar("train/loss_step", avg_loss, global_idx)
                postfix_parts.append(f"{name}:{avg_loss:.3f}")

                if slot_diag:
                    n = slot_diag.pop("_count", 1)
                    writers[name].add_scalar("slots/mass_min", slot_diag["mass_min"], global_idx)
                    writers[name].add_scalar("slots/mass_mean", slot_diag["mass_mean"] / n, global_idx)
                    writers[name].add_scalar("slots/usage_frac", slot_diag["usage_frac"], global_idx)

            pbar.set_postfix_str(" ".join(postfix_parts))
            global_idx += 1

            if j < n_batches - 1:
                video_data = next_video
                data_info = next_info

        # End-of-epoch logging
        print(f"  Epoch {e+1}/{cfg.epochs}")
        for name in head_names:
            e_loss = epoch_losses[name] / max(epoch_frames[name], 1)
            writers[name].add_scalar("train/epoch_loss", e_loss, e)
            print(f"    {name}: avg_loss={e_loss:.4f}  ({epoch_frames[name]} frames)")

        # Checkpoint
        if (e + 1) % cfg.snapshot_interval == 0:
            for name in head_names:
                ckpt_path = os.path.join(output_dirs[name], "checkpoints", f"model-{e+1:02d}.pt")
                torch.save(
                    {
                        "epoch": e,
                        "model_state_dict": _head_state_dict_without_encoder(model.heads[name]),
                        "optimizer_state_dict": optimizer[name].state_dict(),
                    },
                    ckpt_path,
                )
                print(f"    {name} checkpoint → {ckpt_path}")

        # Validation — free training tensors first to avoid stacking
        if testdata_loader is not None and (e + 1) % validation_epoch_step == 0:
            torch.cuda.empty_cache()          # flush training tensors before eval
            print(f"\n  === Validation at epoch {e+1} ===")
            _evaluate_model(cfg, model, testdata_loader, e + 1, writers=writers)
            torch.cuda.empty_cache()          # flush eval tensors before next epoch
            model.train(True)

    for w in writers.values():
        w.close()

    # Save final checkpoint if not already saved at the last epoch
    if cfg.epochs % cfg.snapshot_interval != 0:
        for name in head_names:
            ckpt_path = os.path.join(output_dirs[name], "checkpoints", f"model-{cfg.epochs:02d}.pt")
            torch.save(
                {
                    "epoch": cfg.epochs - 1,
                    "model_state_dict": _head_state_dict_without_encoder(model.heads[name]),
                    "optimizer_state_dict": optimizer[name].state_dict(),
                },
                ckpt_path,
            )
            print(f"    {name} final checkpoint → {ckpt_path}")


# ---------------------------------------------------------------------------
# Shared evaluation helper — one encode → evaluate all heads sequentially
# Used by both train-time validation and standalone testing.
# ---------------------------------------------------------------------------
@torch.no_grad()
def _evaluate_model(cfg, model, testdata_loader, epoch, writers=None):
    """Run validation/test inference and compute metrics for all heads.

    Parameters
    ----------
    cfg : EasyDict
    model : MultiHeadVJEPA
    testdata_loader : DataLoader
    epoch : int — current epoch number (used for result filenames and logging step)
    writers : dict[str, SummaryWriter] | None — if provided, metrics are logged
              to tensorboard under ``val/<metric>`` for each head.

    Returns
    -------
    dict[str, dict] — per-head metrics dict as returned by ``evaluation()``.
    """
    fb = cfg.NF
    head_names = list(model.heads.keys())

    per_head = {
        n: {
            "targets_all": [], "outputs_all": [], "toas_all": [],
            "teas_all": [], "idxs_all": [], "info_all": [], "frames_counter": [],
        }
        for n in head_names
    }

    model.eval()

    # Detect precomputed embedding loader (dataset has an .is_precomputed flag)
    _precomputed = getattr(testdata_loader.dataset, "is_precomputed", False)

    for batch_idx, batch in enumerate(tqdm(testdata_loader, desc=f"Val epoch {epoch}")):
        if _precomputed:
            # Collate has already padded patches_full [B, max_clips, N, D] fp16.
            # Keep fp16 on GPU — the temporal model's LayerNorm+Mamba handle it.
            data = batch
            B = data["patches_full"].shape[0]
            patches = data["patches_full"].to(cfg.device, non_blocking=True)
            data_info = data["data_info"].to(cfg.device, non_blocking=True)
            v_lens = data["v_len"].to(cfg.device)  # [B] original frame counts
            v_len_max = int(v_lens.max())           # max padded length in bucket
        else:
            video_data, data_info = batch
            video_data = video_data.to(cfg.device, non_blocking=True)
            data_info = data_info.to(cfg.device, non_blocking=True)
            video_data = torch.swapaxes(video_data, 1, 2)
            B = video_data.shape[0]
            v_len_max = video_data.shape[2]         # already padded to max in bucket
            patches = model.encode_video_clips(video_data, fb)

        video_len_orig = data_info[:, 0]            # [B] — truth for masking
        idx_batch = data_info[:, 1]
        toa_batch = data_info[:, 2]
        tea_batch = data_info[:, 3]
        info_batch = data_info[:, 7:11]

        for name in head_names:
            head = model.heads[name]
            is_slot = head._slot_based
            patches_in = patches if is_slot else patches.mean(dim=2)

            t_shape = (B, v_len_max - fb)
            targets = torch.full(t_shape, -100).to(cfg.device)
            outputs = torch.full(t_shape, -100, dtype=torch.float).to(cfg.device)

            state = None
            for i in range(fb, v_len_max):
                target = gt_cls_target(i, toa_batch, tea_batch).long()
                feat = patches_in[:, i - fb, ...].float()   # fp16→fp32 for LayerNorm
                output, state = head.forward_temporal_step(feat, state)

                # Mask padded positions — identical pattern to training loop
                flt = i >= video_len_orig
                target[flt] = -100
                output[flt] = -100

                if cfg.get("apply_softmax", True):
                    output = output.softmax(dim=1)

                targets[:, i - fb] = target.clone()
                outputs[:, i - fb] = output[:, 1].clone()

            res = per_head[name]
            # Append per-video results, filtering out padded frames
            for b in range(B):
                vl = int(video_len_orig[b].item())
                valid_frames = vl - fb
                if valid_frames <= 0:
                    continue
                res["targets_all"].append(targets[b, :valid_frames].tolist())
                res["outputs_all"].append(outputs[b, :valid_frames].tolist())
                res["toas_all"].append(toa_batch[b].item())
                res["teas_all"].append(tea_batch[b].item())
                res["idxs_all"].append(idx_batch[b].item())
                res["info_all"].append(info_batch[b].tolist())
                res["frames_counter"].append(vl)

        # Free GPU tensors from this batch.  With descending sort the
        # largest block was allocated first; subsequent batches are smaller
        # and reuse it without fragmentation.
        if _precomputed:
            del patches
        # Periodically flush the CUDA caching allocator to release any
        # cached-but-unusable blocks back to the OS.
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()

    # Save and evaluate per-head
    import pickle

    all_metrics = {}
    for name in head_names:
        res = per_head[name]
        head_output = os.path.join(cfg.output, name)
        eval_dir = os.path.join(head_output, "eval")
        os.makedirs(eval_dir, exist_ok=True)
        filename = os.path.join(eval_dir, f"results-{epoch:02d}.pkl")

        with open(filename, "wb") as f:
            pickle.dump(
                {
                    "targets": res["targets_all"],
                    "outputs": res["outputs_all"],
                    "toas": np.array(res["toas_all"]).reshape(-1),
                    "teas": np.array(res["teas_all"]).reshape(-1),
                    "idxs": np.array(res["idxs_all"]).reshape(-1),
                    "info": np.array(res["info_all"]).reshape(-1, 4),
                    "frames_counter": np.array(res["frames_counter"]).reshape(-1),
                },
                f,
            )

        content = movad_utils.load_results(filename)
        (auc_roc, auc_pr, f1_one, f1_mean, accuracy,
         report, eval_per_class, eval_per_class_ego) = evaluation(FPS=cfg.FPS, **content)
        all_metrics[name] = {
            "auc_roc": auc_roc, "auc_pr": auc_pr, "f1": f1_one,
            "f1_mean": f1_mean, "accuracy": accuracy,
        }

        # Log to tensorboard if writers provided (train-time validation)
        if writers is not None and name in writers:
            writers[name].add_scalar("val/auc_roc", auc_roc, epoch)
            writers[name].add_scalar("val/auc_pr", auc_pr, epoch)
            writers[name].add_scalar("val/f1", f1_one, epoch)
            writers[name].add_scalar("val/f1_mean", f1_mean, epoch)
            writers[name].add_scalar("val/accuracy", accuracy, epoch)

        print(f"  [{name}] validation results (epoch {epoch}):")
        print_results(cfg, auc_roc, auc_pr, f1_one, f1_mean, accuracy,
                      report, eval_per_class, eval_per_class_ego)

    return all_metrics


# ---------------------------------------------------------------------------
# Standalone testing — load checkpoints and evaluate
# ---------------------------------------------------------------------------
@torch.no_grad()
def test(cfg, model, testdata_loader, epoch):
    return _evaluate_model(cfg, model, testdata_loader, epoch, writers=None)


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
    if cfg.phase == "train":
        # Training loader — always raw video
        traindata_loader, _ = setup_dota(
            Dota, cfg, num_workers=cfg.num_workers,
            VCL=cfg.get("VCL", None), phase="train",
        )

        # Validation loader — precomputed embeddings if available, else raw video
        testdata_loader = None
        if cfg.get("enable_validation", False):
            testdata_loader = _try_build_precomputed_val_loader(cfg.data_path, cfg.batch_size, cfg.num_workers)
            if testdata_loader is not None:
                print(f"  Using precomputed val embeddings from {cfg.data_path}/embedding_val")
            else:
                _, testdata_loader = setup_dota(
                    Dota, cfg, num_workers=cfg.num_workers,
                    VCL=None, phase="test",
                )
                print("  Precomputed embeddings not found — using raw video for validation")

    elif cfg.phase == "test":
        traindata_loader = None
        testdata_loader = _try_build_precomputed_val_loader(cfg.data_path, cfg.batch_size, cfg.num_workers)
        if testdata_loader is not None:
            print(f"  Using precomputed val embeddings from {cfg.data_path}/embedding_val")
        else:
            _, testdata_loader = setup_dota(
                Dota, cfg, num_workers=cfg.num_workers,
                VCL=None, phase="test",
            )
            print("  Precomputed embeddings not found — using raw video for testing")

    # --- Model -------------------------------------------------------------
    epoch_val = 0
    model = build_multi_head_vjepa(cfg)

    # Resume / load checkpoints per head
    if cfg.epoch != -1:
        for name, head in model.heads.items():
            head_output = os.path.join(cfg.output, name)
            try:
                ckpt_cfg = EasyDict({
                    "output": head_output,
                    "epoch": cfg.epoch,
                    "device": cfg.device,
                })
                ckpt = movad_utils.load_checkpoint(ckpt_cfg)
                _load_head_state_dict(head, ckpt["model_state_dict"])
                ep = ckpt["epoch"] + 1
                print(f"  [{name}] resumed from epoch {ep}")
            except FileNotFoundError:
                print(f"  [{name}] no checkpoint at epoch {cfg.epoch} — starting fresh")
        epoch_val = cfg.epoch if cfg.epoch > 0 else 0

    if cfg.phase == "train":
        val_step = cfg.get("validation_epoch_step", 10)
        if testdata_loader is not None:
            val_step = min(val_step, cfg.epochs)
        train(cfg, model, traindata_loader, epoch_val,
              testdata_loader=testdata_loader,
              validation_epoch_step=val_step)

    elif cfg.phase == "test":
        if cfg.epoch == -1:
            cfg.epoch = 0
        # Load each head's checkpoint
        for name, head in model.heads.items():
            head_output = os.path.join(cfg.output, name)
            try:
                ckpt_cfg = EasyDict({
                    "output": head_output,
                    "epoch": cfg.epoch,
                    "device": cfg.device,
                })
                ckpt = movad_utils.load_checkpoint(ckpt_cfg)
                _load_head_state_dict(head, ckpt["model_state_dict"])
                ep = ckpt["epoch"] + 1
                print(f"  [{name}] loaded checkpoint epoch {ep}")
            except FileNotFoundError:
                print(f"  [{name}] WARNING: no checkpoint at epoch {cfg.epoch}")
        test(cfg, model, testdata_loader, cfg.epoch)
