"""
Precompute frozen V-JEPA encoder embeddings for the DoTA validation set.

Saves one ``.pt`` file per video under ``<data_path>/embedding_val/``.
Each file contains full patch tokens (float16) plus metadata — ready to be
consumed by the precomputed validation path in ``main.py``.

Usage
-----
    python scripts/precompute_val_embeddings.py --config cfgs/vjepa_mamba.yaml
    python scripts/precompute_val_embeddings.py --config cfgs/vjepa_mamba.yaml --output /another/path/
"""
from __future__ import annotations

import argparse
import os
import sys
import pathlib

import torch
import yaml
from easydict import EasyDict
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

# Compatibility shim (torchvision >= 0.20)
try:
    import torchvision.transforms._functional_tensor as _ft
    sys.modules.setdefault("torchvision.transforms.functional_tensor", _ft)
except ImportError:
    pass

from movad_core.dota import Dota, setup_dota
from model import build_multi_head_vjepa


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute V-JEPA val embeddings")
    parser.add_argument("--config", nargs="+", default=["cfgs/vjepa_mamba.yaml"],
                        help="YAML config(s). First = master (encoder + data settings).")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: <data_path>/embedding_val)")
    parser.add_argument("--device", default=None,
                        help="Device override (default: cuda if available)")
    parser.add_argument("--verify", action="store_true",
                        help="Only encode 1 video, save it, and verify the output shape")
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Load master config ------------------------------------------------
    with open(args.config[0], "r") as f:
        cfg = EasyDict(yaml.safe_load(f))

    # Replicate head-config parsing from main.py so build_multi_head_vjepa works
    cfg._head_names = [pathlib.Path(p).stem for p in args.config]
    head_cfgs_flat = []
    for name, path in zip(cfg._head_names, args.config):
        with open(path, "r") as f:
            hc = yaml.safe_load(f)
        hc["name"] = name
        head_cfgs_flat.append(hc)
    cfg._head_cfgs_flat = head_cfgs_flat

    # --- Setup -------------------------------------------------------------
    cfg.device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if "NF" not in cfg:
        cfg.NF = cfg.num_frames

    embed_dir = args.output or os.path.join(cfg.data_path, "embedding_val")
    os.makedirs(embed_dir, exist_ok=True)

    # --- Model -------------------------------------------------------------
    # Disable torch.compile for precomputation — it's a one-time pass and
    # compile would cache a separate graph for each unique video length,
    # gradually exhausting VRAM over 1400+ videos.
    print("Building model …")
    model = build_multi_head_vjepa(cfg)
    model.eval()
    fb = cfg.NF

    # --- Validation dataset ------------------------------------------------
    # Original evaluation ran on full videos (VCL=None).  To keep that
    # behaviour, we must use batch_size=1 — full videos have different
    # lengths and cannot be stacked.
    _, val_loader = setup_dota(
        Dota, cfg, num_workers=cfg.get("num_workers", 4),
        VCL=None, phase="test",
    )
    val_dataset = val_loader.dataset

    # --- Verify mode: encode 1 video, save, and validate shapes ------------
    if args.verify:
        print("--- Verify mode: encoding 1 video ---")
        video_key = val_dataset.keys[0]
        video_data, data_info = next(iter(val_loader))
        video_data = video_data.to(cfg.device, non_blocking=True)
        data_info = data_info.to(cfg.device, non_blocking=True)
        video_data = torch.swapaxes(video_data, 1, 2)
        v_len = video_data.shape[2]

        with torch.no_grad():
            patches = model.encode_video_clips(video_data, fb)

        # Expected dimensions
        enc = model.encoder.encoder
        expected_N = (cfg.num_frames // enc.tubelet_size) \
                     * (cfg.img_size // enc.patch_size) \
                     * (cfg.img_size // enc.patch_size)
        expected_D = model.encoder.embed_dim
        expected_clips = v_len - cfg.num_frames

        print(f"  Video key:               {video_key}")
        print(f"  Video frames (T):       {v_len}")
        print(f"  num_frames (fb):        {cfg.num_frames}")
        print(f"  Expected n_clips:       {expected_clips}")
        print(f"  Expected N_patches:     {expected_N}")
        print(f"  Expected embed_dim:     {expected_D}")
        print(f"  patches_full.shape:     {list(patches.shape)}")
        print(f"  data_info.shape:        {list(data_info.shape)}")

        # Validate
        assert patches.ndim == 4, f"patches should be 4D [B,clips,N,D], got {patches.ndim}D"
        assert patches.shape[0] == 1, f"batch dim should be 1, got {patches.shape[0]}"
        assert patches.shape[1] == expected_clips, \
            f"n_clips mismatch: got {patches.shape[1]}, expected {expected_clips}"
        assert patches.shape[2] == expected_N, \
            f"N_patches mismatch: got {patches.shape[2]}, expected {expected_N}"
        assert patches.shape[3] == expected_D, \
            f"embed_dim mismatch: got {patches.shape[3]}, expected {expected_D}"
        assert data_info.shape == (1, 11), \
            f"data_info shape mismatch: got {data_info.shape}, expected (1, 11)"
        assert patches.dtype == torch.float32, f"patches dtype should be float32, got {patches.dtype}"

        # Save to disk and reload to verify disk format
        save_dict = {
            "patches_full": patches[0].cpu().half(),
            "data_info": data_info[0].cpu(),
            "v_len": int(v_len),
        }
        verify_path = os.path.join(embed_dir, f"{video_key}.pt")
        torch.save(save_dict, verify_path)
        loaded = torch.load(verify_path, weights_only=True)

        print(f"\n  Saved & reloaded from: {verify_path}")
        print(f"  loaded[patches_full].shape: {list(loaded['patches_full'].shape)}")
        print(f"  loaded[patches_full].dtype: {loaded['patches_full'].dtype}")
        print(f"  loaded[data_info].shape:    {list(loaded['data_info'].shape)}")
        print(f"  loaded[data_info].dtype:    {loaded['data_info'].dtype}")
        print(f"  loaded[v_len]:              {loaded['v_len']}")

        assert list(loaded["patches_full"].shape) == [expected_clips, expected_N, expected_D], \
            f"Loaded shape mismatch: {list(loaded['patches_full'].shape)}"
        assert loaded["patches_full"].dtype == torch.float16, "Saved patches should be float16"
        assert list(loaded["data_info"].shape) == [11]
        assert loaded["v_len"] == v_len

        print("\n  All checks passed ✓")
        return

    # --- Encode & save -----------------------------------------------------
    # Accumulate in memory and flush to disk in chunks — per-video torch.save
    # on a network/WSL mount can bottleneck the whole pipeline.
    print(f"Encoding {len(val_dataset)} videos → {embed_dir}")
    global_idx = 0
    pending: list[tuple[str, dict]] = []   # [(video_key, save_dict), …]
    _flush_every = 200   # flush every N videos to keep RAM ~2 GB

    def _flush():
        for video_key, save_dict in pending:
            out_path = os.path.join(embed_dir, f"{video_key}.pt")
            torch.save(save_dict, out_path)

    for video_data, data_info in tqdm(val_loader, desc="Encoding val"):
        video_key = val_dataset.keys[global_idx]
        video_data = video_data.to(cfg.device, non_blocking=True)
        data_info = data_info.to(cfg.device, non_blocking=True)
        video_data = torch.swapaxes(video_data, 1, 2)  # [1, C, T, H, W]

        with torch.no_grad():
            patches = model.encode_video_clips(video_data, fb)  # [1, n_clips, N, D]

        # Move to CPU immediately and free GPU tensors — full-video
        # mega-batches are large (100+ clips) and variable-sized, which
        # fragments the CUDA allocator cache across 1400+ iterations.
        v_len = video_data.shape[2]  # capture before del
        patches_cpu = patches[0].cpu().half()          # [n_clips, N, D]  fp16
        info_cpu = data_info[0].cpu()                  # [11]  fp32
        del patches, video_data, data_info
        torch.cuda.empty_cache()

        pending.append((video_key, {
            "patches_full": patches_cpu,
            "data_info": info_cpu,
            "v_len": int(v_len),
        }))
        global_idx += 1

        if len(pending) >= _flush_every:
            _flush()
            pending.clear()

    if pending:
        _flush()
        pending.clear()

    print(f"Done — {global_idx} embeddings saved to {embed_dir}")


if __name__ == "__main__":
    main()
