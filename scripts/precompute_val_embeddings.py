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
    parser.add_argument("--data_path", default=None,
                        help="Dataset root directory (overrides config)")
    parser.add_argument("--checkpoint_path", default=None,
                        help="Pretrained encoder checkpoint (overrides config)")
    parser.add_argument("--device", default=None,
                        help="Device override (default: cuda if available)")
    parser.add_argument("--verify", action="store_true",
                        help="Only encode 1 video, save it, and verify the output shape")
    parser.add_argument("--amp", default=None, choices=["fp32", "fp16", "bf16"],
                        help="AMP dtype for encoder pass (default: from config, or fp16)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for encoding (default: 4)")
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Load master config ------------------------------------------------
    with open(args.config[0], "r") as f:
        cfg = EasyDict(yaml.safe_load(f))

    # CLI overrides (explicitly passed values win over YAML)
    if args.data_path is not None:
        cfg.data_path = args.data_path
    if args.checkpoint_path is not None:
        cfg.checkpoint_path = args.checkpoint_path

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

    # encode_video_clips temporally averages internally — lossless since the
    # downstream Conv2d is linear and mean(conv(x_t)) = conv(mean(x_t)).

    # --- Validation dataset ------------------------------------------------
    cfg.batch_size = args.batch_size
    _, val_loader = setup_dota(
        Dota, cfg, num_workers=cfg.get("num_workers", 4),
        VCL=None, phase="test",
    )
    val_dataset = val_loader.dataset

    # --- AMP ---------------------------------------------------------------
    amp = args.amp or cfg.get("amp", "fp16")
    _AMP_DTYPE = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}
    amp_dtype = _AMP_DTYPE[amp]
    amp_ctx = (torch.amp.autocast("cuda", dtype=amp_dtype) if amp_dtype
               else __import__("contextlib").nullcontext())

    # --- Verify mode: encode 1 video, save, and validate shapes ------------
    if args.verify:
        print("--- Verify mode: encoding 1 video ---")
        video_key = val_dataset.keys[0]
        video_data, data_info = next(iter(val_loader))
        video_data = video_data.to(cfg.device, non_blocking=True)
        data_info = data_info.to(cfg.device, non_blocking=True)
        video_data = torch.swapaxes(video_data, 1, 2)
        v_len = video_data.shape[2]

        with torch.no_grad(), amp_ctx:
            patches = model.encode_video_clips(video_data, fb)  # [1, n_clips, S, D]  (temporally averaged)

        S_patches = patches.shape[2]  # spatial patches (T already averaged out by encode_video_clips)
        # patches shape: [1, n_clips, S_patches, D]

        # Expected dimensions (encoder-agnostic — derived from actual output)
        expected_D = model.encoder.embed_dim
        expected_clips = v_len - cfg.num_frames

        print(f"  Video key:               {video_key}")
        print(f"  Video frames (T):       {v_len}")
        print(f"  num_frames (fb):        {cfg.num_frames}")
        print(f"  tubelet_size:           {cfg.get('tubelet_size', 2)}")
        print(f"  Expected n_clips:       {expected_clips}")
        print(f"  S_patches (spatial):    {S_patches}")
        print(f"  Expected embed_dim:     {expected_D}")
        print(f"  patches_full.shape:     {list(patches.shape)}")
        print(f"  data_info.shape:        {list(data_info.shape)}")
        print(f"  save_dtype:             float32 (full encoder output)")

        # Validate
        assert patches.ndim == 4, f"patches should be 4D [B,clips,S,D], got {patches.ndim}D"
        assert patches.shape[0] == 1, f"batch dim should be 1, got {patches.shape[0]}"
        assert patches.shape[1] == expected_clips, \
            f"n_clips mismatch: got {patches.shape[1]}, expected {expected_clips}"
        assert patches.shape[2] == S_patches, \
            f"S_patches mismatch: got {patches.shape[2]}, expected {S_patches}"
        assert patches.shape[3] == expected_D, \
            f"embed_dim mismatch: got {patches.shape[3]}, expected {expected_D}"
        assert data_info.shape == (1, 11), \
            f"data_info shape mismatch: got {data_info.shape}, expected (1, 11)"
        assert patches.dtype == torch.float32, f"patches dtype should be float32, got {patches.dtype}"

        # Save to disk and reload to verify disk format
        save_dict = {
            "patches_full": patches[0].cpu(),
            "data_info": data_info[0].cpu(),
            "v_len": int(v_len),
        }
        verify_path = os.path.join(embed_dir, f"{video_key}.pt")
        torch.save(save_dict, verify_path)
        loaded = torch.load(verify_path, weights_only=True)

        print(f"\n  Saved & reloaded from: {verify_path}")
        print(f"  loaded[patches_full].shape: {list(loaded['patches_full'].shape)}")
        stored_S = patches.shape[2]  # spatial patches only
        print(f"  loaded[patches_full].dtype: {loaded['patches_full'].dtype}")
        print(f"  loaded[data_info].shape:    {list(loaded['data_info'].shape)}")
        print(f"  loaded[data_info].dtype:    {loaded['data_info'].dtype}")
        print(f"  loaded[v_len]:              {loaded['v_len']}")

        assert list(loaded["patches_full"].shape) == [expected_clips, stored_S, expected_D], \
            f"Loaded shape mismatch: {list(loaded['patches_full'].shape)}"
        assert loaded["patches_full"].dtype == torch.float32, \
            f"Saved patches should be float32, got {loaded['patches_full'].dtype}"
        assert list(loaded["data_info"].shape) == [11]
        assert loaded["v_len"] == v_len

        print("\n  All checks passed ✓")
        return

    # --- Encode & save -----------------------------------------------------

    print(f"Encoding {len(val_dataset)} videos → {embed_dir}  (AMP: {amp}, batch_size: {args.batch_size})")
    global_idx = 0

    for video_data, data_info in tqdm(val_loader, desc="Encoding val"):
        B = video_data.shape[0]
        video_data = video_data.to(cfg.device, non_blocking=True)
        data_info = data_info.to(cfg.device, non_blocking=True)
        video_data = torch.swapaxes(video_data, 1, 2)  # [B, C, T, H, W]

        with torch.no_grad(), amp_ctx:
            patches = model.encode_video_clips(video_data, fb)  # [B, n_clips, S, D]  (temporally averaged)

        # Per-video original frame counts (data_info[:, 0]) — pad_collate_videos
        # pads to the batch max, so video_data.shape[2] is identical for all items.
        v_len_orig = data_info[:, 0].long()  # [B]

        for b in range(B):
            v_len_b = int(v_len_orig[b].item())
            n_clips_valid = v_len_b - fb  # discard clips from padded frames
            video_key = val_dataset.keys[global_idx + b]
            save_dict = {
                "patches_full": patches[b, :n_clips_valid].cpu(),
                "data_info": data_info[b].cpu(),                          # [11]  fp32
                "v_len": v_len_b,
            }
            out_path = os.path.join(embed_dir, f"{video_key}.pt")
            torch.save(save_dict, out_path)

        del patches, video_data, data_info
        if B > 1:
            torch.cuda.empty_cache()
        global_idx += B

    print(f"Done — {global_idx} embeddings saved to {embed_dir}")


if __name__ == "__main__":
    main()
