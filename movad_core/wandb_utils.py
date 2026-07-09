"""wandb helpers — init, resume, log, finish for multi-head training.

On first call, ``wandb.login()`` reads the API key from the environment or
``~/.netrc`` (written by a prior ``wandb.login()`` in a Colab notebook cell).

Each temporal-model head gets its own wandb run so that metrics appear as
independently selectable lines in the dashboard.  ``reinit=True`` makes
multiple runs in the same process possible.
"""

from __future__ import annotations

import os
import torch


def _ensure_logged_in() -> None:
    """Silently re-establish a wandb session using the stored API key.

    If the user already called ``wandb.login()`` in the notebook (which writes
    ``~/.netrc``), this is a no-op.  Otherwise it attempts an offline/anonymous
    login so that the training script doesn't fail hard.
    """
    import wandb

    try:
        if not wandb.api.api_key:
            wandb.login(anonymous="allow")
    except Exception:
            wandb.login(anonymous="allow")


def init_wandb_for_head(
    name: str,
    head_cfg: dict,
    output_dir: str,
    begin_epoch: int,
    resume_id: str | None = None,
) -> "wandb.run":
    """Create or resume a wandb run for one temporal-model head.

    Parameters
    ----------
    name : str
        Head name (e.g. ``"vjepa_mamba"``).
    head_cfg : dict
        Full resolved config for this head — logged as wandb config for
        reproducibility.  Supports ``wandb_mode: "disabled"`` to skip wandb
        entirely (returns a no-op wrapper).
    output_dir : str
        Head output directory (contains ``checkpoints/``).
    begin_epoch : int
        Epoch the training loop will start at.  If ``> 0`` and *resume_id*
        is not provided, we look for a checkpoint at that epoch and try to
        extract a stored ``wandb_run_id`` so the run graph continues as one
        uninterrupted line.
    resume_id : str | None
        Explicit wandb run ID to resume.  Takes precedence over auto-detection
        from a checkpoint file.  Use when the caller already loaded the
        checkpoint (e.g. in a multi-head setup where the checkpoint lives in a
        per-head subdirectory).

    Returns
    -------
    wandb.run
        Initialised wandb run object.  Call ``.log(metrics, step=...)`` and
        ``.finish()`` on it.
    """
    import wandb

    _ensure_logged_in()

    wandb_mode = head_cfg.get("wandb_mode", "online")
    if wandb_mode == "disabled":
        return wandb.init(mode="disabled", project="movad")

    # Derive a project-group label from the parent output directory so that
    # runs from the same CLI invocation are visually clustered in the dashboard.
    project = head_cfg.get("wandb_project", "movad")
    group = os.path.basename(
        os.path.dirname(output_dir.rstrip("/\\"))
    )  # e.g. "2025-07-06_experiment"

    if resume_id is None and begin_epoch > 0:
        ckpt_path = os.path.join(
            output_dir, "checkpoints", f"model-{begin_epoch:02d}.pt"
        )
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            resume_id = ckpt.get("wandb_run_id")

    if resume_id:
        run = wandb.init(
            project=project,
            group=group,
            name=name,
            id=resume_id,
            resume="must",
            config=head_cfg,
            dir=output_dir,
            reinit=True,
        )
    else:
        run = wandb.init(
            project=project,
            group=group,
            name=name,
            config=head_cfg,
            dir=output_dir,
            reinit=True,
        )

    return run


def log_metrics(run, metrics: dict, step: int) -> None:
    """Thin wrapper so call sites don't need to know the wandb API shape."""
    run.log(metrics, step=step)


def finish_run(run) -> None:
    """Close a single wandb run."""
    run.finish()
