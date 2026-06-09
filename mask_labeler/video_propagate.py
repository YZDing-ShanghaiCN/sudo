"""SAM2 video propagation.

Extracted from RoboCamCalib (robocamcalib/data_pipeline/label_and_track_mask.py),
with os.system shelling replaced by shutil and a Hugging Face-or-local-checkpoint switch.
"""
from __future__ import annotations

import bisect
import os
import shutil
import tempfile
from pathlib import Path
from typing import Mapping, Optional

import imageio.v2 as imageio
import numpy as np
import torch
import tqdm

from .vis import overlay_mask


def _int_or_none(s: str) -> Optional[int]:
    """Return the integer value of a plain non-negative integer frame index/stem,
    else None. The single source of truth for "is this an integer index".

    Strict ASCII-digits only: notably ``int()`` would happily parse
    ``"20260604_200559_111996"`` (underscores are digit separators) and unicode
    digits, which would wrongly classify timestamp stems as integers."""
    if isinstance(s, str) and s.isascii() and s.isdigit():
        return int(s)
    return None


def _frame_key(path: Path) -> str:
    """Output/seed-mask filename stem for a frame.

    Integer-named frames keep the historical zero-padded form (``5`` -> ``000005``)
    so existing integer dirs are byte-identical; non-integer stems (e.g. timestamp
    names ``20260604_200559_111996``) are used verbatim."""
    stem = path.stem
    v = _int_or_none(stem)
    return f"{v:06d}" if v is not None else stem


def _canonical_order(frames_dir: Path, frame_pattern: str) -> list[Path]:
    """Sorted frame list defining the canonical propagation order.

    When *every* matched file has an integer stem the list is sorted numerically
    (so ragged-width integer dirs like ``1,2,10`` stay correct); otherwise it is
    sorted lexicographically, which is chronological for fixed-width timestamp
    names like ``YYYYMMDD_HHMMSS_microseconds``."""
    frames = list(Path(frames_dir).glob(frame_pattern))
    if not frames:
        raise SystemExit(f"no frames matching {frame_pattern} under {frames_dir}")
    all_int = all(_int_or_none(p.stem) is not None for p in frames)
    if all_int:
        return sorted(frames, key=lambda p: (_int_or_none(p.stem), p.name))
    return sorted(frames, key=lambda p: p.name)


def _match_path_in_order(arg: str, order: list[Path], frames_dir: Path) -> Optional[int]:
    """Resolve a path-like ``arg`` to a position in ``order``, or None if it is not
    a path form. Raises a targeted error when the path exists but is not in ``order``
    (typically a frame-pattern mismatch)."""
    p = Path(arg).expanduser()
    is_abs = p.is_absolute()
    is_rel_path = (os.sep in arg) or (frames_dir / arg).exists()
    if not (is_abs or is_rel_path):
        return None
    target = (p if is_abs else (frames_dir / arg)).resolve()
    for i, q in enumerate(order):
        if q.name == target.name and q.resolve() == target:
            return i
    if target.exists():
        raise SystemExit(
            f"{arg!r} exists but is not in {frames_dir} matching the frame pattern; "
            f"pass --frame-pattern to match it (e.g. '*{p.suffix}')"
        )
    raise SystemExit(f"frame path not found: {arg}")


def resolve_seed_position(arg: str, order: list[Path], frames_dir: Path) -> int:
    """Resolve the seed-frame argument to an exact position in ``order`` (must exist).

    Accepts an absolute/relative path, a bare basename, a stem (extension dropped),
    or a bare integer frame index."""
    pos = _match_path_in_order(arg, order, frames_dir)
    if pos is not None:
        return pos
    by_name = {q.name: i for i, q in enumerate(order)}
    if arg in by_name:
        return by_name[arg]
    by_stem = {q.stem: i for i, q in enumerate(order)}
    if arg in by_stem:
        return by_stem[arg]
    v = _int_or_none(arg)
    if v is not None:
        for i, q in enumerate(order):
            if _int_or_none(q.stem) == v:
                return i
        raise SystemExit(f"seed frame {v} not present under {frames_dir}")
    raise SystemExit(
        f"could not resolve seed-frame {arg!r}: not a path, basename, stem, or integer index"
    )


def resolve_to_exclusive(arg: str, order: list[Path], frames_dir: Path, seed_pos: int) -> int:
    """Resolve the to-frame argument to an *exclusive* position bound.

    Forms and semantics:
    - ``end``/``last`` -> ``len(order)`` (include the final frame, forward run);
      ``start``/``first`` -> ``-1`` (include index 0, backward run).
    - path/basename/stem -> that frame is **inclusive**; converted to an exclusive
      bound one step past it in the seed->target direction.
    - bare integer -> historical **exclusive** value bound via bisect over the
      integer stem values (reproduces the old behaviour even when absent on disk)."""
    s = arg.lower()
    if s in ("end", "last"):
        return len(order)
    if s in ("start", "first"):
        return -1

    pos = _match_path_in_order(arg, order, frames_dir)
    if pos is None:
        by_name = {q.name: i for i, q in enumerate(order)}
        by_stem = {q.stem: i for i, q in enumerate(order)}
        pos = by_name.get(arg, by_stem.get(arg))
    if pos is not None:
        # Named to-frame is inclusive -> step one past it toward the seed direction.
        step = 1 if pos >= seed_pos else -1
        return pos + step

    v = _int_or_none(arg)
    if v is not None:
        int_values = sorted(
            iv for iv in (_int_or_none(q.stem) for q in order) if iv is not None
        )
        if not int_values:
            raise SystemExit(
                f"integer to-frame {arg} given but frames have non-integer names; "
                f"pass a path/filename or 'end'/'last'"
            )
        return bisect.bisect_left(int_values, v)

    raise SystemExit(
        f"could not resolve to-frame {arg!r}: not a path, basename, stem, integer, "
        f"or sentinel (end/last/start/first)"
    )


def _build_video_predictor(
    model_id: Optional[str],
    checkpoint: Optional[str],
    config: Optional[str],
    device: str,
):
    if model_id and (not checkpoint):
        from sam2.sam2_video_predictor import SAM2VideoPredictor
        return SAM2VideoPredictor.from_pretrained(model_id, device=device)
    if not (checkpoint and config):
        raise ValueError(
            "video_propagate: either --sam2-model OR both --sam2-checkpoint and --sam2-config required"
        )
    from sam2.build_sam import build_sam2_video_predictor
    return build_sam2_video_predictor(config, checkpoint, device=device)


def propagate_video(
    frames_dir: Path,
    seed_pos: int,
    to_pos: int,
    seed_mask: np.ndarray,
    out_dir: Path,
    frame_pattern: str = "*.jpg",
    *,
    order: Optional[list[Path]] = None,
    model_id: Optional[str] = None,
    sam2_checkpoint: Optional[str] = None,
    sam2_config: Optional[str] = None,
    device: str = "cuda",
    offload_video_to_cpu: bool = True,
    offload_state_to_cpu: bool = True,
) -> None:
    """Propagate `seed_mask` (uint8/bool) from `order[seed_pos]` toward `to_pos` (exclusive).

    `seed_pos`/`to_pos` are positions in the canonical sorted frame list (`order`,
    rebuilt via `_canonical_order` if not supplied). The seed position is inclusive,
    `to_pos` exclusive (and may be `len(order)` or `-1` for the end/start sentinels);
    direction is inferred from their order.
    Writes <out_dir>/<frame_key>.mask.png (uint8 0/255) and <out_dir>/vis/<frame_key>.jpg
    overlays, where frame_key is the source stem (zero-padded for integer names).
    """
    propagate_video_multi(
        frames_dir=frames_dir,
        seed_pos=seed_pos,
        to_pos=to_pos,
        seed_masks={seed_pos: seed_mask},
        out_dir=out_dir,
        frame_pattern=frame_pattern,
        order=order,
        model_id=model_id,
        sam2_checkpoint=sam2_checkpoint,
        sam2_config=sam2_config,
        device=device,
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
    )


def propagate_video_multi(
    frames_dir: Path,
    seed_pos: int,
    to_pos: int,
    seed_masks: Mapping[int, np.ndarray],
    out_dir: Path,
    frame_pattern: str = "*.jpg",
    *,
    order: Optional[list[Path]] = None,
    model_id: Optional[str] = None,
    sam2_checkpoint: Optional[str] = None,
    sam2_config: Optional[str] = None,
    device: str = "cuda",
    offload_video_to_cpu: bool = True,
    offload_state_to_cpu: bool = True,
) -> None:
    """Propagate masks from one or more manually-labeled frames.

    ``seed_masks`` is keyed by positions in the canonical sorted frame list
    (``order``). The selected propagation range still starts at ``seed_pos`` and
    runs toward ``to_pos`` exactly like :func:`propagate_video`.
    """
    frames_dir = Path(frames_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vis").mkdir(parents=True, exist_ok=True)

    if order is None:
        order = _canonical_order(frames_dir, frame_pattern)

    step = 1 if to_pos > seed_pos else -1
    selected_positions = list(range(seed_pos, to_pos, step))
    if not selected_positions:
        raise SystemExit("seed_pos == to_pos; nothing to propagate")
    selected = [order[p] for p in selected_positions]
    pos_to_local = {pos: i for i, pos in enumerate(selected_positions)}

    seed_by_local: dict[int, np.ndarray] = {}
    for pos, mask in seed_masks.items():
        if pos not in pos_to_local:
            raise SystemExit(f"seed mask position {pos} is outside the propagation range")
        seed_by_local[pos_to_local[pos]] = mask
    if not seed_by_local:
        raise SystemExit("no seed masks supplied")

    print(f"[mask_labeler] {len(selected)} frames in range")
    if len(seed_by_local) > 1:
        print(f"[mask_labeler] {len(seed_by_local)} manual seed masks")

    predictor = _build_video_predictor(model_id, sam2_checkpoint, sam2_config, device)

    with tempfile.TemporaryDirectory() as tdir:
        tdir = Path(tdir)
        # SAM2VideoPredictor expects sequentially-numbered JPEGs starting at 0.
        local_to_src: list[Path] = []
        for local_idx, src in enumerate(selected):
            local_to_src.append(src)
            dst = tdir / f"{local_idx:06d}.jpg"
            if src.suffix.lower() == ".jpg":
                shutil.copyfile(src, dst)
            else:
                img = imageio.imread(src)
                imageio.imwrite(dst, img)

        # init_state otherwise loads every frame onto the GPU as one tensor,
        # which OOMs on long clips / small cards. Offloading frames and the
        # per-frame inference state to CPU trades a small fps hit for bounded
        # GPU memory (roughly constant in clip length).
        state = predictor.init_state(
            str(tdir),
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
        )
        pred_masks = {
            local_idx: np.asarray(mask) > 0
            for local_idx, mask in seed_by_local.items()
        }
        for local_idx, mask in seed_by_local.items():
            predictor.add_new_mask(state, local_idx, 1, mask > 0)

        autocast_kwargs = dict(dtype=torch.bfloat16) if device == "cuda" else dict(enabled=False)
        with torch.inference_mode(), torch.autocast(
            "cuda" if device == "cuda" else "cpu", **autocast_kwargs
        ):
            for frame_idx, _obj_ids, masks in predictor.propagate_in_video(state):
                pred_masks[frame_idx] = (masks[0] > 0.0).cpu().numpy()[0]
        for local_idx, mask in seed_by_local.items():
            pred_masks[local_idx] = np.asarray(mask) > 0

    for local_idx, src in tqdm.tqdm(
        list(enumerate(local_to_src)), desc="saving"
    ):
        m = pred_masks[local_idx]
        key = _frame_key(src)
        save = (m.astype(np.uint8)) * 255
        imageio.imwrite(out_dir / f"{key}.mask.png", save)
        rgb = imageio.imread(src)
        imageio.imwrite(out_dir / "vis" / f"{key}.jpg", overlay_mask(rgb, m))
