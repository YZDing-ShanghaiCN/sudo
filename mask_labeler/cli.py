"""Console entry point: `mask-labeler single | video | video-from-mask | video-multi`."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

DEFAULT_SAM2_CKPT = os.environ.get(
    "MASK_LABELER_SAM2_CKPT",
    "/home/user/Desktop/checkpoints/sam/sam2.1_hiera_large.pt",
)
DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_FRAME_PATTERN = "*.jpg"


def _parse_size(s: str) -> tuple[int, int]:
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"expected WxH, got {s!r}") from e


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--sam2-checkpoint", default=DEFAULT_SAM2_CKPT,
                   help="local SAM2 .pt (default: $MASK_LABELER_SAM2_CKPT or RoboCamCalib's local copy)")
    p.add_argument("--sam2-config", default=DEFAULT_SAM2_CONFIG,
                   help="SAM2 config (Hydra name resolved inside the sam2 package)")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])


def _read_rgb(path: Path) -> np.ndarray:
    rgb = imageio.imread(path)
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    elif rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    return rgb


def _read_mask(path: Path) -> np.ndarray:
    if not path.is_file():
        raise ValueError(f"mask not found: {path}")
    arr = np.asarray(imageio.imread(path))
    if arr.ndim == 2:
        return arr > 0
    if arr.ndim == 3:
        # Ignore alpha: opaque black backgrounds are common in RGBA mask exports.
        channels = arr[..., :3]
        if channels.shape[-1] == 1:
            return channels[..., 0] > 0
        return channels.any(axis=-1)
    raise ValueError(f"expected 2D or RGB/RGBA mask image, got shape {arr.shape}")


def cmd_single(args: argparse.Namespace) -> int:
    image_path = Path(args.image)
    out_path = Path(args.out)
    if not image_path.is_file():
        print(f"error: image not found: {image_path}", file=sys.stderr)
        return 2
    rgb = imageio.imread(image_path)
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    elif rgb.shape[-1] == 4:
        rgb = rgb[..., :3]

    from .prompt_drawer import PromptDrawer

    drawer = PromptDrawer(
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        device=args.device,
        max_window_size=_parse_size(args.max_window),
    )
    if args.preload_mask:
        init = imageio.imread(args.preload_mask)
        drawer.set_init_mask(init)

    mask = drawer.run(rgb)
    drawer.close()

    if mask is None:
        print("[mask_labeler] aborted; nothing written")
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out_path, (mask.astype(np.uint8)) * 255)
    print(f"[mask_labeler] wrote {out_path}  (foreground px: {int(mask.sum())})")
    return 0


def _looks_like_path(arg: str) -> bool:
    """Heuristic: is this CLI arg a path/filename rather than a bare integer index?"""
    if arg.isascii() and arg.isdigit():
        return False  # bare integer index (strict: timestamp stems have underscores)
    return (os.sep in arg) or bool(Path(arg).suffix) or Path(arg).expanduser().exists()


def cmd_video(args: argparse.Namespace) -> int:
    from .prompt_drawer import PromptDrawer
    from .video_propagate import (
        _canonical_order,
        _frame_key,
        propagate_video,
        resolve_seed_position,
        resolve_to_exclusive,
    )

    out_dir = Path(args.out_dir)

    # Resolve the frames directory: explicit flag, else the seed-frame path's parent.
    if args.frames_dir is not None:
        frames_dir = Path(args.frames_dir)
    elif _looks_like_path(args.seed_frame):
        frames_dir = Path(args.seed_frame).expanduser().resolve().parent
    else:
        print("error: --frames-dir is required unless --seed-frame is a path", file=sys.stderr)
        return 2

    # If the pattern was left default and the seed is a path, match its extension.
    frame_pattern = args.frame_pattern
    if frame_pattern == DEFAULT_FRAME_PATTERN and _looks_like_path(args.seed_frame):
        seed_suffix = Path(args.seed_frame).suffix
        if seed_suffix:
            frame_pattern = f"*{seed_suffix}"

    try:
        order = _canonical_order(frames_dir, frame_pattern)
        seed_pos = resolve_seed_position(args.seed_frame, order, frames_dir)
        to_pos = resolve_to_exclusive(args.to_frame, order, frames_dir, seed_pos)
    except SystemExit as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if seed_pos == to_pos:
        print("error: seed-frame == to-frame; nothing to propagate", file=sys.stderr)
        return 2

    seed_img_path = order[seed_pos]
    rgb = imageio.imread(seed_img_path)
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    elif rgb.shape[-1] == 4:
        rgb = rgb[..., :3]

    drawer = PromptDrawer(
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        device=args.device,
        max_window_size=_parse_size(args.max_window),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_mask_path = out_dir / f"{_frame_key(seed_img_path)}.mask.png"
    if args.preload_mask:
        drawer.set_init_mask(imageio.imread(args.preload_mask))
    elif seed_mask_path.is_file():
        print(f"[mask_labeler] preloading existing seed mask {seed_mask_path}")
        drawer.set_init_mask(imageio.imread(seed_mask_path))

    seed_mask = drawer.run(rgb)
    drawer.close()
    if seed_mask is None or not seed_mask.any():
        print("[mask_labeler] aborted or empty mask; nothing propagated")
        return 1

    imageio.imwrite(seed_mask_path, (seed_mask.astype(np.uint8)) * 255)
    print(f"[mask_labeler] seed mask -> {seed_mask_path}")

    propagate_video(
        frames_dir=frames_dir,
        seed_pos=seed_pos,
        to_pos=to_pos,
        seed_mask=seed_mask,
        out_dir=out_dir,
        frame_pattern=frame_pattern,
        order=order,
        model_id=args.sam2_model or None,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        device=args.device,
        offload_video_to_cpu=not args.no_offload_video,
        offload_state_to_cpu=not args.no_offload_state,
    )
    print(f"[mask_labeler] propagation done -> {out_dir}")
    return 0


def cmd_video_from_mask(args: argparse.Namespace) -> int:
    from .video_propagate import (
        _canonical_order,
        _frame_key,
        propagate_video,
        resolve_seed_position,
        resolve_to_exclusive,
    )

    out_dir = Path(args.out_dir)

    if args.frames_dir is not None:
        frames_dir = Path(args.frames_dir)
    elif _looks_like_path(args.seed_frame):
        frames_dir = Path(args.seed_frame).expanduser().resolve().parent
    else:
        print("error: --frames-dir is required unless --seed-frame is a path", file=sys.stderr)
        return 2

    frame_pattern = args.frame_pattern
    if frame_pattern == DEFAULT_FRAME_PATTERN and _looks_like_path(args.seed_frame):
        seed_suffix = Path(args.seed_frame).suffix
        if seed_suffix:
            frame_pattern = f"*{seed_suffix}"

    try:
        order = _canonical_order(frames_dir, frame_pattern)
        seed_pos = resolve_seed_position(args.seed_frame, order, frames_dir)
        to_pos = resolve_to_exclusive(args.to_frame, order, frames_dir, seed_pos)
    except SystemExit as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if seed_pos == to_pos:
        print("error: seed-frame == to-frame; nothing to propagate", file=sys.stderr)
        return 2

    seed_img_path = order[seed_pos]
    rgb = _read_rgb(seed_img_path)
    try:
        seed_mask = _read_mask(Path(args.seed_mask))
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if seed_mask.shape != rgb.shape[:2]:
        print(
            f"error: seed mask shape {seed_mask.shape} does not match seed frame "
            f"shape {rgb.shape[:2]}: {args.seed_mask}",
            file=sys.stderr,
        )
        return 2
    if not seed_mask.any():
        print("[mask_labeler] empty seed mask; nothing propagated")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    seed_mask_path = out_dir / f"{_frame_key(seed_img_path)}.mask.png"
    imageio.imwrite(seed_mask_path, (seed_mask.astype(np.uint8)) * 255)
    print(f"[mask_labeler] seed mask -> {seed_mask_path}")

    propagate_video(
        frames_dir=frames_dir,
        seed_pos=seed_pos,
        to_pos=to_pos,
        seed_mask=seed_mask,
        out_dir=out_dir,
        frame_pattern=frame_pattern,
        order=order,
        model_id=args.sam2_model or None,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        device=args.device,
        offload_video_to_cpu=not args.no_offload_video,
        offload_state_to_cpu=not args.no_offload_state,
    )
    print(f"[mask_labeler] propagation from existing mask done -> {out_dir}")
    return 0


def cmd_video_multi(args: argparse.Namespace) -> int:
    from .prompt_drawer import PromptDrawer
    from .video_propagate import (
        _canonical_order,
        _frame_key,
        propagate_video_multi,
        resolve_seed_position,
        resolve_to_exclusive,
    )

    if args.manual_count < 1:
        print("error: --manual-count must be >= 1", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)

    if args.frames_dir is not None:
        frames_dir = Path(args.frames_dir)
    elif _looks_like_path(args.seed_frame):
        frames_dir = Path(args.seed_frame).expanduser().resolve().parent
    else:
        print("error: --frames-dir is required unless --seed-frame is a path", file=sys.stderr)
        return 2

    frame_pattern = args.frame_pattern
    if frame_pattern == DEFAULT_FRAME_PATTERN and _looks_like_path(args.seed_frame):
        seed_suffix = Path(args.seed_frame).suffix
        if seed_suffix:
            frame_pattern = f"*{seed_suffix}"

    try:
        order = _canonical_order(frames_dir, frame_pattern)
        seed_pos = resolve_seed_position(args.seed_frame, order, frames_dir)
        to_pos = resolve_to_exclusive(args.to_frame, order, frames_dir, seed_pos)
    except SystemExit as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if seed_pos == to_pos:
        print("error: seed-frame == to-frame; nothing to propagate", file=sys.stderr)
        return 2

    step = 1 if to_pos > seed_pos else -1
    positions = list(range(seed_pos, to_pos, step))
    if len(positions) < args.manual_count:
        print(
            f"error: --manual-count {args.manual_count} needs at least "
            f"{args.manual_count} frames in range, but only {len(positions)} found",
            file=sys.stderr,
        )
        return 2
    manual_positions = positions[:args.manual_count]

    drawer = PromptDrawer(
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        device=args.device,
        max_window_size=_parse_size(args.max_window),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_masks: dict[int, np.ndarray] = {}
    try:
        for label_idx, pos in enumerate(manual_positions, start=1):
            img_path = order[pos]
            mask_path = out_dir / f"{_frame_key(img_path)}.mask.png"
            drawer.reset()
            if label_idx == 1 and args.preload_mask:
                drawer.set_init_mask(imageio.imread(args.preload_mask))
            elif mask_path.is_file():
                print(f"[mask_labeler] preloading existing manual mask {mask_path}")
                drawer.set_init_mask(imageio.imread(mask_path))

            print(
                f"[mask_labeler] labeling manual seed {label_idx}/{args.manual_count}: "
                f"{img_path}"
            )
            mask = drawer.run(_read_rgb(img_path))
            if mask is None or not mask.any():
                print("[mask_labeler] aborted or empty mask; nothing propagated")
                return 1

            imageio.imwrite(mask_path, (mask.astype(np.uint8)) * 255)
            print(f"[mask_labeler] manual seed mask -> {mask_path}")
            seed_masks[pos] = mask
    finally:
        drawer.close()

    propagate_video_multi(
        frames_dir=frames_dir,
        seed_pos=seed_pos,
        to_pos=to_pos,
        seed_masks=seed_masks,
        out_dir=out_dir,
        frame_pattern=frame_pattern,
        order=order,
        model_id=args.sam2_model or None,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        device=args.device,
        offload_video_to_cpu=not args.no_offload_video,
        offload_state_to_cpu=not args.no_offload_state,
    )
    print(f"[mask_labeler] multi-seed propagation done -> {out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mask-labeler", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_single = sub.add_parser("single", help="label one image, write one mask")
    p_single.add_argument("--image", required=True, type=Path)
    p_single.add_argument("--out", required=True, type=Path, help="output mask PNG (uint8 0/255)")
    p_single.add_argument("--preload-mask", type=Path, default=None,
                          help="seed mask to start iterative refinement from")
    p_single.add_argument("--max-window", default="1600x900", help="WxH cap for the labeler window")
    _add_common(p_single)
    p_single.set_defaults(func=cmd_single)

    p_video = sub.add_parser("video", help="label seed frame and propagate via SAM2 video predictor")
    p_video.add_argument("--frames-dir", required=False, type=Path, default=None,
                         help="directory of frames; if omitted, derived from --seed-frame's parent "
                              "(seed-frame must then be a path)")
    p_video.add_argument("--seed-frame", required=True,
                         help="seed frame as an integer index OR a path/filename/stem "
                              "(e.g. 0 or /.../20260604_200559_111996.png)")
    p_video.add_argument("--to-frame", required=True,
                         help="end frame: a path/filename/stem is INCLUSIVE; a bare integer index is "
                              "EXCLUSIVE (legacy); 'end'/'last' includes the final frame, "
                              "'start'/'first' the first. Propagates backward if it precedes the seed.")
    p_video.add_argument("--out-dir", required=True, type=Path)
    p_video.add_argument("--frame-pattern", default=DEFAULT_FRAME_PATTERN,
                         help="glob for frames; if left default and --seed-frame is a path, the "
                              "seed file's extension is used (e.g. '*.png')")
    p_video.add_argument("--preload-mask", type=Path, default=None)
    p_video.add_argument("--max-window", default="1600x900")
    p_video.add_argument("--sam2-model", default="",
                         help="Hugging Face SAM2 model id (e.g. facebook/sam2.1-hiera-large); "
                              "if set, used for the VIDEO predictor instead of --sam2-checkpoint")
    p_video.add_argument("--no-offload-video", action="store_true",
                         help="keep all decoded frames on the GPU (faster, but OOMs on long clips; "
                              "by default frames are offloaded to CPU RAM)")
    p_video.add_argument("--no-offload-state", action="store_true",
                         help="keep per-frame inference state on the GPU (faster, more VRAM; "
                              "by default state is offloaded to CPU RAM)")
    _add_common(p_video)
    p_video.set_defaults(func=cmd_video)

    p_video_from_mask = sub.add_parser(
        "video-from-mask",
        help="propagate from an existing seed mask without opening the labeler",
    )
    p_video_from_mask.add_argument("--frames-dir", required=False, type=Path, default=None,
                                   help="directory of frames; if omitted, derived from "
                                        "--seed-frame's parent (seed-frame must then be a path)")
    p_video_from_mask.add_argument("--seed-frame", required=True,
                                   help="seed frame matching --seed-mask, as an integer index OR a "
                                        "path/filename/stem")
    p_video_from_mask.add_argument("--to-frame", required=True,
                                   help="end frame: a path/filename/stem is INCLUSIVE; a bare "
                                        "integer index is EXCLUSIVE (legacy); 'end'/'last' includes "
                                        "the final frame, 'start'/'first' the first. Propagates "
                                        "backward if it precedes the seed.")
    p_video_from_mask.add_argument("--seed-mask", required=True, type=Path,
                                   help="existing mask for --seed-frame; must match seed frame size")
    p_video_from_mask.add_argument("--out-dir", required=True, type=Path)
    p_video_from_mask.add_argument("--frame-pattern", default=DEFAULT_FRAME_PATTERN,
                                   help="glob for frames; if left default and --seed-frame is a "
                                        "path, the seed file's extension is used (e.g. '*.png')")
    p_video_from_mask.add_argument("--sam2-model", default="",
                                   help="Hugging Face SAM2 model id (e.g. "
                                        "facebook/sam2.1-hiera-large); if set, used for the VIDEO "
                                        "predictor instead of --sam2-checkpoint")
    p_video_from_mask.add_argument("--no-offload-video", action="store_true",
                                   help="keep all decoded frames on the GPU (faster, but OOMs on "
                                        "long clips; by default frames are offloaded to CPU RAM)")
    p_video_from_mask.add_argument("--no-offload-state", action="store_true",
                                   help="keep per-frame inference state on the GPU (faster, more "
                                        "VRAM; by default state is offloaded to CPU RAM)")
    _add_common(p_video_from_mask)
    p_video_from_mask.set_defaults(func=cmd_video_from_mask)

    p_video_multi = sub.add_parser(
        "video-multi",
        help="label several consecutive seed frames and propagate via SAM2 video predictor",
    )
    p_video_multi.add_argument("--frames-dir", required=False, type=Path, default=None,
                               help="directory of frames; if omitted, derived from --seed-frame's "
                                    "parent (seed-frame must then be a path)")
    p_video_multi.add_argument("--seed-frame", required=True,
                               help="first manual seed frame as an integer index OR a "
                                    "path/filename/stem")
    p_video_multi.add_argument("--to-frame", required=True,
                               help="end frame: a path/filename/stem is INCLUSIVE; a bare integer "
                                    "index is EXCLUSIVE (legacy); 'end'/'last' includes the final "
                                    "frame, 'start'/'first' the first. Propagates backward if it "
                                    "precedes the seed.")
    p_video_multi.add_argument("--out-dir", required=True, type=Path)
    p_video_multi.add_argument("--frame-pattern", default=DEFAULT_FRAME_PATTERN,
                               help="glob for frames; if left default and --seed-frame is a path, "
                                    "the seed file's extension is used (e.g. '*.png')")
    p_video_multi.add_argument("--manual-count", type=int, default=5,
                               help="number of consecutive frames to label manually before "
                                    "propagation (default: 5)")
    p_video_multi.add_argument("--preload-mask", type=Path, default=None,
                               help="optional mask to preload for the first manual seed frame; "
                                    "later frames auto-preload existing out-dir masks if present")
    p_video_multi.add_argument("--max-window", default="1600x900")
    p_video_multi.add_argument("--sam2-model", default="",
                               help="Hugging Face SAM2 model id (e.g. facebook/sam2.1-hiera-large); "
                                    "if set, used for the VIDEO predictor instead of --sam2-checkpoint")
    p_video_multi.add_argument("--no-offload-video", action="store_true",
                               help="keep all decoded frames on the GPU (faster, but OOMs on long "
                                    "clips; by default frames are offloaded to CPU RAM)")
    p_video_multi.add_argument("--no-offload-state", action="store_true",
                               help="keep per-frame inference state on the GPU (faster, more VRAM; "
                                    "by default state is offloaded to CPU RAM)")
    _add_common(p_video_multi)
    p_video_multi.set_defaults(func=cmd_video_multi)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
