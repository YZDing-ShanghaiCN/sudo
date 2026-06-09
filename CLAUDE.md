# CLAUDE.md — mask_labeler

> **Maintenance:** keep this file current when the CLI flags, file layout, or
> SAM2 backend assumptions change.

## Purpose

Interactive single-image mask labeler + SAM2 video propagation. Extracted
from `RoboCamCalib/robocamcalib/data_pipeline/{prompt_drawer.py, label_and_track_mask.py}`
so any SudoDeploy task (currently: FoundationPose battery_base test) can produce
high-quality masks without depending on RoboCamCalib's package.

## Stack

- Standalone **uv project** (own `pyproject.toml` + `uv.lock` + `.venv/`).
- Python 3.10. Deps: `torch>=2.7` + `torchvision` (CUDA 12.8 wheel via
  explicit uv index — cu128 is required for Blackwell sm_120; earlier wheels
  ship kernels only through sm_90), `sam-2` (git source, no PyPI release),
  `opencv-python`, `imageio`, `numpy`, `tqdm`, `hydra-core`.
- Not part of SudoDeploy's main editable install. Invoke via `uv run mask-labeler ...`.

## Layout

```
tools/mask_labeler/
├── pyproject.toml          # uv project + console script
├── uv.lock                 # committed
├── README.md
├── CLAUDE.md
└── mask_labeler/
    ├── __init__.py
    ├── cli.py              # `mask-labeler` entry; subcommands single / video / video-from-mask / video-multi
    ├── prompt_drawer.py    # interactive labeler (SAM2 ImagePredictor)
    ├── video_propagate.py  # SAM2 video propagation (SAM2VideoPredictor)
    └── vis.py              # overlay_mask helper
```

## Pipeline

```
single mode:
  imread → PromptDrawer.run(rgb) → mask (HxW bool) → write uint8 PNG

video mode:
  imread seed → PromptDrawer.run() → seed mask
              → propagate_video()
                  ├─ copy frame range to tempdir as 000000.jpg, 000001.jpg, ...
                  ├─ SAM2VideoPredictor.init_state(tempdir)
                  ├─ add_new_mask(state, frame=0, obj_id=1, seed>0)
                  └─ propagate_in_video(state) → per-frame masks
              → write <out>/<key>.mask.png + vis overlay JPGs  (key = source stem)

video-from-mask mode:
  imread seed mask from --seed-mask → propagate_video()
                                  ├─ same tempdir/init_state path as video mode
                                  └─ write the same mask/vis outputs

video-multi mode:
  resolve seed→to range → label first N consecutive frames (default 5)
                       → save each manual mask immediately
                       → propagate_video_multi()
                           ├─ same tempdir/init_state path as video mode
                           ├─ add_new_mask(...) for each manual frame local index
                           └─ propagate_in_video(state) → per-frame masks
                       → write the same mask/vis outputs; manual masks are preserved exactly
```

## Strategies (the things that shape the code)

### Single SAM2 backend
RoboCamCalib uses SAM v1 for clicks and SAM2 only for video. Here both modes
share SAM2's ImagePredictor / VideoPredictor — one checkpoint, one autocast
policy. Means you can't fall back to SAM v1 without code changes; that's fine
because SAM2 large is strictly better and is already on disk.

### Window-fit without xrandr
Original `PromptDrawer.run` shelled out to `xrandr` to size the window. That's
X11-only and brittle inside Docker. Here `max_window_size: tuple[int,int]` is
constructor-injected (CLI: `--max-window 1600x900`).

### Mouse coordinates → image coordinates
The display image is scaled by `self.ratio = min(max_w/W, max_h/H, 1.0)`.
Mouse callback stores raw display coordinates; `_detect()` divides them by
`self.ratio` before handing to SAM2. Box rendering on the preview stays in
display coordinates — keep this invariant if you touch either path.

### Mask combination rules (from RoboCamCalib, kept verbatim)
For multiple box prompts on the same image:
- positive box → `final = final OR new_mask`
- negative box → `final = final AND (NOT new_mask)`
- init_mask (if set) seeds `final` before the loop

Point prompts are always passed to every SAM2 call; their labels (0/1) live
in SAM2's own prompt system.

## Output format

- Single mode: `<out>` is a single PNG, uint8 with values 0 or 255.
- Video / video-multi mode: `<out_dir>/<key>.mask.png` (uint8 0/255) and
  `<out_dir>/vis/<key>.jpg` (RGB overlay), where `<key>` is the SOURCE frame
  stem (`_frame_key`): zero-padded `{int:06d}` for integer-named frames
  (`000000.mask.png`), verbatim stem for timestamp names
  (`20260604_200559_111996.mask.png`). Keyed by the source frame, not the
  propagation-local index.

## Gotchas

- **CUDA required for SAM2 video**. CPU mode might work for `single` on
  small images but propagation will be unusable; the autocast branch is gated
  on `device == "cuda"`.
- **`sam-2` is a git dep**, not on PyPI. First `uv sync` will clone facebookresearch/sam2;
  if that fails, network or git auth is the cause.
- **Default checkpoint path is host-specific** (`/home/yuzeren/sudo/ws/RoboCamCalib/sam2.1_hiera_large.pt`).
  Override with `MASK_LABELER_SAM2_CKPT` or `--sam2-checkpoint`.
- **`frame-pattern` matters** — RoboCamCalib hardcoded `*.jpg`; this tool
  accepts any glob, but the tempdir copy always re-encodes non-JPG inputs to
  JPG since `SAM2VideoPredictor.init_state` requires JPEG-numbered frames. When
  `--frame-pattern` is left at its `*.jpg` default AND `--seed-frame` is a path,
  `cmd_video` auto-switches the pattern to the seed file's extension (e.g.
  `*.png`), so timestamp `.png` dirs work without passing the flag.
- **Endpoint forms (`seed-frame` / `to-frame`)**: each accepts an integer index
  OR a path / basename / stem. Frame order is `_canonical_order` =
  `sorted(glob)`, sorted numerically only when EVERY stem is an integer (so
  ragged-width integer dirs stay correct), else lexicographic — which is
  chronological for fixed-width `YYYYMMDD_HHMMSS_micro` timestamp names.
  Resolvers live in `video_propagate.py` (`resolve_seed_position`,
  `resolve_to_exclusive`); `propagate_video` now takes POSITIONS (`seed_pos`,
  `to_pos` exclusive), not integer stem values, plus the prebuilt `order`.
  Critical subtlety: Python `int("2026_06_04")` succeeds (underscores are digit
  separators), so `_int_or_none` is STRICT `isascii()+isdigit()` — without that,
  timestamp stems would be misclassified as integers.
- **`to-frame` inclusivity**: a path/basename/stem `to-frame` is INCLUSIVE (that
  frame is masked; converted to an exclusive bound one step past it toward the
  seed). A bare integer `to-frame` stays EXCLUSIVE (legacy, via `bisect_left`
  over integer stems — reproduces the old value-bound even when absent on disk).
  Sentinels: `end`/`last` include the final frame, `start`/`first` the first.
- **`frames-dir` optional**: derived from `--seed-frame`'s parent when omitted
  and the seed is a path. `seed_img_path = order[seed_pos]` (the exact matched
  file) — no longer reconstructed as `<seed:06d><suffix>`.
- **Seed-mask direct propagation**: `video-from-mask --seed-mask path/to/mask.png`
  is a separate non-interactive command. It skips `PromptDrawer` entirely and
  uses that mask as the seed condition for SAM2 video propagation. The mask is
  read as 2D bool; RGB/RGBA masks use RGB channels only so opaque black alpha
  backgrounds do not become foreground. Keep `video` itself interactive to
  preserve existing scripts and behavior.
- **Seed-mask reuse**: `video` mode auto-loads `<out_dir>/<seed key>.mask.png`
  (`_frame_key` of the seed: `000000.mask.png` for integer dirs,
  `20260604_200559_111996.mask.png` for timestamps) as the labeler's init_mask
  if it exists — lets you iterate on the seed without restarting.
  `video-multi` does the same for each manual frame's output mask; its
  `--preload-mask` applies only to the first manual frame.
- **Sparse / subset frame dirs**: propagation does NOT assume dense numbering.
  It propagates over the positions actually present in `order` between the
  endpoints. SAM2 sees the kept frames as a contiguous video, so a large stride
  means larger inter-frame motion and more tracking drift — inherent to masking
  a subset, not a bug.
- **GPU memory / long clips**: `init_state` defaults to loading every decoded
  frame onto the GPU and OOMs on long clips or small cards. `video` mode now
  passes `offload_video_to_cpu=True` and `offload_state_to_cpu=True` by default
  (bounded VRAM, small fps cost). Use `--no-offload-video` / `--no-offload-state`
  to keep them on GPU when VRAM is plentiful.

## Provenance

- `prompt_drawer.py` ← `RoboCamCalib/robocamcalib/data_pipeline/prompt_drawer.py`
- `video_propagate.py` ← `RoboCamCalib/robocamcalib/data_pipeline/label_and_track_mask.py::track_with_sam2`
- `vis.overlay_mask` ← `RoboCamCalib/robocamcalib/utils/vis_utils.py::overlay_mask`

When upstream RoboCamCalib evolves the labeler, port relevant changes here
manually — there is no submodule link.
