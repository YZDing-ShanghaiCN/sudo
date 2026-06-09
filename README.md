# mask_labeler

Interactive **SAM2** mask labeler + SAM2 video propagation.

Extracted from the [RoboCamCalib](https://git.sudoai.cc/realrobot/RoboCamCalib)
data pipeline (`PromptDrawer` + `LabelTrackMask`). The original used SAM v1 for the
interactive UI and SAM2 only for video — this tool unifies on SAM2 across both
modes so there's a single checkpoint to manage.

## Install

This is a standalone uv project — it does NOT use the SudoDeploy editable install.

```bash
cd tools/mask_labeler
uv sync
```

uv creates `.venv/` here with PyTorch (CUDA 12.8 — cu128 wheels are required
for Blackwell / RTX 50-series sm_120; earlier wheels lack those kernels), `sam-2` (built from
[facebookresearch/sam2](https://github.com/facebookresearch/sam2)), and the
remaining deps. First sync downloads ~3 GB; subsequent `uv sync`s are instant.

### Weights

Default points at the SAM2 large checkpoint already on disk:

```
/home/yuzeren/sudo/ws/RoboCamCalib/sam2.1_hiera_large.pt
```

Override with `--sam2-checkpoint PATH` or `export MASK_LABELER_SAM2_CKPT=...`.

## Usage

### Single image

```bash
uv run mask-labeler single \
    --image  /path/to/rgb.png \
    --out    /path/to/mask.png
```

Output is uint8 0/255 PNG. Pass `--preload-mask EXISTING.png` to start
iterative refinement on top of a previous result.

### Video propagation

`--seed-frame` / `--to-frame` accept either an **integer index** or a
**path / filename / stem**, so timestamp-named captures work directly:

```bash
# Timestamp-named frames (e.g. 20260604_200559_111996.png):
uv run mask-labeler video \
    --seed-frame '/media/yuzeren/My PSSD/20260604_200558/video/chest_left_camera/20260604_200559_111996.png' \
    --to-frame   '/media/yuzeren/My PSSD/20260604_200558/video/chest_left_camera/20260604_200600_233555.png' \
    --out-dir    /path/to/masks

# Integer-indexed frames (legacy):
uv run mask-labeler video \
    --frames-dir /path/to/rgb_frames --frame-pattern '*.png' \
    --seed-frame 0 --to-frame 20 \
    --out-dir    /path/to/masks
```

You label the seed frame interactively, then SAM2 propagates forward (or
backward if the to-frame precedes the seed). Outputs are named after the source
frame stem:

- `<out-dir>/20260604_200559_111996.mask.png` … one per frame
  (`000000.mask.png` for integer-named dirs)
- `<out-dir>/vis/<stem>.jpg` … overlay previews

The seed mask is auto-reloaded from `<out-dir>/<seed stem>.mask.png` if present,
so you can re-run to tweak the seed without restarting.

### Propagate from an existing mask

If the seed frame already has a mask and you do **not** want the interactive
labeling window, use the separate `video-from-mask` command. The mask must
match the seed frame size.

```bash
uv run mask-labeler video-from-mask \
    --seed-frame '/media/yuzeren/My PSSD/20260604_200558/video/chest_left_camera/20260604_200559_111996.png' \
    --to-frame   end \
    --seed-mask  /path/to/existing_seed.mask.png \
    --out-dir    /path/to/masks
```

Use `--frames-dir /path/to/rgb_frames --frame-pattern '*.png'` when
`--seed-frame` is an integer, filename, or stem rather than a full path.

### Multi-seed video propagation

Use `video-multi` when one seed frame is not stable enough. It preserves the
old `video` command and adds a separate workflow that labels several consecutive
frames first, then uses all of those masks as SAM2 video conditioning frames:

```bash
uv run mask-labeler video-multi \
    --seed-frame /path/to/first_rgb.png \
    --to-frame   /path/to/last_rgb.png \
    --manual-count 5 \
    --out-dir    /path/to/masks
```

`--manual-count` defaults to `5`. The frames are chosen from `--seed-frame`
toward `--to-frame`, so backward propagation labels the first five frames in
that backward direction. Each manual mask is saved immediately as
`<out-dir>/<stem>.mask.png`; existing masks in `out-dir` are preloaded for
iterative fixes, and `--preload-mask` applies to the first manual frame only.

**Endpoint forms & ranges.** Frame order is `sorted(glob)` (numeric when all
names are integers; for fixed-width timestamps this is chronological).
`--seed-frame` / `--to-frame` resolve as: absolute path → path under
`--frames-dir` → basename → stem (extension dropped) → integer index. A
**path/stem `--to-frame` is inclusive** (that frame is masked); a bare
**integer `--to-frame` is exclusive** (legacy). Use `--to-frame end`/`last` to
include the final frame, or `start`/`first` to run backward to the first.
`--frames-dir` may be omitted when `--seed-frame` is a path (its parent is
used), and when `--frame-pattern` is left at the default it follows the seed
file's extension.

**Sparse / subset frame dirs.** Numbering need not be contiguous — propagation
runs over whatever frames exist between the endpoints. Note SAM2 then treats the
kept frames as a continuous video, so a large stride between frames means more
tracking drift.

**GPU memory.** By default frames and per-frame state are offloaded to CPU RAM
(`offload_video_to_cpu` / `offload_state_to_cpu`), keeping VRAM roughly constant
in clip length so long sequences don't OOM. Pass `--no-offload-video` and/or
`--no-offload-state` to keep them on the GPU (faster, more VRAM) when memory is
plentiful.

## Keys

| Key       | Action                          |
| --------- | ------------------------------- |
| L-drag    | positive box (box mode)         |
| Ctrl+L-drag | negative box                  |
| L-click   | positive point (point mode)     |
| R-click   | positive point (point mode)     |
| Ctrl+L-click | negative point (point mode) |
| `b` / `p` | switch to box / point mode      |
| `r`       | reset all prompts               |
| `z`       | undo last prompt                |
| `Enter`   | save and exit                   |
| `q` / ESC | abort (no output written)       |

## Troubleshooting

- **`uv sync` fails on torch download** — adjust the CUDA index in
  `pyproject.toml` (replace `cu128` with your driver, e.g. `cu121`) and rerun.
  For CPU-only:
  ```bash
  uv add torch torchvision --index https://download.pytorch.org/whl/cpu --force
  uv sync
  ```
- **`Hydra config not found`** — the SAM2 config path is resolved inside the
  `sam2` package's `configs/` tree. The default `configs/sam2.1/sam2.1_hiera_l.yaml`
  matches the SAM2 large checkpoint. If you swap to a different size, set
  `--sam2-config` accordingly (e.g. `configs/sam2.1/sam2.1_hiera_b+.yaml`).
- **Window doesn't appear** — check `$DISPLAY` is set; mask labeling is
  inherently interactive and needs an X server.
- **`CUDA out of memory` during `init_state`** — offloading is on by default;
  if you passed `--no-offload-video`/`--no-offload-state`, drop them. You can
  also reduce fragmentation with
  `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **`missing frame …`** — no longer fatal; sparse frame dirs are propagated over
  the frames that exist. If the *seed* frame itself is absent the run still aborts.
