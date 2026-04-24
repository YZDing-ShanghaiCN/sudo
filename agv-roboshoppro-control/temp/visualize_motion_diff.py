"""Visualize pixel-level differences between two IR motion-check frames.

Outputs a 2x3 panel figure:
    (1) Frame A (grayscale)
    (2) Frame B (grayscale)
    (3) Absolute difference |A - B|
    (4) Difference heatmap (JET colormap)
    (5) Binary motion mask (Otsu threshold)
    (6) Mask overlay on Frame B
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

IMG_DIR_DEFAULT: Path = Path(
    "/home/u24/ws_lq/sudo/aaa_useful_scripts/V4L2/multi_hard_sync/V4L2/"
    "obs_data/session_20260417_163314/TestIR/motion_check"
)
IMG_A_DEFAULT: str = "000408.jpg"
IMG_B_DEFAULT: str = "002828.jpg"


def load_gray(path: Path) -> NDArray[np.uint8]:
    """Load an image from disk and return a single-channel grayscale array.

    Args:
        path: Absolute path to the image file.

    Returns:
        A 2D uint8 array containing the grayscale image.

    Raises:
        FileNotFoundError: If the image cannot be read.
    """
    img: NDArray[np.uint8] | None = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def compute_difference(
    a: NDArray[np.uint8], b: NDArray[np.uint8]
) -> tuple[NDArray[np.uint8], NDArray[np.uint8], float]:
    """Compute absolute difference and Otsu-thresholded binary mask.

    Args:
        a: Frame A, single-channel uint8.
        b: Frame B, single-channel uint8 of identical shape.

    Returns:
        A tuple of (absolute difference, binary mask 0/255, Otsu threshold value).

    Raises:
        ValueError: If the two frames have mismatched shapes.
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    diff: NDArray[np.uint8] = cv2.absdiff(a, b)
    # Light Gaussian smoothing suppresses single-pixel sensor noise before
    # thresholding; kernel size 5 is a conservative default for IR imagery.
    diff_blur: NDArray[np.uint8] = cv2.GaussianBlur(diff, (5, 5), 0)
    thresh_value, mask = cv2.threshold(
        diff_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return diff, mask.astype(np.uint8), float(thresh_value)


def build_overlay(
    base_gray: NDArray[np.uint8], mask: NDArray[np.uint8]
) -> NDArray[np.uint8]:
    """Overlay a red translucent mask on a grayscale base image.

    Args:
        base_gray: Single-channel uint8 image used as the background.
        mask: Binary mask (0/255) marking pixels to highlight.

    Returns:
        A 3-channel RGB uint8 image with the masked region shaded red.
    """
    base_rgb: NDArray[np.uint8] = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2RGB)
    red_layer: NDArray[np.uint8] = np.zeros_like(base_rgb)
    red_layer[..., 0] = 255  # R channel
    alpha: float = 0.45
    mask_bool: NDArray[np.bool_] = mask.astype(bool)
    overlay: NDArray[np.uint8] = base_rgb.copy()
    overlay[mask_bool] = (
        (1.0 - alpha) * base_rgb[mask_bool] + alpha * red_layer[mask_bool]
    ).astype(np.uint8)
    return overlay


def summarize(
    diff: NDArray[np.uint8], mask: NDArray[np.uint8], thresh_value: float
) -> dict[str, float]:
    """Compute scalar statistics describing the inter-frame difference.

    Args:
        diff: Absolute-difference image (uint8).
        mask: Binary motion mask (0/255).
        thresh_value: Otsu threshold used when generating the mask.

    Returns:
        A dict of summary statistics suitable for printing and annotation.
    """
    total_pixels: int = int(mask.size)
    changed_pixels: int = int(np.count_nonzero(mask))
    stats: dict[str, float] = {}
    stats["mean_abs_diff"] = float(np.mean(diff))
    stats["max_abs_diff"] = float(np.max(diff))
    stats["otsu_threshold"] = float(thresh_value)
    stats["changed_pixel_ratio"] = changed_pixels / total_pixels
    return stats


def render_panel(
    a: NDArray[np.uint8],
    b: NDArray[np.uint8],
    diff: NDArray[np.uint8],
    mask: NDArray[np.uint8],
    overlay: NDArray[np.uint8],
    stats: dict[str, float],
    name_a: str,
    name_b: str,
    output_path: Path,
) -> None:
    """Render a 2x3 comparison figure and save it to disk.

    Args:
        a: Frame A grayscale image.
        b: Frame B grayscale image.
        diff: Absolute difference image.
        mask: Binary motion mask.
        overlay: RGB image with mask overlaid on Frame B.
        stats: Dict returned by ``summarize``.
        name_a: Display name for Frame A.
        name_b: Display name for Frame B.
        output_path: Destination PNG path.
    """
    fig_raw, axes = plt.subplots(2, 3, figsize=(18, 10))
    # Pyright narrows ``plt.subplots`` to ``FigureBase``; cast to ``Figure`` so
    # downstream calls to ``tight_layout`` / ``savefig`` type-check cleanly.
    fig: Figure = fig_raw  # type: ignore[assignment]

    axes[0, 0].imshow(a, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title(f"Frame A: {name_a}")

    axes[0, 1].imshow(b, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title(f"Frame B: {name_b}")

    im_diff = axes[0, 2].imshow(diff, cmap="gray", vmin=0, vmax=255)
    axes[0, 2].set_title("Absolute Difference |A - B|")
    fig.colorbar(im_diff, ax=axes[0, 2], fraction=0.035)

    im_heat = axes[1, 0].imshow(diff, cmap="jet", vmin=0, vmax=255)
    axes[1, 0].set_title("Difference Heatmap (JET)")
    fig.colorbar(im_heat, ax=axes[1, 0], fraction=0.035)

    axes[1, 1].imshow(mask, cmap="gray", vmin=0, vmax=255)
    axes[1, 1].set_title(
        f"Motion Mask (Otsu={stats['otsu_threshold']:.1f}, "
        f"ratio={stats['changed_pixel_ratio'] * 100.0:.2f}%)"
    )

    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title("Mask Overlay on Frame B")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    suptitle: str = (
        f"Inter-frame Difference | mean={stats['mean_abs_diff']:.2f}  "
        f"max={stats['max_abs_diff']:.0f}  "
        f"changed={stats['changed_pixel_ratio'] * 100.0:.2f}%"
    )
    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The populated ``argparse.Namespace``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        type=Path,
        default=IMG_DIR_DEFAULT,
        help="Directory containing the two frames.",
    )
    parser.add_argument("--a", type=str, default=IMG_A_DEFAULT, help="Frame A filename.")
    parser.add_argument("--b", type=str, default=IMG_B_DEFAULT, help="Frame B filename.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/home/u24/ws_lq/tmp/motion_diff.png"),
        help="Output PNG path for the panel figure.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively in addition to saving it.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: load frames, compute diff, render panel, print stats."""
    args: argparse.Namespace = parse_args()
    path_a: Path = args.dir / args.a
    path_b: Path = args.dir / args.b

    frame_a: NDArray[np.uint8] = load_gray(path_a)
    frame_b: NDArray[np.uint8] = load_gray(path_b)

    diff, mask, thresh_value = compute_difference(frame_a, frame_b)
    overlay: NDArray[np.uint8] = build_overlay(frame_b, mask)
    stats: dict[str, float] = summarize(diff, mask, thresh_value)

    render_panel(
        frame_a,
        frame_b,
        diff,
        mask,
        overlay,
        stats,
        name_a=args.a,
        name_b=args.b,
        output_path=args.out,
    )

    print(f"[motion-diff] saved panel to: {args.out}")
    print(f"[motion-diff] mean_abs_diff      = {stats['mean_abs_diff']:.3f}")
    print(f"[motion-diff] max_abs_diff       = {stats['max_abs_diff']:.1f}")
    print(f"[motion-diff] otsu_threshold     = {stats['otsu_threshold']:.1f}")
    print(
        "[motion-diff] changed_pixel_ratio = "
        f"{stats['changed_pixel_ratio'] * 100.0:.3f}%"
    )

    if args.show:
        import matplotlib.image as mpimg

        img = mpimg.imread(str(args.out))
        plt.figure(figsize=(16, 9))
        plt.imshow(img)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
