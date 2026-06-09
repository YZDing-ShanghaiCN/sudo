"""Interactive SAM2 prompt drawer.

Extracted from RoboCamCalib (robocamcalib/data_pipeline/prompt_drawer.py).
Backend swapped from SAM v1 (`segment_anything`) to SAM2's ImagePredictor —
the SAM2 large weights already on disk are reused by `video_propagate.py` too.
Window-fit no longer depends on `xrandr`; pass `max_window_size` explicitly.

Keys (in the labeler window):
  Enter   save and return the current mask
  q, ESC  abort (returns None)
  b / p   switch to box mode / point mode
  r       reset all prompts
  z       undo the last prompt

Box mode:
  L-drag           positive box (green)
  Ctrl + L-drag    negative box (red)
Point mode:
  L-click          positive point (green)
  R-click          positive point (green)
  Ctrl + L-click   negative point (red)
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

import cv2
import numpy as np
import torch


class DrawingMode(Enum):
    Box = 0
    Point = 1


class PromptDrawer:
    def __init__(
        self,
        sam2_checkpoint: str,
        sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        device: str = "cuda",
        window_name: str = "mask_labeler",
        max_window_size: tuple[int, int] = (1600, 900),
    ):
        self.window_name = window_name
        self.max_window_size = max_window_size
        self.device = device
        self.reset()

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        model = build_sam2(sam2_config, sam2_checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(model)

    # ----- prompt state -----

    def reset(self) -> None:
        self.done = False
        self.aborted = False
        self.drawing = False
        self.current = (0, 0)
        self.points = np.empty((0, 2))
        self.labels = np.empty((0,), dtype=int)
        self.boxes = np.zeros((0, 4), dtype=np.float32)
        self.box_labels = np.empty((0,), dtype=int)
        self.mask: Optional[np.ndarray] = None
        self.mode = DrawingMode.Box
        self.init_mask: Optional[np.ndarray] = None

    def set_init_mask(self, mask: Optional[np.ndarray]) -> None:
        """Seed the labeler with a previously-saved mask (uint8 or bool, same H,W as image)."""
        if mask is None:
            self.init_mask = None
            return
        self.init_mask = (np.asarray(mask) > 0)

    # ----- mouse callback -----

    def _on_mouse(self, event, x, y, flags, _):
        if self.done:
            return
        self.current = (x, y)
        if self.mode == DrawingMode.Box:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                label = 0 if (flags & cv2.EVENT_FLAG_CTRLKEY) else 1
                self.box_labels = np.hstack([self.box_labels, label])
                self.boxes = np.vstack([self.boxes, [x, y, x, y]])
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                self.boxes[-1, 2] = x
                self.boxes[-1, 3] = y
            elif event == cv2.EVENT_LBUTTONUP and self.drawing:
                self.drawing = False
                self.boxes[-1, 2] = x
                self.boxes[-1, 3] = y
                x1, y1, x2, y2 = self.boxes[-1]
                self.boxes[-1] = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                self._detect()
        else:  # Point
            if event == cv2.EVENT_LBUTTONDOWN:
                label = 0 if (flags & cv2.EVENT_FLAG_CTRLKEY) else 1
                self.points = np.vstack([self.points, [x, y]])
                self.labels = np.hstack([self.labels, label])
                self._detect()
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.points = np.vstack([self.points, [x, y]])
                self.labels = np.hstack([self.labels, 1])
                self._detect()

    # ----- SAM2 inference -----

    def _predict_one(
        self,
        point_coords: Optional[np.ndarray],
        point_labels: Optional[np.ndarray],
        box: Optional[np.ndarray],
    ) -> np.ndarray:
        with torch.inference_mode(), torch.autocast(
            "cuda" if self.device == "cuda" else "cpu",
            dtype=torch.bfloat16,
            enabled=(self.device == "cuda"),
        ):
            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=True,
            )
        # SAM2 returns float masks; binarize so callers can use logical ops.
        return masks[int(np.argmax(scores))] > 0.0

    def _detect(self) -> None:
        if len(self.points) > 0:
            pts = self.points / self.ratio
            lbls = self.labels.astype(int)
        else:
            pts = None
            lbls = None

        final: Optional[np.ndarray] = None
        if self.init_mask is not None:
            final = self.init_mask.copy()

        if len(self.boxes) == 0 and pts is not None:
            final = self._predict_one(pts, lbls, None)
        else:
            for i, box in enumerate(self.boxes):
                if (box[2] - box[0]) < 1 or (box[3] - box[1]) < 1:
                    continue
                lbl = self.box_labels[i]
                box_scaled = box / self.ratio
                m = self._predict_one(pts, lbls, box_scaled)
                if final is None:
                    final = m.copy()
                elif lbl == 0:
                    final = np.logical_and(final, ~m)
                else:
                    final = np.logical_or(final, m)

        if final is not None:
            self.mask = final
        elif self.mask is not None:
            self.mask = np.zeros_like(self.mask)

    # ----- main loop -----

    def run(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        """Open the window on `rgb` (HxWx3 uint8 RGB) and return the final mask (HxW bool) or None on abort."""
        self.predictor.set_image(rgb)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        max_w, max_h = self.max_window_size
        self.ratio = min(max_w / w, max_h / h, 1.0)
        target = (int(w * self.ratio), int(h * self.ratio))
        view = cv2.resize(bgr, target) if self.ratio != 1.0 else bgr.copy()

        if self.init_mask is not None:
            self.mask = self.init_mask.copy()

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        self._print_help()

        while not self.done:
            tmp = view.copy()

            if self.mask is not None:
                m = cv2.resize(self.mask.astype(np.uint8), target).astype(bool)
                from .vis import overlay_mask
                tmp = overlay_mask(tmp, m)

            for box, lbl in zip(self.boxes, self.box_labels):
                color = (0, 255, 0) if lbl == 1 else (0, 0, 255)
                cv2.rectangle(tmp, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 1)

            for (px, py), pl in zip(self.points, self.labels):
                color = (0, 255, 0) if pl == 1 else (0, 0, 255)
                cv2.circle(tmp, (int(px), int(py)), 5, color, -1)

            cv2.circle(tmp, self.current, 2, (0, 0, 255), -1)
            cv2.imshow(self.window_name, tmp)

            key = cv2.waitKey(50) & 0xFF
            if key == 13:               # Enter
                self.done = True
            elif key in (ord("q"), 27): # q / ESC
                self.done = True
                self.aborted = True
            elif key == ord("r"):
                init = self.init_mask
                self.reset()
                self.init_mask = init
            elif key == ord("p"):
                self.mode = DrawingMode.Point
                print("[mask_labeler] mode → POINT  (L=positive, Ctrl+L=negative, R=positive)")
            elif key == ord("b"):
                self.mode = DrawingMode.Box
                print("[mask_labeler] mode → BOX  (L-drag=positive, Ctrl+L-drag=negative)")
            elif key == ord("h"):
                self._print_help()
            elif key == ord("z"):
                if self.mode == DrawingMode.Point and len(self.points) > 0:
                    self.points = self.points[:-1]
                    self.labels = self.labels[:-1]
                    self._detect()
                elif self.mode == DrawingMode.Box and len(self.boxes) > 0:
                    self.boxes = self.boxes[:-1]
                    self.box_labels = self.box_labels[:-1]
                    self._detect()

        cv2.destroyWindow(self.window_name)
        return None if self.aborted else self.mask

    @staticmethod
    def _print_help() -> None:
        print(
            "\n"
            "─── mask_labeler keybindings ──────────────────────────────────\n"
            "  BOX mode (default):\n"
            "      L-drag        positive box (green)\n"
            "      Ctrl+L-drag   negative box (red)\n"
            "  POINT mode:\n"
            "      L-click       positive point (green)\n"
            "      Ctrl+L-click  negative point (red)\n"
            "      R-click       positive point (green)\n"
            "  Keys:\n"
            "      b / p         switch to box / point mode\n"
            "      z             undo last prompt\n"
            "      r             reset all prompts\n"
            "      Enter         save and exit\n"
            "      q / ESC       abort (no save)\n"
            "      h             reprint this help\n"
            "──────────────────────────────────────────────────────────────\n",
            flush=True,
        )

    def close(self) -> None:
        del self.predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
