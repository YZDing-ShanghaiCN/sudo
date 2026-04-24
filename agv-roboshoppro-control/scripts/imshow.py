import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import yaml


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def open_camera_by_index(index: int):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap.isOpened():
        return cap

    cap.release()
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        return cap

    cap.release()
    return None


def probe_camera_indexes(max_index: int) -> list:
    available = []
    for idx in range(max_index + 1):
        cap = open_camera_by_index(idx)
        if cap is not None:
            available.append(idx)
            cap.release()
    return available


def build_detector(dictionary_name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV未包含aruco模块，请安装 opencv-contrib-python")

    if not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(f"不支持的字典: {dictionary_name}")

    dict_id = getattr(cv2.aruco, dictionary_name)
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)

    if hasattr(cv2.aruco, "DetectorParameters"):
        params = cv2.aruco.DetectorParameters()
    else:
        params = cv2.aruco.DetectorParameters_create()

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, params)

        def _detect(gray):
            corners, ids, _ = detector.detectMarkers(gray)
            return corners, ids
    else:
        def _detect(gray):
            corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
            return corners, ids

    return _detect


def marker_center_and_area(pts):
    cx = float((pts[0][0] + pts[1][0] + pts[2][0] + pts[3][0]) / 4.0)
    cy = float((pts[0][1] + pts[1][1] + pts[2][1] + pts[3][1]) / 4.0)
    area = float(abs(cv2.contourArea(pts.astype("float32"))))
    return cx, cy, area


def detect_target(gray, detect, target_id: int):
    corners, ids = detect(gray)
    if ids is None or len(ids) == 0:
        return None

    ids_flat = ids.flatten().tolist()
    for i, mid in enumerate(ids_flat):
        if int(mid) != target_id:
            continue

        pts = corners[i][0]
        cx, cy, area = marker_center_and_area(pts)
        return {
            "corners": corners[i],
            "pts": pts,
            "cx": cx,
            "cy": cy,
            "area": area,
        }

    return None


def build_record(ts: float, target_id: int, cx: float, cy: float, w: int, h: int, area: float, extra: dict):
    rec = {
        "ts": ts,
        "marker_id": target_id,
        "u_px": cx,
        "v_px": cy,
        "u_norm": cx / float(w),
        "v_norm": cy / float(h),
        "du_px": cx - (w / 2.0),
        "dv_px": cy - (h / 2.0),
        "marker_area_px": area,
    }
    if extra:
        rec.update(extra)
    return rec


def warmup_camera(cap, warmup_frames: int):
    for _ in range(max(0, warmup_frames)):
        cap.read()


def capture_snapshot(
    cap,
    detect,
    target_id: int,
    sample_count: int,
    max_frames: int,
    min_marker_area_px: float,
    show: bool,
):
    accepted = []
    scanned = 0
    last_frame = None

    while scanned < max_frames and len(accepted) < sample_count:
        ok, frame = cap.read()
        if not ok:
            continue

        scanned += 1
        last_frame = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hit = detect_target(gray, detect, target_id)

        if hit is not None and hit["area"] >= min_marker_area_px:
            accepted.append({
                "cx": hit["cx"],
                "cy": hit["cy"],
                "area": hit["area"],
            })

        if show:
            if hit is not None:
                cv2.aruco.drawDetectedMarkers(frame, [hit["corners"]])
                cv2.circle(frame, (int(hit["cx"]), int(hit["cy"])), 5, (0, 255, 0), -1)
            cv2.putText(
                frame,
                f"snapshot {len(accepted)}/{sample_count}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow("camera_position_2d", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    if not accepted:
        return None, scanned, last_frame

    h, w = last_frame.shape[:2]
    xs = [x["cx"] for x in accepted]
    ys = [x["cy"] for x in accepted]
    areas = [x["area"] for x in accepted]

    cx = float(statistics.median(xs))
    cy = float(statistics.median(ys))
    std_u = float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0
    std_v = float(statistics.pstdev(ys)) if len(ys) > 1 else 0.0

    rec = build_record(
        ts=time.time(),
        target_id=target_id,
        cx=cx,
        cy=cy,
        w=w,
        h=h,
        area=float(statistics.median(areas)),
        extra={
            "mode": "snapshot",
            "samples_used": len(accepted),
            "snapshot_scanned_frames": scanned,
            "u_std_px": std_u,
            "v_std_px": std_v,
            "radial_std_px": (std_u ** 2 + std_v ** 2) ** 0.5,
        },
    )
    return rec, scanned, last_frame


def main():
    parser = argparse.ArgumentParser(description="USB相机2D位置检测（无标定）")
    parser.add_argument("--config", default="config/camera.yaml")
    parser.add_argument("--camera-index", type=int, default=None, help="覆盖配置中的相机索引")
    parser.add_argument("--list-cameras", action="store_true", help="列出可用相机索引后退出")
    parser.add_argument("--max-camera-index", type=int, default=5, help="扫描相机索引上限")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--output", default="output/camera_2d_positions.jsonl")
    parser.add_argument("--append", action=argparse.BooleanOptionalAction, default=True, help="是否追加写入输出文件")
    parser.add_argument("--max-frames", type=int, default=0, help="0表示不限")
    parser.add_argument("--warmup-frames", type=int, default=20, help="相机预热帧数")
    parser.add_argument("--min-marker-area-px", type=float, default=0.0, help="最小标记面积阈值（像素）")
    parser.add_argument("--run-id", default="", help="测试批次ID")
    parser.add_argument("--shot-id", default="", help="单次导航任务ID")
    parser.add_argument("--snapshot", action="store_true", help="采集一条稳健快照后退出")
    parser.add_argument("--snapshot-samples", type=int, default=20, help="快照模式最少有效样本数")
    parser.add_argument("--snapshot-max-frames", type=int, default=240, help="快照模式最多扫描帧数")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    cam_cfg = cfg["camera"]
    marker_cfg = cfg["marker"]

    if args.list_cameras:
        available = probe_camera_indexes(args.max_camera_index)
        if available:
            print(json.dumps({"available_camera_indexes": available}, ensure_ascii=False))
        else:
            print(json.dumps({"available_camera_indexes": []}, ensure_ascii=False))
        return

    detect = build_detector(marker_cfg.get("dictionary", "DICT_4X4_50"))
    target_id = int(marker_cfg["id"])

    camera_index = int(cam_cfg.get("index", 0)) if args.camera_index is None else int(args.camera_index)
    cap = open_camera_by_index(camera_index)
    if cap is None:
        raise RuntimeError(f"无法打开USB相机，camera_index={camera_index}")

    print(json.dumps({"using_camera_index": camera_index}, ensure_ascii=False))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cam_cfg.get("width", 1280)))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cam_cfg.get("height", 720)))
    cap.set(cv2.CAP_PROP_FPS, int(cam_cfg.get("fps", 30)))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_mode = "a" if args.append else "w"

    warmup_camera(cap, args.warmup_frames)

    common_meta = {}
    if args.run_id:
        common_meta["run_id"] = args.run_id
    if args.shot_id:
        common_meta["shot_id"] = args.shot_id

    if args.snapshot:
        try:
            rec, scanned, _ = capture_snapshot(
                cap=cap,
                detect=detect,
                target_id=target_id,
                sample_count=max(1, args.snapshot_samples),
                max_frames=max(1, args.snapshot_max_frames),
                min_marker_area_px=max(0.0, args.min_marker_area_px),
                show=args.show,
            )
            if rec is None:
                raise RuntimeError(f"快照模式未检测到目标marker(id={target_id})，已扫描{scanned}帧")

            rec.update(common_meta)
            print(json.dumps(rec, ensure_ascii=False))
            with out_path.open(output_mode, encoding="utf-8") as fout:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return
        finally:
            cap.release()
            if args.show:
                cv2.destroyAllWindows()

    frame_count = 0
    try:
        with out_path.open(output_mode, encoding="utf-8") as fout:
            while True:
                ok, frame = cap.read()
                if not ok:
                    continue

                h, w = frame.shape[:2]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hit = detect_target(gray, detect, target_id)

                if hit is not None and hit["area"] >= args.min_marker_area_px:
                    rec = build_record(
                        ts=time.time(),
                        target_id=target_id,
                        cx=hit["cx"],
                        cy=hit["cy"],
                        w=w,
                        h=h,
                        area=hit["area"],
                        extra={
                            "mode": "stream",
                            "frame_idx": frame_count,
                            **common_meta,
                        },
                    )

                    print(json.dumps(rec, ensure_ascii=False))
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()

                    if args.show:
                        cv2.aruco.dra4848wDetectedMarkers(frame, [hit["corners"]])
                        cv2.circle(frame, (int(hit["cx"]), int(hit["cy"])), 5, (0, 255, 0), -1)
                        cv2.putText(
                            frame,
                            f"u={hit['cx']:.1f}, v={hit['cy']:.1f}",
                            (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )

                if args.show:
                    cv2.imshow("camera_position_2d", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break

                frame_count += 1
                if args.max_frames > 0 and frame_count >= args.max_frames:
                    break
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()