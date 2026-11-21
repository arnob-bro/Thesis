"""Interactive detector demo: run the Bangladesh YOLO26m model on images or videos.

Launch (from the accident_risk_prediction folder):

    streamlit run tools/detector/detect_app.py

(.streamlit/config.toml in that folder raises the upload limit to 4 GB.)
Then open http://localhost:8501. Upload one or more images, or a video, and the
app draws the boxes, counts objects per class and lets you download the results
(annotated image / video + a CSV of every detection).
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[2]            # accident_risk_prediction/
DEFAULT_WEIGHTS = ROOT / "runs/detector/ods_yolo26m/weights/best.pt"
CONFIG = ROOT / "configs/detector_ods.yaml"
IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]
VIDEO_TYPES = ["mp4", "avi", "mov", "mkv", "webm", "m4v"]

# 20 well-separated BGR colours, one per class id.
PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
    (10, 249, 72), (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0),
    (168, 153, 44), (255, 194, 0), (147, 69, 52), (255, 115, 100), (236, 24, 0),
    (255, 56, 132), (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255),
]

st.set_page_config(page_title="BD Road Detector", page_icon="🛺", layout="wide")


# ----------------------------------------------------------------------------- helpers
@st.cache_resource(show_spinner="Loading model…")
def load_model(weights: str):
    from ultralytics import YOLO
    return YOLO(weights)


@st.cache_data
def load_group_map() -> dict[str, str]:
    """Fine class -> pipeline group (e.g. MPV -> car) from configs/detector_ods.yaml."""
    try:
        return yaml.safe_load(CONFIG.read_text())["pipeline"]["map"]
    except Exception:
        return {}


def find_ffmpeg() -> str | None:
    import shutil
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def to_h264(ffmpeg: str, src: Path, dst: Path) -> bool:
    import subprocess
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(dst)]
    try:
        return subprocess.run(cmd, capture_output=True).returncode == 0 and dst.stat().st_size > 0
    except Exception:
        return False


def device_options() -> list[str]:
    import torch
    return (["cuda:0"] if torch.cuda.is_available() else []) + ["cpu"]


def draw(frame: np.ndarray, boxes, names: dict, label_fn, show_conf: bool,
         track_ids=None, thickness: int | None = None) -> np.ndarray:
    """Draw boxes on a BGR frame (in place) and return it."""
    h, w = frame.shape[:2]
    t = thickness or max(1, round((h + w) / 900))
    fs = max(0.4, t / 3)
    for i, (xyxy, cls, conf) in enumerate(boxes):
        x1, y1, x2, y2 = map(int, xyxy)
        color = PALETTE[int(cls) % len(PALETTE)]
        text = label_fn(names[int(cls)])
        if track_ids is not None and track_ids[i] is not None:
            text = f"#{track_ids[i]} {text}"
        if show_conf:
            text += f" {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, t, cv2.LINE_AA)
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, t - 1))
        ty = y1 - 4 if y1 - th - base - 4 > 0 else y1 + th + base + 4
        cv2.rectangle(frame, (x1, ty - th - base), (x1 + tw + 4, ty + 2), color, -1)
        cv2.putText(frame, text, (x1 + 2, ty - base + 1), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (255, 255, 255), max(1, t - 1), cv2.LINE_AA)
    return frame


def result_rows(res, names, label_fn, frame_idx=None, time_s=None):
    """Turn an Ultralytics result into (draw-tuples, track ids, table rows)."""
    b = res.boxes
    if b is None or len(b) == 0:
        return [], [], []
    xyxy = b.xyxy.cpu().numpy()
    cls = b.cls.cpu().numpy().astype(int)
    conf = b.conf.cpu().numpy()
    ids = b.id.cpu().numpy().astype(int).tolist() if b.id is not None else [None] * len(cls)
    rows = []
    for (x1, y1, x2, y2), c, p, tid in zip(xyxy, cls, conf, ids):
        row = {"class": names[c], "group": label_fn(names[c]), "conf": round(float(p), 3),
               "x1": round(float(x1), 1), "y1": round(float(y1), 1),
               "x2": round(float(x2), 1), "y2": round(float(y2), 1)}
        if frame_idx is not None:
            row = {"frame": frame_idx, "time_s": round(time_s, 3), "track_id": tid, **row}
        rows.append(row)
    return list(zip(xyxy, cls, conf)), ids, rows


def count_table(df: pd.DataFrame, key: str, unique_tracks: bool = False) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[key, "count"])
    if unique_tracks and "track_id" in df and df["track_id"].notna().any():
        s = df.dropna(subset=["track_id"]).groupby(key)["track_id"].nunique()
    else:
        s = df[key].value_counts()
    return s.sort_values(ascending=False).rename("count").reset_index()


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Model")
    weights = st.text_input("Weights (.pt)", str(DEFAULT_WEIGHTS))
    if not Path(weights).is_file():
        st.error("Weights file not found.")
        st.stop()
    model = load_model(weights)
    names: dict[int, str] = model.names
    device = st.selectbox("Device", device_options())

    st.header("Detection")
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    iou = st.slider("NMS IoU", 0.1, 0.9, 0.7, 0.05)
    imgsz = st.select_slider("Inference size", [320, 480, 640, 800, 960, 1280], 640,
                             help="Model was trained at 640. Larger can help tiny/distant objects.")
    chosen = st.multiselect("Classes to detect", list(names.values()), default=list(names.values()))
    class_ids = [k for k, v in names.items() if v in chosen]

    st.header("Labels")
    group_map = load_group_map()
    use_groups = st.toggle("Show pipeline groups (10) instead of 17 classes", False,
                           help="e.g. MPV→car, Leguna→bus, ShoppingVan→rickshaw",
                           disabled=not group_map)
    show_conf = st.toggle("Show confidence", True)

label_fn = (lambda n: group_map.get(n, n)) if use_groups else (lambda n: n)
pred_args = dict(conf=conf, iou=iou, imgsz=imgsz, classes=class_ids or None,
                 device=device, verbose=False)

st.title("Bangladesh road-user detector")
st.caption(f"`{Path(weights).relative_to(ROOT) if Path(weights).is_relative_to(ROOT) else weights}` · "
           f"{len(names)} classes: {', '.join(names.values())}")
if not class_ids:
    st.warning("No classes selected.")
    st.stop()

tab_img, tab_vid = st.tabs(["🖼️ Images", "🎞️ Video"])

# ----------------------------------------------------------------------------- images
with tab_img:
    files = st.file_uploader("Upload image(s)", type=IMAGE_TYPES, accept_multiple_files=True)
    all_rows = []
    for f in files or []:
        img = cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            st.error(f"Could not read {f.name}")
            continue
        t0 = time.perf_counter()
        res = model.predict(img, **pred_args)[0]
        ms = (time.perf_counter() - t0) * 1000
        boxes, _, rows = result_rows(res, names, label_fn)
        for r in rows:
            r["image"] = f.name
        all_rows += rows
        out = draw(img.copy(), boxes, names, label_fn, show_conf)

        st.subheader(f.name)
        c1, c2 = st.columns([3, 1])
        with c1:
            st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_container_width=True)
        with c2:
            st.metric("Detections", len(rows), help=f"{ms:.0f} ms")
            df = pd.DataFrame(rows)
            st.dataframe(count_table(df, "group" if use_groups else "class"),
                         hide_index=True, use_container_width=True)
            ok, png = cv2.imencode(".png", out)
            st.download_button("Download annotated", png.tobytes(),
                               f"{Path(f.name).stem}_det.png", "image/png", key=f"dl_{f.name}")
        with st.expander("All detections"):
            st.dataframe(df, hide_index=True, use_container_width=True)
    if len(files or []) > 1 and all_rows:
        st.download_button("Download CSV for all images",
                           pd.DataFrame(all_rows).to_csv(index=False), "detections.csv", "text/csv")

# ----------------------------------------------------------------------------- video
with tab_vid:
    vfile = st.file_uploader("Upload a video", type=VIDEO_TYPES)
    c1, c2, c3 = st.columns(3)
    track = c1.toggle("Track objects (IDs across frames)", True,
                      help="ByteTrack. Gives each object an ID so counts are unique objects, not boxes.")
    stride = c2.number_input("Process every Nth frame", 1, 30, 1,
                             help="2 = half the frames, ~2x faster; output video plays at the reduced rate.")
    max_sec = c3.number_input("Max seconds to process (0 = all)", 0, 36000, 0)
    run = st.button("▶ Run detection", type="primary", disabled=vfile is None)

    if run and vfile is not None:
        tmp = Path(tempfile.mkdtemp(prefix="bd_det_"))
        src = tmp / vfile.name
        src.write_bytes(vfile.getvalue())
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            st.error("OpenCV could not open this video.")
            st.stop()
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        limit = min(total, int(max_sec * fps)) if max_sec else total
        out_fps = fps / stride

        # mp4v is fast to write but browsers can't play it, so it is converted to
        # H.264 at the end. Without ffmpeg, a small VP8 WebM is written as the preview.
        ffmpeg = find_ffmpeg()
        raw_p, mp4_p = tmp / "raw.mp4", tmp / f"{src.stem}_det.mp4"
        w_raw = cv2.VideoWriter(str(raw_p), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (W, H))
        w_prev, prev_size = None, None
        if not ffmpeg:
            pw = min(W, 640)
            prev_size = (pw, round(H * pw / W / 2) * 2)
            w_prev = cv2.VideoWriter(str(tmp / "preview.webm"), cv2.VideoWriter_fourcc(*"VP80"),
                                     out_fps, prev_size)

        # A fresh model instance per run = fresh tracker state (IDs start at 1 again).
        from ultralytics import YOLO
        vmodel = YOLO(weights)

        bar = st.progress(0.0, "Starting…")
        live = st.empty()
        rows, idx, done, t0 = [], -1, 0, time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if limit and idx >= limit:
                break
            if idx % stride:
                continue
            if track:
                res = vmodel.track(frame, persist=True, tracker="bytetrack.yaml", **pred_args)[0]
            else:
                res = vmodel.predict(frame, **pred_args)[0]
            boxes, ids, r = result_rows(res, names, label_fn, idx, idx / fps)
            rows += r
            out = draw(frame, boxes, names, label_fn, show_conf, ids if track else None)
            w_raw.write(out)
            if w_prev is not None:
                w_prev.write(cv2.resize(out, prev_size, interpolation=cv2.INTER_AREA))
            done += 1
            if done % 10 == 1:
                small = cv2.resize(out, (960, round(H * 960 / W))) if W > 960 else out
                live.image(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), use_container_width=True,
                           caption=f"frame {idx}")
            if limit:
                el = time.perf_counter() - t0
                frac = min(1.0, (idx + 1) / limit)
                bar.progress(frac, f"Frame {idx + 1}/{limit} · {done / el:.1f} fps · "
                                   f"~{el / frac - el:.0f}s left")
        cap.release(); w_raw.release()
        if w_prev is not None:
            w_prev.release()
        del vmodel
        el = time.perf_counter() - t0
        bar.progress(1.0, f"Done: {done} frames in {el:.0f}s ({done / max(el, 1e-6):.1f} fps)")
        live.empty()

        playable = False
        if ffmpeg:
            with st.spinner("Encoding H.264…"):
                playable = to_h264(ffmpeg, raw_p, mp4_p)
        if not playable:
            raw_p.replace(mp4_p)
        st.session_state["video_result"] = dict(
            mp4=mp4_p.read_bytes(), playable=playable, name=src.stem,
            preview=(tmp / "preview.webm").read_bytes() if w_prev is not None else None,
            df=pd.DataFrame(rows), frames=done, track=track)

    vr = st.session_state.get("video_result")
    if vr:
        df, key = vr["df"], "group" if use_groups else "class"
        c1, c2 = st.columns([3, 1])
        with c1:
            if vr["playable"]:
                st.video(vr["mp4"], format="video/mp4")
            elif vr["preview"]:
                st.video(vr["preview"], format="video/webm")
                st.caption("Low-res preview. `pip install imageio-ffmpeg` for full-res playback here.")
            else:
                st.info("Video encoded but can't be previewed here; download it below.")
        with c2:
            st.metric("Frames processed", vr["frames"])
            if vr["track"]:
                st.metric("Unique tracked objects",
                          int(df["track_id"].nunique()) if not df.empty else 0)
                st.caption("Unique objects per class")
            else:
                st.caption("Total boxes per class (no tracking)")
            st.dataframe(count_table(df, key, unique_tracks=vr["track"]),
                         hide_index=True, use_container_width=True)
            st.download_button("Download video (.mp4)", vr["mp4"], f"{vr['name']}_det.mp4", "video/mp4")
            st.download_button("Download detections CSV", df.to_csv(index=False),
                               f"{vr['name']}_detections.csv", "text/csv")
        if not df.empty:
            st.caption("Boxes per second of video")
            per_s = (df.assign(sec=df["time_s"].astype(int))
                       .pivot_table(index="sec", columns=key, values="conf", aggfunc="count")
                       .fillna(0))
            st.area_chart(per_s)
