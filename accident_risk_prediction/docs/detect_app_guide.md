# Detector Demo App — Guide and Report

An interactive web app for running the Bangladesh-trained **YOLO26m** road-user
detector on images and videos. You upload media in a browser page, the app draws
boxes, counts objects per class, tracks them across video frames, and lets you
download the annotated results and a CSV of every detection.

- **App code:** `tools/detector/detect_app.py` (single file, ~330 lines)
- **Model used by default:** `runs/detector/ods_yolo26m/weights/best.pt`
- **Built:** 2026-11-19 · Streamlit 1.44, Ultralytics 8.4, OpenCV 4.13, PyTorch 2.7 (CUDA)

---

## 1. Where everything is

All paths are relative to `D:\Thesis\accident_risk_prediction\`.

| File | What it is |
|---|---|
| `tools/detector/detect_app.py` | **The app.** UI, inference, tracking, drawing, video encoding and downloads. |
| `.streamlit/config.toml` | Streamlit settings picked up when you launch from this folder: 4 GB upload limit, file watcher off. |
| `runs/detector/ods_yolo26m/weights/best.pt` | The trained detector the app loads by default (17 classes). |
| `configs/detector_ods.yaml` | Read for its `pipeline.map` section: the 17 → 10 class grouping shown by the "pipeline groups" switch. |
| `requirements.txt` | Now also lists `streamlit` and `imageio-ffmpeg` (bottom of the file). |
| `../.claude/launch.json` | Only for Claude Code's browser pane (`detect-app` entry). Not needed to run the app yourself. |

The app does not modify anything in the project. Uploaded files and generated
videos go to a temporary folder (`%TEMP%\bd_det_*`).

---

## 2. Setup (one time)

Everything except `imageio-ffmpeg` was already installed in your Python 3.13
environment; `imageio-ffmpeg` was installed while building the app. On a new
machine:

```powershell
cd D:\Thesis\accident_risk_prediction
python -m pip install -r requirements.txt
```

Minimum needed for the app alone:

```powershell
python -m pip install ultralytics streamlit opencv-python pandas pyyaml imageio-ffmpeg
```

`imageio-ffmpeg` ships its own `ffmpeg.exe`; you don't need to install ffmpeg
system-wide. A GPU is optional but strongly recommended for video.

---

## 3. How to run

```powershell
cd D:\Thesis\accident_risk_prediction
```

```powershell
streamlit run tools/detector/detect_app.py
```

Streamlit prints a URL and normally opens your browser. If it doesn't, open
**http://localhost:8501**. The first page load takes a few seconds while the
model loads ("Loading model…" in the sidebar); after that the model stays in
memory.

**To stop it:** press `Ctrl+C` in the terminal where it is running.

> Launch from the `accident_risk_prediction` folder. If you launch from
> elsewhere, the app still finds its weights (paths are resolved from the
> script's location), but `.streamlit/config.toml` is not picked up, so the
> upload limit falls back to 200 MB. Then add
> `--server.maxUploadSize 4096` to the command.

Useful variants:

```powershell
streamlit run tools/detector/detect_app.py --server.port 8600
```

```powershell
streamlit run tools/detector/detect_app.py --server.headless true
```

(`headless` = don't auto-open a browser, e.g. when running on a lab machine and
connecting from another computer at `http://<machine-ip>:8501`.)

---

## 4. Using the app

### 4.1 Sidebar (applies to both tabs)

| Control | Default | What it does |
|---|---|---|
| **Weights (.pt)** | `runs/detector/ods_yolo26m/weights/best.pt` | Path to any Ultralytics detection model. Paste another path (e.g. `last.pt`, a newer run, or `weights/yolo26x.pt` for the COCO model) to compare models. Class names are read from the weights. |
| **Device** | `cuda:0` if available, else `cpu` | Where inference runs. |
| **Confidence threshold** | 0.25 | Boxes below this score are dropped. 0.25 is the same operating point used in `configs/detector_ods.yaml` (`eval.conf`). Lower it to see more (and more false) detections. |
| **NMS IoU** | 0.70 | Overlap above which duplicate boxes of the same class are merged. Lower = more aggressive merging (helps if one vehicle gets two boxes; hurts in dense traffic where vehicles overlap). |
| **Inference size** | 640 | Image size the model sees. Trained at 640. 960/1280 can find more small, distant objects but are slower. |
| **Classes to detect** | all 17 | Remove classes you don't care about (e.g. keep only Pedestrian + Motorbike). |
| **Show pipeline groups** | off | Labels and counts use the 10 risk-pipeline groups instead of the 17 classes (MPV→car, Leguna→bus, Pickup/Van→truck, ShoppingVan→rickshaw, Bhotbhoti/PowerTiller/Wheelbarrow→slow_vehicle, …). Detection itself is unchanged. |
| **Show confidence** | on | Prints the score next to each label. |

Changing any sidebar setting re-runs image detection immediately. For video,
press **Run detection** again.

### 4.2 Images tab

1. Drag in one or several images (JPG, PNG, BMP, WEBP).
2. For each image you get:
   - the annotated image,
   - **Detections** count (hover the ⓘ to see inference time),
   - a per-class count table,
   - **Download annotated** (PNG at original resolution),
   - **All detections** (expand): class, group, confidence, box corners `x1,y1,x2,y2` in pixels.
3. With more than one image, **Download CSV for all images** gives one CSV with an `image` column.

### 4.3 Video tab

1. Upload a video (MP4, AVI, MOV, MKV, WEBM, M4V; up to 4 GB).
2. Options:
   - **Track objects** (on): uses ByteTrack so each object keeps an ID (`#12 Rickshaw`) across frames. Counts then mean *unique objects*, not boxes summed over frames.
   - **Process every Nth frame** (1): 2 = every other frame, about twice as fast. The output video then plays at the reduced frame rate (it looks shorter-framed but covers the same time). Tracking gets less reliable at high strides because objects move further between processed frames.
   - **Max seconds to process** (0 = whole video): handy for a quick check of a long ride.
3. Press **▶ Run detection**. You'll see a progress bar with speed and time left, and a live preview updated every 10 processed frames.
4. When it finishes:
   - the annotated video plays in the page (full resolution, H.264),
   - **Frames processed** and **Unique tracked objects**,
   - a per-class table (unique objects if tracking, otherwise total boxes),
   - **Download video (.mp4)** and **Download detections CSV**,
   - an area chart of boxes per second, by class.

The result stays on screen until you run another video or reload the page. You
can flip **Show pipeline groups** afterwards and the tables and chart re-group
without re-running. The labels burned into the video itself don't change.

### 4.4 Output CSV formats

**Images:** `class, group, conf, x1, y1, x2, y2, image`

**Video:** `frame, time_s, track_id, class, group, conf, x1, y1, x2, y2`

- `frame`: 0-based index in the original video (not the processed-frame count).
- `time_s`: `frame / fps`.
- `track_id`: empty when tracking is off, or for a box the tracker hasn't confirmed yet.
- `group`: the pipeline group, or the class name itself when groups are off.
- Coordinates are pixels in the original resolution.

---

## 5. How it works (report)

### 5.1 Design choice: Streamlit

Streamlit was already installed in the environment and turns one Python file
into a local web page with upload widgets, tables, charts and download buttons.
That needs no web code and no separate front end, and it runs entirely on
your machine; no images or video leave it. Gradio would have been an equal
choice but wasn't installed.

### 5.2 Processing flow

```
upload ─► decode (OpenCV) ─► YOLO26m predict / track (Ultralytics, GPU)
       ─► rows: class, group, conf, box, [frame, time, track id]
       ─► draw boxes (own OpenCV drawing, colour per class)
       ─► image: show + PNG     video: write mp4v ─► ffmpeg → H.264 ─► play + download
```

Key implementation points:

- **Model caching.** `@st.cache_resource` loads the weights once per weights
  path, so moving sliders doesn't reload the 44 MB model.
- **Own box drawing** (`draw()`), not Ultralytics' `plot()`, so labels can show
  the pipeline group, the track ID and the confidence together, with line width
  and font scaled to the frame size.
- **Class grouping** reuses `pipeline.map` in `configs/detector_ods.yaml`, so
  the app's groups always match what the risk pipeline uses.
- **Tracking** calls `model.track(..., persist=True, tracker="bytetrack.yaml")`
  frame by frame. Each run creates a fresh model instance, so track IDs start
  at 1 for every video and never leak from the previous run.
- **Video encoding.** OpenCV on Windows cannot write H.264 (the format browsers
  play), and the browser-friendly VP8 encoder in OpenCV is very slow at 1080p.
  So the app writes a fast `mp4v` file during processing, then converts it once
  to H.264 with the ffmpeg bundled in `imageio-ffmpeg` (`libx264`, `veryfast`,
  CRF 23, `+faststart`). If ffmpeg is unavailable, it falls back to a ≤640 px
  VP8 WebM preview and still offers the full-resolution mp4v for download.

### 5.3 Measured performance

Measured on this machine (CUDA GPU, PyTorch 2.7.1+cu118), 1920×1080 helmet
video `data/raw_videos/test/107.mp4` (first 4 s, 119 frames), 640 inference
size, tracking on:

| Stage | Speed |
|---|---|
| YOLO26m + ByteTrack alone | ~27 fps |
| OpenCV VP8 (WebM) encode, 1080p | ~3.6 fps (the original bottleneck) |
| OpenCV mp4v encode, 1080p | ~86 fps |
| **Whole app, first version** (VP8 + mp4v during processing) | **2.5 fps** (47 s for 4 s of video) |
| **Whole app, final version** (mp4v, then one ffmpeg H.264 pass) | **25.1 fps** (5 s for 4 s of video) |

So on the GPU the app processes 30 fps footage at close to real time. The final
H.264 conversion adds a few seconds per minute of video. On CPU expect roughly
1–3 fps; use **Process every Nth frame** and **Max seconds** to keep runs short.

### 5.4 Testing done

| Test | Result |
|---|---|
| App starts, model loads, all 17 class names shown | ✅ |
| Image upload (`ODS_New/images/1.jpg`) | ✅ 7 detections (pedestrians + an MPV), boxes and table correct |
| Video upload + tracking (4 s, 1080p clip of `107.mp4`) | ✅ 119 frames, 34 unique tracked objects (Motorbike 8, Rickshaw 7, Easybike 7, Leguna 6, …) |
| Output plays in the browser | ✅ H.264, 1920×1080, 3.97 s, no playback error |
| Same clip before/after the encoding change | ✅ identical counts (34), 10× faster |

Not tested: the no-ffmpeg WebM fallback path, CPU-only inference, very long
videos (>10 min), and the download buttons' files opened in an external player.

---

## 6. Limitations and things to know

- **Memory with long videos.** Streamlit keeps the uploaded video in RAM, and
  the app holds the output MP4 in memory for the player and download. A 2 GB
  upload therefore needs several GB of free RAM. For hour-long rides, trim first
  or use **Max seconds**.
- **Temp files are not deleted.** Each video run leaves a `%TEMP%\bd_det_*`
  folder (copy of the upload + output). Delete them occasionally:
  ```powershell
  Remove-Item "$env:TEMP\bd_det_*" -Recurse -Force
  ```
- **"Unique tracked objects" is an upper bound.** When an object is lost (occlusion,
  motion blur, leaving and re-entering the frame), ByteTrack may give it a new ID,
  so the same rickshaw can be counted twice. A brief false detection can also get
  its own ID. Treat the number as indicative, not an exact census.
- **The rider's own motorcycle.** In helmet-POV footage the ego bike (handlebar,
  mirrors, dashboard) is in frame. The ODS dataset has no `ego_vehicle` class, so
  it's normally *not* detected; if you see it boxed as Motorbike, raise the
  confidence threshold or ignore boxes touching the bottom edge.
- **One user at a time.** It's a local single-user tool, not a server for many
  people. Two browser tabs running videos at once share the same GPU.
- **Labels in the exported video are fixed** at the settings used when you pressed
  Run (groups on/off, confidence shown or not). Change settings and re-run to
  re-export.
- **Unknown frame count.** For some streams OpenCV can't read the total frame
  count; then the progress bar doesn't move until the end, but processing still works.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `streamlit` is not recognised | `python -m streamlit run tools/detector/detect_app.py` |
| "Weights file not found" | Check the path in the sidebar; default is `runs/detector/ods_yolo26m/weights/best.pt` under `accident_risk_prediction`. |
| Upload refused: file too large (200 MB) | You launched from another folder. `cd` into `accident_risk_prediction` first, or add `--server.maxUploadSize 4096`. |
| Device list shows only `cpu` | PyTorch can't see the GPU: `python -c "import torch; print(torch.cuda.is_available())"`. |
| `CUDA out of memory` | Lower **Inference size**, or close other GPU programs (e.g. a training run). |
| Video shows "Low-res preview" | `imageio-ffmpeg` is missing: `python -m pip install imageio-ffmpeg`. |
| "OpenCV could not open this video" | Unusual codec. Re-encode it first (e.g. to H.264 MP4) and upload again. |
| Port 8501 already in use | Another copy is running. Stop it with `Ctrl+C`, or use `--server.port 8600`. |

---

## 8. Ideas for later

- A **command-line batch mode** (folder of videos → annotated videos + CSVs)
  for processing the whole `raw_videos` set without the browser. Ultralytics' own
  CLI already covers the basic case:
  `yolo track model=runs/detector/ods_yolo26m/weights/best.pt source=video.mp4 save=True`.
- A **"hide ego vehicle" option** that drops boxes anchored at the bottom-centre
  of helmet footage.
- **Side-by-side model comparison** (e.g. fine-tuned YOLO26m vs COCO YOLO26x) on
  the same frame, useful as a figure for the thesis.
- **Automatic temp cleanup** of `bd_det_*` folders when the app starts.
