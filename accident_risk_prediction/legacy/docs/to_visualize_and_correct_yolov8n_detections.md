You have two reasonable paths: _visual QA only_ (just look) or _visual QA + correction_ (look, edit, re-import). The pipeline already saves YOLO outputs as JSON, so corrections fit cleanly into the existing flow. Let me add three small helper scripts under a new tools/ folder.

### Tool 1 — render YOLOv8 detections onto frames for visual QA

### Tool 2 — export detections as YOLO .txt for editing in LabelImg / CVAT / Roboflow

### Tool 3 — re-import corrected .txt back into detections.json

All three tools parse. Here's the full workflow.

## Workflow: see YOLOv8 detections and correct them

_Step 1 — generate detections_ (once per split, you may already have done this):

python -m pipelines.detection_pipeline --config configs/config.yaml --split train

That writes data/annotations/detections/train/<video>/detections.json.

_Step 2 — render boxes onto frames for eyeballing:_

# single video

python -m tools.visualize_detections --split train --video video_dhaka_sunny_001

# or for numeric video folders

python -m tools.visualize_detections --split train --video 2

> The visualizer supports both frame naming conventions:
>
> - `frame_000001.jpg`
> - `frame_<video>_000001.jpg`

**Step 3 — if you spot bad boxes, export to YOLO .txt for editing:**

python -m tools.export_yolo_labels --config configs/config.yaml --split train --video video_dhaka_sunny_001

This produces data/qa/labels/train/<video>/:
frame_xxx.jpg (copies of the originals)
frame_xxx.txt (YOLO format: class cx cy w h, normalised)
classes.txt (car, motorcycle, bicycle, pedestrian, bus, truck, rickshaw)

_Step 4 — open the folder in a labeller and fix boxes._ Any tool that reads YOLO format works:

| Tool           | Notes                                                                                                                             |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| _LabelImg_     | pip install labelImg && labelImg data/qa/labels/train/<video>/ data/qa/labels/train/<video>/classes.txt — fastest for local edits |
| _CVAT_         | Web UI, supports import/export of YOLO 1.1 — best for team labelling                                                              |
| _Roboflow_     | Drag the folder in, edit in browser, re-export YOLO format                                                                        |
| _Label Studio_ | Open source, YOLO format supported                                                                                                |

Add, delete, resize boxes, change class IDs — whatever the frames need.

_Step 5 — re-import the corrected labels into the pipeline:_

python -m tools.import_yolo_labels --config configs/config.yaml --split train --video video_dhaka_sunny_001

This overwrites detections.json with your corrections (the original is auto-backed up to detections.json.bak).

_Step 6 — propagate corrections downstream by re-running tracking and graphs:_

python -m pipelines.tracking_pipeline --config configs/config.yaml --split train
python -m pipelines.graph_pipeline --config configs/config.yaml --split train

That's it. Because the downstream stages (tracking → graph → training) only read detections.json, your hand corrections become the ground truth for the rest of the system — while YOLOv8 itself stays frozen, exactly per the MD contract.

### Quick tips

- The bbox format in detections.json is **absolute pixel [x, y, w, h]** (top-left corner + size). The .txt files use **normalised cx cy w h** — the export/import scripts handle the conversion both ways.
- Class IDs follow configs/config.yaml → yolo.class_mapping (car=0, motorcycle=1, bicycle=2, pedestrian=3, bus=4, truck=5, rickshaw=6). COCO IDs from YOLOv8n are remapped to this scheme at detection time, so the .txt files always use our class IDs.
- If you only want to inspect a handful of frames without setting up a labeller, just open the annotated images from step 2 — that's usually enough to decide whether YOLOv8n is good enough for that video or needs a manual pass.
