# Training a Bangladesh traffic detector

The main pipeline uses **YOLO26x** (`weights/yolo26x.pt`), the most accurate
COCO detector Ultralytics ships. COCO has no rickshaw, CNG or easy-bike
classes, so it labels CNGs as "car" or "bus" and misses rickshaws. It also
has no idea that the handlebars at the bottom of the frame are your own bike.
This guide fine-tunes a detector that knows all of these.

Final classes (`configs/detector.yaml`):

| id | class | notes |
|---|---|---|
| 0 | car | car, SUV, taxi, microbus |
| 1 | motorcycle | other riders only, not your own bike |
| 2 | bicycle | |
| 3 | pedestrian | |
| 4 | bus | bus, minibus, leguna / human hauler |
| 5 | truck | truck, pickup, covered van |
| 6 | rickshaw | pedal rickshaw and van-rickshaw |
| 7 | cng | CNG / auto-rickshaw (green or grey) |
| 8 | easy_bike | battery auto, e-rickshaw |
| 9 | ego_vehicle | your own handlebars, tank, mirrors, arms. The pipeline drops this class. |

## Fastest route: train on the ODS_New dataset (ready now)

`D:\Thesis\ODS_New` is a labelled Bangladeshi road dataset: 17,326 street-level
frames from videos (including rainy and foggy ones), 81,542 boxes, 17 classes
(Bicycle, Bus, Bhotbhoti, Car, CNG, Easybike, Leguna, Motorbike, MPV,
Pedestrian, Pickup, PowerTiller, Rickshaw, ShoppingVan, Truck, Van,
Wheelbarrow). No hand-labelling is needed. Settings: `configs/detector_ods.yaml`.

```powershell
python -m tools.detector.prepare_ods            # 1. split (15 s)
python -m tools.detector.train_ods --quick      # 2a. optional 2-minute check that everything runs
python -m tools.detector.train_ods              # 2b. real training, then automatic evaluation
python -m tools.detector.evaluate_detector      # 3. re-run the evaluation any time
```

**1. Split.** 80 / 10 / 10 into train / val / test, written as hard links in
`data/detector/ods/`, so it takes no extra disk space and never changes
`ODS_New`. Neighbouring images are frames of the same clip and look alike, so
images move in blocks of 50 consecutive frames. A random image-by-image split
would put near-copies of test images into training and inflate the scores.
Rainy and foggy images are split separately, so every split has some. Blocks
are swapped until every class has about 10% of its boxes in val and in test.
`split_report.md` shows the counts. 112 labels with boxes reaching outside the
image are clipped in the copy only.

**2. Train.** YOLO26m from COCO weights, 640 px, up to 150 epochs, with early
stopping after 30 epochs without improvement. Expect roughly 5–9 hours on the
RTX 3060. Leave the PC on; if it restarts, continue with
`python -m tools.detector.train_ods --resume`. Results:
`runs/detector/ods_yolo26m/` (`weights/best.pt`, training curves).
- Out of GPU memory: set `batch: 8`.
- "Paging file too small": set `workers: 2`.
- To try 960 px (better for small, distant vehicles, about 2.3x slower):
  set `imgsz: 960` and `batch: 8`, and use `--name ods_yolo26m_960` so the
  first run is kept.

**3. Evaluate.** Writes `runs/detector/ods_yolo26m/eval_test/report.md` with:
1. standard per-class precision, recall, mAP50 and mAP50-95 (Ultralytics);
2. accuracy for the 10 groups the risk pipeline uses (Car + MPV = car,
   Pickup + Truck + Van = truck, Rickshaw + ShoppingVan = rickshaw, ...);
3. accuracy by weather (normal / rainy / foggy);
4. recall by object size (small = far away);
5. comparison with the COCO YOLO26x the pipeline uses now. In a test run,
   YOLO26x put a box of any class on only 18% of rickshaws, 37% of easy-bikes
   and 0% of wheelbarrows. This table is the evidence for the thesis that a
   Bangladesh detector is needed;
6. speed, and the exact lines to paste into `configs/config.yaml`.

The test split is used only for this final report. Training picks its best
epoch on val, so the test numbers are unbiased.

**4. Use it.** Paste the `yolo:` and `ego_filter:` block from the end of the
report into `configs/config.yaml`. The detector keeps its 17 classes; the
`class_map` groups them into 10 pipeline classes, and `agnostic_nms` makes sure
one object gets one box. Then rebuild:

```powershell
python -m pipelines.build_video_cache --config configs/config.yaml --videos data/raw_videos --steps detect track graphs --force
python -m tools.make_event_sequences --config configs/config.yaml
```

The number of classes changes from 6 to 10, so the risk model must be
retrained (its graph features include the class). Check a few videos with
`tools/visualize_cache.py` first.

**What ODS cannot do.** ODS has no "own bike" class, so the ego-filter
heuristic stays on. ODS frames are not helmet footage either. The ODS test
score is the accuracy on Bangladeshi roads in general, not on your camera. If
time allows, label 300–500 of your own helmet frames and fine-tune
`ods_yolo26m/weights/best.pt` on them. That also gives a helmet-POV accuracy
number. The labelling tools below currently use a 10-class list, so they need
switching to the 17 ODS classes first: a small config change.

## The labelling route (own helmet frames)

### The plan in one line

Pre-label your own helmet frames automatically, correct them by hand, add a
public Bangladeshi dataset for extra rickshaw / CNG examples, fine-tune
YOLO26m, then point the main config at the new weights.

## Step 1. Pick frames from your videos

```powershell
python -m tools.detector.sample_frames --videos data/raw_videos
```

Takes 15 frames spread through each video, plus 5 from the 3 seconds before
every marked collision. Near-duplicate frames are skipped. With 168 videos
that is roughly 2,500–3,000 frames. If you mark collisions first, those
critical moments are included automatically.

## Step 2. Pre-label them

```powershell
python -m tools.detector.prelabel
```

YOLO26x draws car / motorcycle / bicycle / pedestrian / bus / truck boxes,
and one `ego_vehicle` box over your own bike. In a test on 3 videos this
pre-filled about 10 boxes per frame. Your corrections are never overwritten
if you run it again.

## Step 3. Correct the labels by hand (the real work)

Use **CVAT** (free, best for this) or **Label Studio**. Both run in the
browser and read YOLO-format labels.

**CVAT, the simplest route:**
1. Create a free account at cvat.ai, or run it locally with Docker.
2. Create a project and add the 10 labels in exactly the order of the table above.
3. Create a task per batch of about 300 images (upload from
   `data/detector/own/images/<video>/`).
4. Upload the pre-labels: *Actions → Upload annotations → YOLO 1.1*.
   Zip the matching `.txt` files together with `obj.names` (copy of
   `data/detector/own/labels/classes.txt`) and `obj.data`. The CVAT docs
   describe the exact zip layout for "YOLO 1.1".
5. Correct each frame (rules below).
6. Export with *Export task dataset → YOLO 1.1* and copy the `.txt` files
   back into `data/detector/own/labels/<video>/`, replacing the pre-labels.

**Labelling rules (write these down; consistency matters more than speed):**
- Change every CNG and easy-bike from car / bus to its real class. This is
  the most important correction.
- Draw every rickshaw. YOLO never pre-labels them.
- Exactly one `ego_vehicle` box per frame where your bike is visible,
  covering handlebars, dashboard and your hands.
- Draw a box around a person **riding** a motorcycle, bicycle or rickshaw
  only as part of the vehicle, not as a separate pedestrian.
- Skip objects smaller than about 10 pixels or more than half hidden.
- Delete wrong boxes rather than guessing.

**Time estimate:** about 1 minute per corrected frame, so 40–50 hours for
2,500 frames. Splitting it between two people also gives you the
inter-annotator agreement number examiners like: have both label the same
100 frames and compare.

## Step 4. Add a public Bangladeshi dataset (strongly recommended)

Your own frames will have few rickshaw / CNG / easy-bike examples at first.
A public dataset adds thousands. Candidates (check each one's licence, class
list and image count yourself before relying on it):

- **DhakaAI** (Dhaka traffic detection challenge): around 3,000 images, 21
  classes including rickshaw, CNG, auto-rickshaw, leguna and human hauler.
- **Poribohon-BD**: a Bangladeshi vehicle-detection dataset.
- **BadODD**: a newer Bangladeshi road-scene detection dataset.
- Bangladeshi vehicle sets on **Roboflow Universe** and **Kaggle**. Quality
  varies; open some images before using one.

To add one: download it into `data/detector/public/<name>/`, then add an
entry under `public_datasets:` in `configs/detector.yaml`. A commented
example for DhakaAI is already there: copy it, check the `names` list
against the dataset's own class list, and adjust the `map`.

Public images are used for **training only**. Validation uses only your own
helmet videos, so the reported accuracy is the accuracy that matters.

## Step 5. Merge

```powershell
python -m tools.detector.merge_datasets
```

Prints a class-count table. Classes with fewer than 200 training boxes are
flagged. Aim for at least 300–500 boxes of each class, especially rickshaw,
cng and easy_bike. If one is short, label more frames from videos where
that vehicle appears, or add another public dataset.

## Step 6. Train

```powershell
python -m tools.detector.train_detector
```

Fine-tunes YOLO26m (downloaded automatically on first use, about 45 MB).
On an RTX 3060 expect roughly 2–6 hours for 100 epochs on 5,000–10,000
images; it stops early when validation stops improving. Results go to
`runs/detector/bd_traffic/`.

At the end it prints the mAP for each class on your held-out helmet
videos. Rough targets: overall mAP50 above 0.6, and rickshaw / cng above
0.5. That per-class table goes straight into the thesis.

If the GPU runs out of memory, set `batch: 8` in `configs/detector.yaml`.

Why YOLO26m and not x? The x model is about 3 times slower to train for a
small accuracy gain, and with a few thousand images the data, not the model
size, is the limit. If time allows, train both and compare.

## Step 7. Use it in the main pipeline

The training script prints the exact lines to paste into
`configs/config.yaml` (`yolo.weights`, `yolo.classes`, `yolo.class_map`,
and `ego_filter.enabled: false`). Then rebuild detections and graphs:

```powershell
python -m pipelines.build_video_cache --config configs/config.yaml --videos data/raw_videos --steps detect track graphs --force
python -m tools.make_event_sequences --config configs/config.yaml
```

Check the result visually with `tools/visualize_cache.py`, then retrain.

## Thesis angle

This gives you a reportable contribution on its own: per-class mAP of
COCO YOLO26x versus your fine-tuned model on helmet footage, especially for
rickshaw, CNG and easy-bike. You can also compare the accident-risk model's
results with both detectors as an ablation.
