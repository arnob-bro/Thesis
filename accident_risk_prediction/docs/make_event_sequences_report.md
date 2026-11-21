# Event-based sequence generation (`make_event_sequences`)

This report describes how marked collision times are turned into the labelled
training sequences used by the risk model. It covers the inputs, the labelling
rule, which windows are discarded and why, how windows are sampled, how videos
are split into train/val/test and cross-validation folds, the output format,
and the design choices and limitations worth stating in the thesis.

Command (run from `accident_risk_prediction/`):

```powershell
python -m tools.make_event_sequences --config configs/config.yaml
```

Code involved:

| File | Role |
|---|---|
| `tools/make_event_sequences.py` | Entry point: reads events, plans windows per video, writes the manifest |
| `utils/windows.py` | Labelling rule (`label_from_tte`), window filtering (`classify`), sampling (`plan_windows`), fold assignment (`assign_folds`) |
| `utils/events.py` | Events-file schema and loading, split assignment (`assign_split`) |
| `utils/manifest_dataset.py` | How training reads the manifest and picks train/val/test from it (`split_rows`) |
| `configs/config.yaml` | All parameters (`sequence`, `frame_extraction`, `event_labeling`, `cv`, `split`) |

---

## 1. Place in the pipeline

```
raw videos
   │
   ├─► tools/mark_collisions.py        annotator marks collision moments + ignore ranges
   │        └─► data/annotations/collision_events.json
   │
   ├─► pipelines/build_video_cache.py  frames, detections, tracks, ego-motion, graphs, ViT
   │        └─► data/video_cache/<video_id>/
   │
   └─► tools/make_event_sequences.py   (this step)
            └─► data/manifests/sequences.csv  ──►  training / evaluation
```

The cache and the labels are kept separate on purpose. A sequence in the
manifest is only a reference — `(video_id, start frame, end frame, label)` —
into the per-video cache. No frames or features are copied. Changing the
labelling parameters (bins, strides, window length) therefore only requires
re-running this step, which takes seconds, not re-processing the videos.

---

## 2. Inputs

### 2.1 Collision annotations

`data/annotations/collision_events.json`, written by `tools/mark_collisions.py`.
One entry per video:

```json
"12.mp4": {
  "path": "data/raw_videos/train/12.mp4",
  "video_id": "12",
  "split": null,
  "status": "done",
  "events": [{"time_s": 63.4, "type": "collision", "note": ""}],
  "ignore": [[0.0, 5.2], [70.0, 95.0]]
}
```

- **`events`** — the moments of collision, in seconds. The event type
  (`collision`, `near_miss`, `loss_of_control`) is stored in the manifest but
  does **not** change the label: every event type is treated the same.
- **`ignore`** — time ranges that are not real riding footage: intros,
  replays, slow motion, text overlays, edits. No window may touch them.
- **`status`** — only videos marked `done` are used. Unfinished videos are
  skipped and listed in the console output.
- A `done` video with **no events** is valid; all its usable windows become
  *safe* samples.

### 2.2 Frame count per video

To know how many windows fit in a video, the script needs its number of
sampled frames. It reads `n_frames` from `data/video_cache/<video_id>/meta.json`.
If the video has not been cached yet, it opens the original file and computes
`floor(duration × 5 fps)`. If neither is available the video is skipped.

---

## 3. Windows and sequences

A **window** (called a **sequence** once written to the manifest) is a run of
consecutive sampled frames:

| Parameter | Config key | Value |
|---|---|---|
| Sampling rate | `frame_extraction.frame_rate` | 5 fps (one frame every 0.2 s) |
| Window length | `sequence.length` | 12 frames |
| Window duration | length ÷ rate | **2.4 s** |

A window is identified by the index of its **last** frame, `end`, with
`start = end − 11`. Its time is the time of that last frame, `t_end = end / 5`.

The window length is a free parameter. Changing it requires only re-running
this step (and retraining); changing the sampling rate requires rebuilding the
whole video cache.

---

## 4. Labelling rule: time-to-event

Each window is labelled by its **time-to-event (TTE)**:

```
TTE = (time of the next marked event) − t_end
```

Because TTE is measured from the window's **last** frame, the model only ever
sees footage from *before* the moment at which it has to make a prediction.
Nothing after `t_end` is visible to it.

TTE is mapped to one of four risk classes (`event_labeling.bins`, intervals are
`[min, max)` in seconds):

| TTE (s) | Class | Name | Interpretation (motorcycle profile) |
|---|---|---|---|
| 0 – 1.5 | 3 | high | Inside rider reaction time (~1–1.5 s): practically unavoidable |
| 1.5 – 3.0 | 2 | moderate | Avoidable only with immediate braking or swerving |
| 3.0 – 4.5 | 1 | slight | Conflict developing, normal correction still possible |
| 4.5 – 6.0 | — | *skipped* | Ambiguous buffer, not used for training |
| ≥ 6.0, or no later event | 0 | safe | No imminent conflict |

**Why these thresholds.** The bins are compressed compared with typical
car-based TTC thresholds (a car profile would be about 2 / 4 / 6 / 8 s).
Motorcycles in dense urban traffic move at lower speeds with tighter gaps, and
a rider can still lean or swerve out of a conflict later than a car can. The
upper edge of the *high* class is set at roughly the rider reaction time.
*(For the thesis: support the reaction-time figure and the car thresholds with
citations; the code only records the reasoning.)*

**Why the 4.5–6 s buffer is skipped.** Footage just outside the *slight* range
is neither clearly safe nor clearly risky — the conflict may already be
forming. Labelling it *safe* would teach the model that early warning signs
are normal; labelling it risky would over-extend the risk zone. Leaving it out
gives a clean margin between the safe class and the risk classes.

The TTE value itself is also kept (column `tte`) and used as the regression
target of the model's time-to-collision head, for windows in classes 1–3.

---

## 5. Windows that are discarded

A window is dropped entirely (not labelled, not used) if any of the following
holds (`classify` in `utils/windows.py`):

1. **It overlaps an ignore range.** Intros, replays, slow motion and editing
   artefacts are not real riding footage.
2. **It touches an event or its aftermath.** Any window whose span overlaps
   `[event time, event time + skip]` is dropped, where the skip depends on the
   event type (`post_event_skip`: 5 s for `collision` and `loss_of_control`,
   1 s for `near_miss`). The
   crash itself and the seconds after it (falling, stopping, people
   gathering) are neither a prediction target nor normal riding, and would be
   wrongly labelled *safe* (no further event ahead) otherwise. After a near
   miss the rider keeps riding, so only 1 s is skipped. If a separate event
   follows within a few seconds, the windows in between are its real lead-up
   and keep their TTE labels. After a crash, a second event less than 5 s
   later gets no windows of its own; the two are treated as one incident.
3. **An edit cut lies between the window and the event.** If the window is
   within 6 s of an event but an ignore range sits entirely between the
   window's end and the event, the real time gap is unknown (the video was
   cut), so TTE cannot be trusted. Such windows are dropped. This matters for
   compilation videos, where clips are often trimmed just before the crash.
4. **TTE falls in the 4.5–6 s buffer** (Section 4).

---

## 6. Sampling: how many windows per video

Taking every possible window would produce mostly overlapping, mostly safe
samples. Windows are instead sampled at two densities (`plan_windows`):

| Setting | Value | Applies to | Effect |
|---|---|---|---|
| `risk_stride` | 2 frames (0.4 s) | Classes 1–3 | Dense sampling before events, to get as many risky samples as possible |
| `safe_stride` | 12 frames (2.4 s) | Class 0 | Safe windows do not overlap each other |
| `max_safe_per_video` | 40 | Class 0 | At most 40 safe windows per video, spread evenly over the video |

The cap keeps long, uneventful rides from dominating the data set and limits
class imbalance. When a video has more than 40 candidate safe windows, 40 are
picked at even spacing over the whole video, so they still cover different
parts of the ride.

### Worked example

One video, one collision marked at **t = 60.0 s**, default settings (5 fps,
12-frame windows, risk stride 2). Window ends are frames 11, 13, 15, …

| Class | TTE range | `t_end` range | Window end frames | Windows |
|---|---|---|---|---|
| high (3) | 0 – 1.5 s | 58.6 – 59.8 s | 293, 295, 297, 299 | 4 |
| moderate (2) | 1.5 – 3.0 s | 57.4 – 58.2 s | 287, 289, 291 | 3 |
| slight (1) | 3.0 – 4.5 s | 55.6 – 57.0 s | 279, 281, 283, 285 | 4 |
| *buffer* | 4.5 – 6.0 s | 54.2 – 55.4 s | — | 0 (skipped) |
| safe (0) | ≥ 6 s | ≤ 54.0 s | 11, 23, 35, … (every 12) | up to 40 per video |
| *aftermath* | — | windows starting ≤ 65.0 s | ends ≤ 336 | 0 (dropped) |

Points the example makes clear:

- **Each event yields only about 11 risky windows** (3–4 per class). The
  number of marked events is therefore the main limit on the size of the
  risky part of the data set.
- Neighbouring risky windows share 10 of their 12 frames, so they are strongly
  correlated and should not be treated as independent samples. This is one
  reason the data is split by video (Section 7).
- The first usable window after the collision starts at 65.2 s (frame 326),
  after the 5 s aftermath skip.

---

## 7. Train / validation / test splits and folds

All assignment is done **per video, never per window**. All windows from one
ride end up in the same set, so near-identical overlapping windows from the
same scene cannot appear in both training and test data (no leakage).

Two independent assignments are written to the manifest:

### 7.1 Cross-validation fold (`fold` column, main protocol)

`cv.folds: 5`. Videos are divided into 5 folds (`assign_folds`):

1. Videos **with** at least one event are sorted, shuffled with the global
   seed (`seed: 42`) and dealt round-robin into folds 0–4.
2. Videos **without** events are shuffled and dealt the same way, continuing
   from where the first group stopped, so total fold sizes stay balanced.

This stratifies by "has an event or not", so every fold contains collision
videos. The assignment is deterministic for a fixed seed and set of videos.

When training with `--fold k` (`split_rows` in `utils/manifest_dataset.py`):

| Set | Folds |
|---|---|
| test | fold `k` |
| validation | fold `(k + 1) mod 5` |
| train | the remaining 3 folds |

Running k = 0…4 tests every video exactly once.

### 7.2 Fixed split (`split` column, used only without `--fold`)

For a single train/val/test run, each video gets a split by this priority
(`assign_split`):

1. An explicit `split` in the events file for that video.
2. The name of a parent folder (`train/`, `val/`, `test/`) in the video's path.
3. Otherwise a hash of the video id, giving about 70 / 15 / 15 %
   (`split: {train: 0.70, val: 0.15, test: 0.15}`). The hash makes the
   assignment stable: adding new videos never moves existing ones.

Unlike the folds, the hash split is not stratified by event presence.

---

## 8. Output: `data/manifests/sequences.csv`

One row per sequence (UTF-8 with BOM, so it opens cleanly in Excel):

| Column | Meaning |
|---|---|
| `sequence_id` | `<video_id>_seq<NNNN>`, numbered in time order within the video |
| `video_id` | ASCII-safe video id (folder name in the cache) |
| `source_file` | Original video file name |
| `split` | `train` / `val` / `test` (Section 7.2) |
| `fold` | 0–4 (Section 7.1) |
| `start`, `end` | First and last sampled-frame index (inclusive) in the cache |
| `risk_level` | 0 safe, 1 slight, 2 moderate, 3 high |
| `tte` | Time to the next event in seconds (empty if no later event) |
| `event_type` | Type of that next event (empty if none) |
| `event_time` | Time of that next event in seconds (empty if none) |
| `t_end` | Time of the window's last frame in seconds |

The script also prints, for each video, its fold, split, event times and
window count per class, followed by class-count tables per fold and per split.
These tables are the data-set statistics for the thesis.

`--dry-run` prints the same statistics without writing the file, which is the
way to compare settings (e.g. different window lengths) before committing.

### Data-set statistics (fill in after marking)

No videos had been marked yet when this report was written. After marking,
run the dry run and copy its tables here:

```powershell
python -m tools.make_event_sequences --config configs/config.yaml --dry-run
```

| Fold | Videos | Safe | Slight | Moderate | High | Total |
|---|---|---|---|---|---|---|
| 0 | | | | | | |
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |
| 4 | | | | | | |
| **Total** | | | | | | |

---

## 9. Parameter summary

| Key | Value | Meaning |
|---|---|---|
| `frame_extraction.frame_rate` | 5 | Sampled frames per second |
| `sequence.length` | 12 | Frames per window (2.4 s) |
| `event_labeling.bins.high` | [0.0, 1.5) | TTE range for class 3 |
| `event_labeling.bins.moderate` | [1.5, 3.0) | TTE range for class 2 |
| `event_labeling.bins.slight` | [3.0, 4.5) | TTE range for class 1 |
| `event_labeling.safe_min_tte` | 6.0 | Minimum TTE for class 0 |
| `event_labeling.post_event_skip` | collision 5.0 s, loss_of_control 5.0 s, near_miss 1.0 s | Aftermath excluded after each event, per type |
| `event_labeling.risk_stride` | 2 frames | Sampling step for risky windows |
| `event_labeling.safe_stride` | 12 frames | Sampling step for safe windows |
| `event_labeling.max_safe_per_video` | 40 | Safe-window cap per video |
| `cv.folds` | 5 | Number of cross-validation folds |
| `seed` | 42 | Seed for fold shuffling |

Command-line options: `--dry-run` (statistics only), `--videos <folder>`
(find videos not yet cached), `--events <file>` and `--manifest <file>`
(override the config paths).

---

## 10. Design choices and limitations (for the discussion chapter)

**Design choices**

- **Event-anchored labels instead of per-frame manual risk labels.** The
  annotator marks only the collision moment; risk levels follow
  automatically from the time before it. This is fast, consistent between
  annotators, and objective (it depends on when the crash happened, not on a
  subjective judgement of how risky a scene looks).
- **Causal labelling.** TTE is measured from the window's last frame, so the
  model is trained to anticipate, never to recognise a crash that is
  already visible.
- **Labels decoupled from features.** Because sequences are references into
  a cache, labelling parameters can be varied (ablations over bins, window
  length) at almost no cost.
- **Video-level splitting** prevents leakage between overlapping windows of
  the same ride.

**Limitations**

- **Labels depend on time, not on the scene.** Two windows 3 s before a crash
  get the same label whether the danger is already visible or not. Some
  windows labelled *slight* or *moderate* may look perfectly normal, which
  puts a ceiling on achievable accuracy for those classes.
- **Safe windows are assumed safe.** A *safe* window in a video may still
  contain a near-miss that was not marked. Label quality for class 0 depends
  on how completely near-misses are annotated.
- **All event types are treated alike.** Collisions, near-misses and loss of
  control produce the same labels; the type is recorded but not used.
- **Annotation precision.** The collision time is marked by hand. An error of
  a few tenths of a second shifts windows near the bin edges into the
  neighbouring class.
- **Few risky windows per event (~11).** The risky classes grow only with the
  number of marked events, and their windows are highly correlated.
- **Imbalance is reduced, not removed.** With up to 40 safe windows per video
  and ~11 risky windows per event, safe windows still dominate unless videos
  contain several events. The remaining imbalance is handled by class
  weighting in the training loss (`loss.class_weights`, computed from class
  counts when set to `auto`).
- **Fixed thresholds.** The bins assume one rider reaction time and urban
  speeds; they are not adapted to the actual speed in each clip.
