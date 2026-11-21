# Collision Marking Guide

How to mark collision / near-miss / loss-of-control times in the raw videos, and
how our three-person team splits the work so the labels are consistent and
defensible in the thesis.

Tool: [`tools/mark_collisions.py`](../tools/mark_collisions.py)
Run every command from `D:\Thesis\accident_risk_prediction`.

---

## 1. Why the marks matter

Labels are not drawn by hand per sequence any more. Each video gets one or more
**event times**. Every training window is then labelled by its *time-to-event*
(TTE): the gap between the window's last frame and the next event.

| Class | TTE |
|---|---|
| high | < 1.5 s |
| moderate | 1.5 – 3 s |
| slight | 3 – 4.5 s |
| (skipped) | 4.5 – 6 s |
| safe | ≥ 6 s, or no upcoming event |
| (dropped) | the aftermath right after an event (section 2.1) |

The bins are only 1.5 s wide, so **a mark that is 0.5 s late moves a third of
the windows into the wrong class**. Precise, consistent marking is the whole
point of this step.

The tool writes to `data/annotations/collision_events.json` by default
(`event_labeling.events_file` in `configs/config.yaml`). For team work each
person uses their own file instead (section 4).

---

## 2. What to mark: shared definitions

Everyone uses these definitions. If a case doesn't fit, write it in the doubt
sheet (section 4.4) instead of guessing.

| Key | Type | Mark the frame where... |
|---|---|---|
| `c` | **collision** | the ego bike (or the rider) first makes **physical contact** with another road user or object. If contact isn't visible, mark the first frame where the fall or impact jolt is visible. |
| `n` | **near_miss** | the other road user is at its **closest approach** without contact, in a situation where a collision was clearly avoided only by braking or swerving. Ordinary close traffic is *not* a near miss. |
| `x` | **loss_of_control** | the first visible sign of the rider losing control with no other road user involved: the start of a skid, wobble or tank-slap, or a solo fall. |
| `[` … `]` | **ignore range** | anything that is not live first-person footage: intros, titles, **replays**, slow motion, cuts or edits, black frames, a static camera after the crash. |

Rules:

1. **Step frame by frame (`a` / `d`) before you press the key.** Never mark while the video is playing.
2. **Always put replays inside an ignore range.** Otherwise a replayed crash counts as a second event.
3. **One incident = one mark.** If one event leads directly into the next (a skid that ends in hitting a car, a collision that becomes a fall), mark only the **first** moment, however far apart they are. The model should learn to anticipate the start of the incident.
4. **Separate incidents = separate marks, even if they are close.** A near miss with a rickshaw followed 3 s later by a collision with a bus is two marks (`n`, then `c`). See section 2.1 for what the pipeline does with them.
5. **Mark every video, including those with no event.** If a video has no event, just press `s`; it then counts as safe footage (negatives). If you skip it, it is simply missing.
6. **Don't mark the aftermath.** The pipeline already skips it after each event (section 2.1).

To decide between rule 3 and rule 4, ask: *would the second event have happened
without the first one?* If not, it's the same incident.

### 2.1 Events close together (< 5 s apart)

The pipeline drops the seconds after each event, and how long it drops depends
on the event type (`post_event_skip` in `configs/config.yaml`):

| First event | Skipped after it | What happens to a second event 3 s later |
|---|---|---|
| `collision` | 5 s | gets no training windows: it falls inside the crash aftermath, so the two count as one incident |
| `loss_of_control` | 5 s | same as above |
| `near_miss` | 1 s | keeps its own lead-up: the windows between the two events are labelled by time to the second event (moderate / high) |

So always mark both separate incidents. Marking both never creates wrong
labels, and it keeps the record accurate if the skip values are changed later.

---

## 3. Using the tool

### 3.1 Interactive player (recommended)

```powershell
python -m tools.mark_collisions --config configs/config.yaml --videos data/raw_videos
```

| Key | Action |
|---|---|
| `space` | play / pause |
| `d` / `a` (or → / ←) | ±1 frame |
| `l` / `j` | ±1 s |
| `L` / `J` | ±10 s |
| `+` / `-` | playback speed |
| `c` / `n` / `x` | mark collision / near miss / loss of control |
| `[` then `]` | ignore range |
| `u` | undo the last mark |
| `s` | save as **done**, go to the next video |
| `w` | save as in-progress, go to the next video |
| `q` / `Esc` | save and quit |

Useful flags:

| Flag | Effect |
|---|---|
| `--only 12.mp4 15.mp4` | open just these videos |
| `--redo` | reopen videos already marked done |
| `--events <file>` | read and write a different events file (used for team work) |

### 3.2 Typing times into a CSV (e.g. while watching in VLC)

```
video,time,type,note
12.mp4,1:03.4,collision,bus cut in from left
12.mp4,0:00-0:06,ignore,
15.mp4,,none,
```

```powershell
python -m tools.mark_collisions --config configs/config.yaml --import-csv my_times.csv
```

Importing a video **replaces** any marks it already had. VLC's time display is
coarse, so use the player when you can.

### 3.3 Checking and exporting

```powershell
python -m tools.mark_collisions --config configs/config.yaml --list
```
```powershell
python -m tools.mark_collisions --config configs/config.yaml --export-csv collision_times.csv
```

---

## 4. Team workflow (3 annotators, 3 locations)

### 4.1 The idea in one paragraph

**Each person writes only their own events file.** No one ever edits a shared
file, so there are no merge conflicts across locations. Most videos are marked
by one person. A small **overlap set is marked by all three without talking
about it**. That overlap gives us an inter-annotator agreement number for the
thesis and shows where our definitions are unclear. A short **calibration
round** comes first so we agree on the rules before the real work starts.

### 4.2 The split (168 videos: 104 train / 32 val / 32 test)

| Set | Videos | Marked by | Purpose |
|---|---|---|---|
| Calibration | 10 | all 3, **then discussed together** | agree on the rules in section 2; these marks are not used for agreement numbers |
| Agreement (IAA) | 20 | all 3, **blind** (no discussing until everyone is finished) | measure agreement for the thesis; adjudicated afterwards |
| Individual | 138 → 46 each | 1 person | the bulk of the labels |

That's about **76 videos per person**. Spread both shared sets across
train/val/test and across event / no-event videos. Don't hand one person all of
the test set.

### 4.3 Setup (once per person)

1. **Use the same video files.** Everyone needs the identical `data/raw_videos`
   folder (same names, same train/val/test subfolders). Re-encoded or trimmed
   copies shift all the times. Check by comparing hash lists:

   ```powershell
   Get-ChildItem data/raw_videos -Recurse -File | Get-FileHash -Algorithm MD5 | Select-Object Hash, @{n='File';e={$_.Name}} | Sort-Object File | Export-Csv video_hashes_<name>.csv -NoTypeInformation
   ```

   Send the CSV to each other. All three files should be identical.

2. **Create the team folder and your assignment list.** One person makes three
   text files (one video file name per line) and shares them:

   ```
   data/annotations/team/assign_<name>.txt     # calibration + agreement + your 46
   ```

3. **Agree on your name tag** (e.g. `arnob`, `rafi`, `nabil`) and use it in every file name.

### 4.4 Marking (each person, any number of sessions)

```powershell
python -m tools.mark_collisions --config configs/config.yaml --videos data/raw_videos --events data/annotations/team/collision_events_<name>.json --only (Get-Content data/annotations/team/assign_<name>.txt)
```

- The tool skips videos you already marked done, so you can stop with `q` and run
  the same command later to continue.
- Check your progress:
  ```powershell
  python -m tools.mark_collisions --config configs/config.yaml --events data/annotations/team/collision_events_<name>.json --list
  ```
- **Doubt sheet.** Keep one shared spreadsheet (Google Sheet) with columns
  `annotator, video, time, question`. Log every unclear case there instead of
  messaging about it, and **don't discuss agreement-set videos until all three are finished**.

### 4.5 Sharing results

`data/annotations/` is git-ignored, so the files don't sync on their own. Pick one:

- **Shared drive folder (simplest):** everyone uploads
  `collision_events_<name>.json` to the same Google Drive / OneDrive folder.
- **Git:** commit only your own file, with `git add -f data/annotations/team/collision_events_<name>.json`.
  Since each person touches only their own file, pulls never conflict.

Upload after every session, so nothing is lost if a laptop dies.

### 4.6 Measuring agreement (for the thesis)

On the 20 agreement videos, report:

| Measure | How | Target |
|---|---|---|
| Event presence | Did each annotator mark an event in the video? Fleiss' κ across the 3 | κ ≥ 0.8 |
| Event type | For events found by all 3: % with the same type | ≥ 85 % |
| Timing | For matched events (within 1 s of each other): mean absolute difference from the median time; % within 0.2 s and within 0.5 s | mean < 0.3 s |

If timing is worse than the target, the 1.5 s bins aren't reliable. Tighten
the definitions, then re-mark the calibration set.

### 4.7 Merging into the final file

The pipeline reads one file, `data/annotations/collision_events.json`. Build it like this:

1. **Individual videos:** take the marks of the person who annotated them.
2. **Calibration + agreement videos:**
   - All three agree there is an event (and on its type) → use the **median** of the three times.
   - Ignore ranges → use their union.
   - Any disagreement (presence, type, or times more than 0.5 s apart) → decide together in a
     short meeting, and log the decision in the doubt sheet.
3. Run `--list` on the merged file and check that all 168 videos show `done`.

Then build the sequences:

```powershell
python -m tools.make_event_sequences --config configs/config.yaml
```

> Steps 4.2 (making the assignment lists), 4.6 and 4.7 are mechanical. Write a
> small script for them (e.g. `tools/team_annotations.py` with `assign`,
> `agreement` and `merge` commands) rather than doing them by hand.

### 4.8 Timeline (suggested)

| Day | Step |
|---|---|
| 1 | Hash check, assignment lists, everyone marks the 10 calibration videos |
| 2 | 1-hour call: compare calibration marks, finalise section 2 |
| 3 – 7 | Blind agreement set + individual videos |
| 8 | Agreement numbers, adjudication meeting, merge, run `make_event_sequences` |
