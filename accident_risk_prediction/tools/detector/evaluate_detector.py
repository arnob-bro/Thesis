"""Step 3 - evaluate a trained detector on the ODS_New test split.

    python -m tools.detector.evaluate_detector                       # newest run in train.project/train.name
    python -m tools.detector.evaluate_detector --weights runs/detector/ods_yolo26m/weights/best.pt
    python -m tools.detector.evaluate_detector --split val --no-coco

Writes <run>/eval_<split>/:
    report.md        all tables below, ready for the thesis
    metrics.json     the same numbers, machine-readable
    per_class.csv    standard per-class table
    ultralytics/     confusion matrix, PR / F1 curves, example predictions

Tables:
  1. Standard COCO-style metrics per class (Ultralytics): precision, recall, mAP50, mAP50-95.
  2. The 10 classes the risk pipeline uses (fine classes grouped, e.g. Car + MPV = car).
  3. Per weather condition (normal / rainy / foggy).
  4. Per object size: small objects are the far-away road users.
  5. Against the COCO YOLO26x the pipeline uses now: AP on the 6 shared classes, and
     how many Bangladeshi vehicles each model finds at all (any class).
  6. Speed, and the lines to paste into configs/config.yaml.
Tables 2-5 use our own matching (IoU >= 0.5, COCO 101-point AP), so they are
directly comparable with each other; table 1 is the standard Ultralytics output.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from utils.config import Config, load_config

SIZES = (("small", 0, 32 ** 2), ("medium", 32 ** 2, 96 ** 2), ("large", 96 ** 2, np.inf))


# --------------------------------------------------------------------- data
def load_split(cfg: Config, split: str, limit: int | None = None) -> tuple[list[str], list[dict]]:
    root = Path(cfg.split.out).resolve()
    names = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))["names"]
    cond = {}
    with (root / "split.csv").open(encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            cond[row["image"]] = row["condition"]
    files = sorted((root / split / "images").iterdir())
    if limit:
        files = files[:: max(1, len(files) // limit)][:limit]
    recs = []
    for f in files:
        cls, boxes = [], []
        lab = root / split / "labels" / f"{f.stem}.txt"
        if lab.exists():
            for line in lab.read_text(encoding="utf-8").splitlines():
                v = line.split()
                if len(v) == 5:
                    c, x, y, w, h = int(float(v[0])), *map(float, v[1:])
                    cls.append(c)
                    boxes.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
        recs.append({"file": f, "cond": cond.get(f.name, "normal"),
                     "gc": np.array(cls, int), "gb": np.array(boxes, float).reshape(-1, 4)})
    return list(names), recs


def predict(model: YOLO, recs: list[dict], imgsz: int, key: str, classes: list[int] | None = None,
            class_lut: np.ndarray | None = None) -> None:
    """Store low-threshold predictions (normalised xyxy) in rec[key]; class_lut maps model ids."""
    for k in range(0, len(recs), 8):          # own batches: results come back in the same order
        chunk = recs[k:k + 8]
        imgs = [cv2.imdecode(np.fromfile(str(r["file"]), np.uint8), cv2.IMREAD_COLOR) for r in chunk]
        for r, res in zip(chunk, model.predict(imgs, conf=0.001, imgsz=imgsz, batch=8, max_det=300,
                                               classes=classes, verbose=False)):
            b = res.boxes
            pc = b.cls.cpu().numpy().astype(int)
            if class_lut is not None:
                pc = class_lut[pc]
            keep = pc >= 0
            r[key] = (pc[keep], b.conf.cpu().numpy()[keep], b.xyxyn.cpu().numpy()[keep])
            h, w = res.orig_shape
            r["area"] = (r["gb"][:, 2] - r["gb"][:, 0]) * w * (r["gb"][:, 3] - r["gb"][:, 1]) * h


# ------------------------------------------------------------------ matching
def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def greedy(pb, pconf, gb, thr=0.5) -> tuple[np.ndarray, np.ndarray]:
    """Match predictions (highest confidence first) to ground truth. Returns (tp per pred, hit per gt)."""
    tp, hit = np.zeros(len(pb), bool), np.zeros(len(gb), bool)
    if len(pb) and len(gb):
        m = iou(pb, gb)
        for i in np.argsort(-pconf):
            cand = np.where(~hit, m[i], -1)
            j = int(cand.argmax())
            if cand[j] >= thr:
                tp[i], hit[j] = True, True
    return tp, hit


def remap(rec: dict, key: str, lut: np.ndarray | None):
    gc, (pc, pconf, pb) = rec["gc"], rec[key]
    gb = rec["gb"]
    if lut is None:
        return gc, gb, pc, pconf, pb
    gc, pc = lut[gc], lut[pc]
    g, p = gc >= 0, pc >= 0
    return gc[g], gb[g], pc[p], pconf[p], pb[p]


def coco_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """COCO 101-point AP: mean over recall levels 0, 0.01 .. 1 of the best precision at that recall or above."""
    if len(recall) == 0:
        return 0.0
    env = np.maximum.accumulate(precision[::-1])[::-1]
    idx = np.searchsorted(recall, np.linspace(0, 1, 101), side="left")
    return float(np.where(idx < len(recall), env[np.minimum(idx, len(recall) - 1)], 0.0).mean())


def ap50(recs: list[dict], key: str, nc: int, lut: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-class AP at IoU 0.5 (COCO 101-point interpolation) and number of GT boxes."""
    confs, tps, ngt = defaultdict(list), defaultdict(list), np.zeros(nc, int)
    for r in recs:
        gc, gb, pc, pconf, pb = remap(r, key, lut)
        for c in np.union1d(gc, pc).astype(int):
            g, p = gc == c, pc == c
            ngt[c] += g.sum()
            tp, _ = greedy(pb[p], pconf[p], gb[g])
            confs[c].append(pconf[p])
            tps[c].append(tp)
    ap = np.full(nc, np.nan)
    for c in range(nc):
        if ngt[c] == 0:
            continue
        conf = np.concatenate(confs[c]) if confs[c] else np.zeros(0)
        tp = np.concatenate(tps[c]) if tps[c] else np.zeros(0, bool)
        order = np.argsort(-conf, kind="stable")
        ctp, cfp = np.cumsum(tp[order]), np.cumsum(~tp[order])
        ap[c] = coco_ap(ctp / ngt[c], ctp / np.maximum(ctp + cfp, 1))
    return ap, ngt


def at_threshold(recs: list[dict], key: str, conf: float, lut: np.ndarray | None = None) -> dict:
    """Per ground-truth box: found with the right class / found as any class, at the operating threshold."""
    rows = {"cls": [], "area": [], "cond": [], "hit": [], "hit_any": []}
    n_pred = tp_total = 0
    for r in recs:
        gc, gb, pc, pconf, pb = remap(r, key, lut)
        area = r["area"] if lut is None else r["area"][lut[r["gc"]] >= 0]
        keep = pconf >= conf
        pc, pconf, pb = pc[keep], pconf[keep], pb[keep]
        n_pred += len(pc)
        hit = np.zeros(len(gc), bool)
        for c in np.unique(gc):
            g, p = gc == c, pc == c
            tp, h = greedy(pb[p], pconf[p], gb[g])
            hit[g] = h
            tp_total += tp.sum()
        _, hit_any = greedy(pb, pconf, gb)
        rows["cls"].append(gc), rows["area"].append(area), rows["hit"].append(hit), rows["hit_any"].append(hit_any)
        rows["cond"].append(np.full(len(gc), r["cond"], object))
    out = {k: np.concatenate(v) if v else np.zeros(0) for k, v in rows.items()}
    out["precision"] = tp_total / max(n_pred, 1)
    return out


# -------------------------------------------------------------------- report
def fmt(v, pct=False) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{v:.1%}" if pct else f"{v:.3f}"


def table(header: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + \
           ["| " + " | ".join(str(c) for c in r) + " |" for r in rows] + [""]


def latest_best(cfg: Config) -> Path:
    best = Path(cfg.train.project).resolve() / cfg.train.name / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"{best} not found. Train first or pass --weights.")
    return best


def evaluate(cfg: Config, weights: Path, split: str = "test", coco: str | None = "config",
             quick: bool = False) -> dict:
    weights = Path(weights).resolve()
    out = weights.parent.parent / f"eval_{split}"
    out.mkdir(parents=True, exist_ok=True)
    data = Path(cfg.split.out).resolve() / "data.yaml"
    imgsz = int(cfg.train.args.get("imgsz", 640))
    op = float(cfg.eval.conf)
    names, recs = load_split(cfg, split, limit=300 if quick else None)
    nc = len(names)
    report = {"weights": str(weights), "split": split, "images": len(recs), "conf": op}
    print(f"Evaluating {weights} on {len(recs)} {split} images")

    # 1. Standard Ultralytics metrics --------------------------------------------------
    model = YOLO(str(weights))
    m = model.val(data=str(data), split=split, imgsz=imgsz, batch=16, workers=2, plots=True,
                  verbose=False, project=str(out), name="ultralytics", exist_ok=True)
    ngt_all = np.bincount(np.concatenate([r["gc"] for r in recs]), minlength=nc) if not quick else None
    std = []
    for i, c in enumerate(m.box.ap_class_index):
        std.append({"class": names[int(c)], "boxes": int(ngt_all[int(c)]) if ngt_all is not None else None,
                    "precision": float(m.box.p[i]), "recall": float(m.box.r[i]),
                    "mAP50": float(m.box.ap50[i]), "mAP50_95": float(m.box.ap[i])})
    report["standard"] = {"per_class": std, "precision": float(m.box.mp), "recall": float(m.box.mr),
                          "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map)}
    report["speed_ms"] = {k: float(v) for k, v in m.speed.items()}
    with (out / "per_class.csv").open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(std[0]))
        w.writeheader()
        w.writerows(std)

    # predictions for our own breakdowns
    predict(model, recs, imgsz, "ours")
    ap_fine, ngt = ap50(recs, "ours", nc)
    thr = at_threshold(recs, "ours", op)

    # 2. Pipeline classes -----------------------------------------------------------------
    pipe = list(cfg.pipeline.classes)
    pmap = dict(cfg.pipeline.map)
    lut = np.array([pipe.index(pmap[n]) if n in pmap else -1 for n in names])
    ap_pipe, ngt_pipe = ap50(recs, "ours", len(pipe), lut)
    thr_pipe = at_threshold(recs, "ours", op, lut)
    report["pipeline"] = [{"class": pipe[c], "members": [n for n in names if pmap.get(n) == pipe[c]],
                           "boxes": int(ngt_pipe[c]), "AP50": float(ap_pipe[c]),
                           "recall": float(thr_pipe["hit"][thr_pipe["cls"] == c].mean()) if ngt_pipe[c] else None}
                          for c in range(len(pipe))]
    report["pipeline_mAP50"] = float(np.nanmean(ap_pipe))

    # 3. Conditions --------------------------------------------------------------------------
    report["conditions"] = []
    for cond in sorted({r["cond"] for r in recs}):
        sub = [r for r in recs if r["cond"] == cond]
        a, _ = ap50(sub, "ours", nc)
        t = at_threshold(sub, "ours", op)
        report["conditions"].append({"condition": cond, "images": len(sub), "boxes": int(len(t["cls"])),
                                     "mAP50": float(np.nanmean(a)), "recall": float(t["hit"].mean()),
                                     "precision": float(t["precision"])})

    # 4. Object size ---------------------------------------------------------------------
    report["sizes"] = []
    for label, lo, hi in SIZES:
        s = (thr["area"] >= lo) & (thr["area"] < hi)
        report["sizes"].append({"size": label, "boxes": int(s.sum()),
                                "recall": float(thr["hit"][s].mean()) if s.any() else None,
                                "recall_any_class": float(thr["hit_any"][s].mean()) if s.any() else None})
    report["overall_at_conf"] = {"precision": float(thr["precision"]), "recall": float(thr["hit"].mean()),
                                 "recall_any_class": float(thr["hit_any"].mean())}

    # 5. COCO baseline ------------------------------------------------------------------
    coco = cfg.eval.get("coco_baseline") if coco == "config" else coco
    if coco and Path(coco).exists():
        print(f"Comparing with COCO baseline {coco}")
        c2o = {int(k): names.index(v) for k, v in dict(cfg.eval.coco_to_ods).items()}
        clut = np.full(100, -1)
        for k, v in c2o.items():
            clut[k] = v
        predict(YOLO(str(coco)), recs, imgsz, "coco", classes=sorted(c2o), class_lut=clut)
        ap_coco, _ = ap50(recs, "coco", nc)
        thr_coco = at_threshold(recs, "coco", op)
        shared = sorted(set(c2o.values()))
        report["coco"] = {
            "weights": str(coco),
            "shared_AP50": [{"class": names[c], "coco": float(ap_coco[c]), "ours": float(ap_fine[c])} for c in shared],
            "found_any_class": [{"class": names[c], "boxes": int(ngt[c]),
                                 "coco": float(thr_coco["hit_any"][thr_coco["cls"] == c].mean()),
                                 "ours": float(thr["hit_any"][thr["cls"] == c].mean())}
                                for c in range(nc) if ngt[c]],
        }
    elif coco:
        print(f"COCO baseline {coco} not found, skipping the comparison")

    report["fine_AP50_own_matching"] = {names[c]: float(ap_fine[c]) for c in range(nc) if ngt[c]}
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_report(report, cfg, names, out / "report.md", imgsz)
    print(f"\nReport: {out / 'report.md'}")
    return report


def write_report(rep: dict, cfg: Config, names: list[str], path: Path, imgsz: int) -> None:
    op = rep["conf"]
    s = rep["standard"]
    L = [f"# Detector evaluation: {Path(rep['weights']).parent.parent.name}", "",
         f"Weights `{rep['weights']}`, {rep['split']} split, {rep['images']} images, image size {imgsz}.", "",
         "## 1. Standard metrics (Ultralytics)", ""]
    L += table(["class", "boxes", "precision", "recall", "mAP50", "mAP50-95"],
               [[r["class"], r["boxes"] if r["boxes"] is not None else "–", fmt(r["precision"]), fmt(r["recall"]),
                 fmt(r["mAP50"]), fmt(r["mAP50_95"])] for r in s["per_class"]]
               + [["**all**", "", fmt(s["precision"]), fmt(s["recall"]), f"**{fmt(s['mAP50'])}**",
                   f"**{fmt(s['mAP50_95'])}**"]])
    o = rep["overall_at_conf"]
    L += [f"At the pipeline's confidence threshold {op}: precision {fmt(o['precision'], True)}, "
          f"recall {fmt(o['recall'], True)}; {fmt(o['recall_any_class'], True)} of all objects get a box "
          "of some class (the risk graph still sees them).", "",
          "## 2. Classes used by the risk pipeline", "",
          "Fine classes grouped as in `pipeline.map`; confusions inside a group (e.g. Car vs MPV) no longer count.", ""]
    L += table(["pipeline class", "detector classes", "boxes", "AP50", f"recall @ {op}"],
               [[r["class"], ", ".join(r["members"]), r["boxes"], fmt(r["AP50"]), fmt(r["recall"], True)]
                for r in rep["pipeline"]] + [["**mean**", "", "", f"**{fmt(rep['pipeline_mAP50'])}**", ""]])
    L += ["## 3. Weather conditions", ""]
    L += table(["condition", "images", "boxes", "mAP50", f"precision @ {op}", f"recall @ {op}"],
               [[r["condition"], r["images"], r["boxes"], fmt(r["mAP50"]), fmt(r["precision"], True),
                 fmt(r["recall"], True)] for r in rep["conditions"]])
    L += ["## 4. Object size", "", "Area in original image pixels (COCO bins: small < 32², large > 96²).", ""]
    L += table(["size", "boxes", f"recall @ {op}", "found as any class"],
               [[r["size"], r["boxes"], fmt(r["recall"], True), fmt(r["recall_any_class"], True)] for r in rep["sizes"]])
    if "coco" in rep:
        c = rep["coco"]
        L += [f"## 5. Against the COCO model (`{Path(c['weights']).name}`)", "",
              "AP50 on the classes COCO also has. COCO calls MPVs and CNGs 'car' and riders 'person', which "
              "shows up here as false positives.", ""]
        L += table(["class", "COCO", "ours"], [[r["class"], fmt(r["coco"]), fmt(r["ours"])] for r in c["shared_AP50"]])
        L += [f"Share of objects that get a box of **any** class at confidence {op}. "
              "This is what the risk graph depends on: an object with no box is invisible to it.", ""]
        L += table(["class", "boxes", "COCO", "ours"],
                   [[r["class"], r["boxes"], fmt(r["coco"], True), fmt(r["ours"], True)] for r in c["found_any_class"]])
    sp = rep["speed_ms"]
    L += ["## 6. Speed", "",
          f"{sp.get('inference', 0):.1f} ms inference + {sp.get('preprocess', 0) + sp.get('postprocess', 0):.1f} ms "
          "pre/post-processing per image (batch 16, this GPU).", "",
          "## Using it in the risk pipeline", "",
          "Set this in `configs/config.yaml`, then rebuild detections and graphs:", "", "```yaml"]
    L += pipeline_snippet(cfg, names, rep["weights"], imgsz)
    L += ["```", "", "```powershell",
          "python -m pipelines.build_video_cache --config configs/config.yaml --videos data/raw_videos "
          "--steps detect track graphs --force", "```", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def pipeline_snippet(cfg: Config, names: list[str], weights: str, imgsz: int) -> list[str]:
    pipe = list(cfg.pipeline.classes)
    pmap = dict(cfg.pipeline.map)
    lines = ["yolo:", f"  weights: {Path(weights).as_posix()}", f"  imgsz: {imgsz}",
             "  agnostic_nms: true      # one box per object even if two fine classes compete",
             f"  classes: [{', '.join(pipe)}]", "  class_map:               # detector class id -> pipeline class"]
    lines += [f"    {i}: {pipe.index(pmap[n])}    # {n} -> {pmap[n]}" for i, n in enumerate(names) if n in pmap]
    lines += ["ego_filter:", "  enabled: true            # ODS has no 'own bike' class, keep the heuristic",
              f"  classes: [{', '.join(str(pipe.index(c)) for c in ('motorcycle', 'bicycle', 'pedestrian'))}]"]
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a detector on the ODS_New split")
    p.add_argument("--config", default="configs/detector_ods.yaml")
    p.add_argument("--weights", help="Default: <train.project>/<train.name>/weights/best.pt")
    p.add_argument("--split", help="test (default) or val")
    p.add_argument("--coco", help="COCO model to compare with (default: eval.coco_baseline)")
    p.add_argument("--no-coco", action="store_true")
    p.add_argument("--quick", action="store_true", help="Own breakdowns on 300 images only")
    args = p.parse_args()
    cfg = load_config(args.config)
    weights = Path(args.weights) if args.weights else latest_best(cfg)
    coco = None if args.no_coco else (args.coco or "config")
    evaluate(cfg, weights, split=args.split or cfg.eval.split, coco=coco, quick=args.quick)


if __name__ == "__main__":
    main()
