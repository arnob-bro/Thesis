"""Step 1 - check the ODS_New dataset and split it into train / val / test.

    python -m tools.detector.prepare_ods

Neighbouring images (1.jpg, 2.jpg, ...) are frames of the same video clip, so
images are split in blocks of consecutive numbers, never one by one. Each
weather subset (rainy, foggy) is split on its own so every split contains
every condition. The best of many random block shuffles is then improved by
swapping blocks between splits until every class is close to its target share.

Writes data/detector/ods/ (paths in configs/detector_ods.yaml):
    train|val|test/images, train|val|test/labels   hard links to the originals
                         (copies for the few broken JPEGs and fixed labels)
    data.yaml            the file Ultralytics trains on
    split.csv            image -> split, subset, condition, block
    split_report.md      class counts per split
The original dataset folder is never modified.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from utils.config import load_config

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NAME_RE = re.compile(r"^(.*?)[\s_]*\(?(\d+)\)?$")
MARKER = ".ods_split"
SPLITS = ("train", "val", "test")


def parse_name(stem: str) -> tuple[str, int]:
    """'rainy (12)' -> ('rainy', 12); '123' -> ('main', 123)."""
    m = NAME_RE.match(stem)
    if not m:
        return stem, 0
    return (m.group(1).strip() or "main"), int(m.group(2))


def condition_of(subset: str, rules: dict) -> str:
    low = subset.lower()
    for word, cond in rules.items():
        if word.lower() in low:
            return cond
    return "normal"


def read_label(path: Path, nc: int) -> tuple[list[str], list[int], list[str]]:
    """Return (clean lines, class ids, problems). Slightly out-of-range boxes are clipped."""
    lines, classes, problems = [], [], []
    if not path.exists():
        return lines, classes, ["no label file (treated as background)"]
    for i, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        parts = raw.split()
        if not parts:
            continue
        if len(parts) != 5:
            problems.append(f"line {i}: {len(parts)} values (segmentation or broken), dropped")
            continue
        try:
            c = int(float(parts[0]))
            x, y, w, h = (float(v) for v in parts[1:])
        except ValueError:
            problems.append(f"line {i}: not numeric, dropped")
            continue
        if not 0 <= c < nc:
            problems.append(f"line {i}: class {c} outside 0..{nc - 1}, dropped")
            continue
        x1, y1, x2, y2 = (max(0.0, min(1.0, v)) for v in (x - w / 2, y - h / 2, x + w / 2, y + h / 2))
        if x2 - x1 <= 1e-4 or y2 - y1 <= 1e-4:
            problems.append(f"line {i}: zero-size box, dropped")
            continue
        if max(-(x - w / 2), -(y - h / 2), x + w / 2 - 1, y + h / 2 - 1) > 0.01:
            problems.append(f"line {i}: box reaches outside the image, clipped")
        lines.append(f"{c} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")  # tiny overshoots are clipped silently
        classes.append(c)
    return lines, classes, problems


def allocate(blocks: list[dict], fractions: dict, rng: random.Random) -> dict[int, str]:
    """Assign blocks of each subset to splits, filling the split furthest below its target."""
    by_subset = defaultdict(list)
    for b in blocks:
        by_subset[b["subset"]].append(b)
    out = {}
    for subset_blocks in by_subset.values():
        order = subset_blocks[:]
        rng.shuffle(order)
        total = sum(b["n"] for b in order)
        filled = dict.fromkeys(SPLITS, 0)
        for b in order:
            s = max(SPLITS, key=lambda k: (fractions[k] * total - filled[k], fractions[k]))
            out[b["id"]] = s
            filled[s] += b["n"]
    return out


def class_shares(blocks: list[dict], assign: dict, nc: int) -> dict[str, np.ndarray]:
    counts = {s: np.zeros(nc) for s in SPLITS}
    for b in blocks:
        counts[assign[b["id"]]] += b["counts"]
    total = sum(counts.values())
    return {s: counts[s] / np.maximum(total, 1) for s in SPLITS}


def split_score(blocks: list[dict], assign: dict, fractions: dict, nc: int) -> float:
    """How far each class (and the image count) is from the target share, worst cases weighted most."""
    shares = class_shares(blocks, assign, nc)
    n = {k: 0 for k in SPLITS}
    for b in blocks:
        n[assign[b["id"]]] += b["n"]
    total = sum(n.values())
    dev = [((shares[k] - fractions[k]) / fractions[k]) ** 2 for k in SPLITS]
    img = sum(((n[k] / total - fractions[k]) / fractions[k]) ** 2 for k in SPLITS)
    return float(np.mean(dev) + np.max(dev) + 4 * img)


def refine(blocks: list[dict], assign: dict, fractions: dict, nc: int, rng: random.Random,
           rounds: int = 20000) -> dict:
    """Hill-climb: move or swap blocks of the same subset between splits while it improves the balance."""
    assign = dict(assign)
    score = split_score(blocks, assign, fractions, nc)
    by_subset = defaultdict(list)
    for b in blocks:
        by_subset[b["subset"]].append(b["id"])
    subsets = [s for s, ids in by_subset.items() if len(ids) > 1]
    for _ in range(rounds):
        ids = by_subset[rng.choice(subsets)]
        a = rng.choice(ids)
        trial = dict(assign)
        if rng.random() < 0.5:                       # swap two blocks in different splits
            b = rng.choice(ids)
            if assign[a] == assign[b]:
                continue
            trial[a], trial[b] = assign[b], assign[a]
        else:                                         # move one block
            trial[a] = rng.choice([k for k in SPLITS if k != assign[a]])
        new = split_score(blocks, trial, fractions, nc)
        if new < score:
            assign, score = trial, new
    return assign


def jpeg_incomplete(path: Path) -> bool:
    """Ultralytics re-saves JPEGs that do not end with the end-of-image marker, in place."""
    if path.suffix.lower() not in (".jpg", ".jpeg"):
        return False
    with path.open("rb") as fp:
        fp.seek(-2, 2)
        return fp.read() != bytes([0xFF, 0xD9])


def link(src: Path, dst: Path) -> None:
    if jpeg_incomplete(src):  # copy, so a repair by Ultralytics never touches the original
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:          # different drive or no hard-link support: copy instead
        shutil.copy2(src, dst)


def main() -> None:
    p = argparse.ArgumentParser(description="Split ODS_New into train/val/test")
    p.add_argument("--config", default="configs/detector_ods.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg.dataset.root).resolve()
    out = Path(cfg.split.out).resolve()
    fractions = {k: float(v) for k, v in cfg.split.fractions.items()}
    if abs(sum(fractions.values()) - 1) > 1e-6 or set(fractions) != set(SPLITS):
        raise SystemExit("split.fractions must have train, val, test summing to 1")

    names = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    names = [str(n).strip() for n in names]
    nc = len(names)

    # ---- read every image + label -------------------------------------------------
    images = sorted(f for f in (root / "images").iterdir() if f.suffix.lower() in IMG_EXT)
    if not images:
        raise SystemExit(f"No images in {root / 'images'}")
    items, problems = [], []
    for f in images:
        lines, classes, probs = read_label(root / "labels" / f"{f.stem}.txt", nc)
        problems += [f"{f.name}: {m}" for m in probs]
        subset, num = parse_name(f.stem)
        items.append({"file": f, "subset": subset, "num": num, "lines": lines, "fixed": bool(probs),
                      "counts": np.bincount(classes, minlength=nc),
                      "condition": condition_of(subset, dict(cfg.dataset.conditions))})
    print(f"{len(items)} images, {sum(int(i['counts'].sum()) for i in items)} boxes, {nc} classes")
    if problems:
        print(f"{len(problems)} label problems (fixed in the split copy, originals untouched), e.g.:")
        for m in problems[:5]:
            print("   ", m)

    # ---- blocks of consecutive images per subset --------------------------------------
    bs = int(cfg.split.block_size)
    blocks = []
    by_subset = defaultdict(list)
    for it in items:
        by_subset[it["subset"]].append(it)
    for subset, its in sorted(by_subset.items()):
        its.sort(key=lambda i: i["num"])
        for k in range(0, len(its), bs):
            members = its[k:k + bs]
            bid = len(blocks)
            for m in members:
                m["block"] = bid
            blocks.append({"id": bid, "subset": subset, "n": len(members),
                           "counts": sum(m["counts"] for m in members)})
    print("subsets:", ", ".join(f"{s} ({len(v)} images, {condition_of(s, dict(cfg.dataset.conditions))})"
                                for s, v in sorted(by_subset.items())))

    # ---- pick the most class-balanced split -------------------------------------------
    best, best_score, best_seed = None, np.inf, 0
    for seed in range(int(cfg.split.seed_search)):
        assign = allocate(blocks, fractions, random.Random(seed))
        score = split_score(blocks, assign, fractions, nc)
        if score < best_score:
            best, best_score, best_seed = assign, score, seed
    best = refine(blocks, best, fractions, nc, random.Random(best_seed))
    shares = class_shares(blocks, best, nc)
    worst = max(np.abs(shares[s] - fractions[s]).max() / fractions[s] for s in SPLITS)
    print(f"chosen split: seed {best_seed} + block swaps; worst class is {worst:.0%} away from its target share")

    # ---- write ---------------------------------------------------------------------
    if out.exists():
        if not (out / MARKER).exists():
            raise SystemExit(f"{out} exists and was not made by this script; remove it or change split.out")
        shutil.rmtree(out)
    for s in SPLITS:
        (out / s / "images").mkdir(parents=True)
        (out / s / "labels").mkdir(parents=True)
    (out / MARKER).write_text("made by tools/detector/prepare_ods.py\n", encoding="utf-8")

    counts = {s: np.zeros(nc, int) for s in SPLITS}
    n_img = {s: defaultdict(int) for s in SPLITS}
    with (out / "split.csv").open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["image", "split", "subset", "condition", "block", "boxes"])
        for it in items:
            s = best[it["block"]]
            f = it["file"]
            link(f, out / s / "images" / f.name)
            src_label = root / "labels" / f"{f.stem}.txt"
            dst_label = out / s / "labels" / f"{f.stem}.txt"
            if it["fixed"]:      # write a cleaned copy; the original stays as it is
                dst_label.write_text("\n".join(it["lines"]) + ("\n" if it["lines"] else ""), encoding="utf-8")
            else:
                link(src_label, dst_label)
            counts[s] += it["counts"]
            n_img[s][it["condition"]] += 1
            n_img[s]["all"] += 1
            w.writerow([f.name, s, it["subset"], it["condition"], it["block"], int(it["counts"].sum())])

    data = {"path": out.as_posix(), "train": "train/images", "val": "val/images",
            "test": "test/images", "nc": nc, "names": names}
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    # ---- report ------------------------------------------------------------------------
    conds = sorted({it["condition"] for it in items})
    rep = ["# ODS_New split", "",
           f"Blocks of {bs} consecutive images per subset; shuffle seed {best_seed}, then balanced by block swaps.", "",
           "| | " + " | ".join(SPLITS) + " |", "|---|" + "---|" * len(SPLITS)]
    for c in ["all"] + conds:
        rep.append(f"| images ({c}) | " + " | ".join(str(n_img[s][c]) for s in SPLITS) + " |")
    rep += ["", "| class | " + " | ".join(SPLITS) + " | test share |", "|---|" + "---|" * (len(SPLITS) + 1)]
    for i, n in enumerate(names):
        tot = sum(counts[s][i] for s in SPLITS)
        rep.append(f"| {n} | " + " | ".join(str(counts[s][i]) for s in SPLITS)
                   + f" | {counts['test'][i] / max(tot, 1):.0%} |")
    if problems:
        rep += ["", f"## Label problems ({len(problems)})", ""] + [f"- {m}" for m in problems[:200]]
    (out / "split_report.md").write_text("\n".join(rep) + "\n", encoding="utf-8")

    print(f"\n{'class':<13}" + "".join(f"{s:>8}" for s in SPLITS))
    for i, n in enumerate(names):
        print(f"{n:<13}" + "".join(f"{counts[s][i]:>8}" for s in SPLITS))
    print(f"{'images':<13}" + "".join(f"{n_img[s]['all']:>8}" for s in SPLITS))
    print(f"\nWrote {out / 'data.yaml'} and split_report.md. Next: python -m tools.detector.train_ods")


if __name__ == "__main__":
    main()
