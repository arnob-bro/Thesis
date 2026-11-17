import random
import json
from pathlib import Path
from collections import defaultdict

DATA_ROOT = Path("data/sequences")

TARGET_SPLIT = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

CLASS_MAP = {
    0: "safe",
    1: "slight",
    2: "moderate",
    3: "high"
}

random.seed(42)


# -----------------------------
# STEP 1: LOAD ALL DATA BY CLASS
# -----------------------------
def load_all():
    class_buckets = defaultdict(list)

    for split in ["train", "val", "test"]:
        split_path = DATA_ROOT / split

        for seq in split_path.iterdir():
            if not seq.is_dir():
                continue

            label_file = seq / "label.json"
            if not label_file.exists():
                continue

            with open(label_file) as f:
                data = json.load(f)

            label = data["risk_level"]
            class_buckets[label].append(seq)

    return class_buckets


# -----------------------------
# STEP 2: CLEAR OLD STRUCTURE (OPTIONAL SAFETY)
# -----------------------------
def clear_new_dirs():
    for split in TARGET_SPLIT:
        path = DATA_ROOT / f"new_{split}"
        path.mkdir(parents=True, exist_ok=True)


# -----------------------------
# STEP 3: SPLIT EACH CLASS
# -----------------------------
def split_class(samples, train_ratio, val_ratio):
    random.shuffle(samples)

    n = len(samples)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return (
        samples[:train_end],
        samples[train_end:val_end],
        samples[val_end:]
    )


# -----------------------------
# STEP 4: COPY DATA
# -----------------------------
def copy(seq_list, target_split):
    target_path = DATA_ROOT / f"new_{target_split}"
    target_path.mkdir(exist_ok=True)

    for seq in seq_list:
        dest = target_path / seq.name
        if dest.exists():
            continue

        # simple directory copy
        import shutil
        shutil.copytree(seq, dest)


# -----------------------------
# MAIN
# -----------------------------
def main():
    class_buckets = load_all()

    train_all, val_all, test_all = [], [], []

    print("\n📊 CLASS DISTRIBUTION AFTER RESPLIT\n")

    for cls, samples in class_buckets.items():
        train, val, test = split_class(
            samples,
            TARGET_SPLIT["train"],
            TARGET_SPLIT["val"]
        )

        print(f"{CLASS_MAP.get(cls, cls)}:")
        print(f"   train={len(train)} val={len(val)} test={len(test)}")

        train_all.extend(train)
        val_all.extend(val)
        test_all.extend(test)

    # copy files
    copy(train_all, "train")
    copy(val_all, "val")
    copy(test_all, "test")

    print("\n✅ Done! New balanced splits created in new_train / new_val / new_test")


if __name__ == "__main__":
    main()