import json
from pathlib import Path

root = Path("data/sequences")

for f in root.rglob("ego_state.json"):
    try:
        with open(f, "r") as fp:
            json.load(fp)
    except Exception as e:
        print(f"\nBAD FILE: {f}")
        print(e)