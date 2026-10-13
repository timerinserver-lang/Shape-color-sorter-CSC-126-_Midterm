"""
evaluate.py
-----------
Measures the detector's accuracy against samples/ground_truth.json.

For each image, every true object is matched to the nearest detection
(within 40 px of its center). We then report:

  Recall     - true objects that were found at all
  Precision  - detections that matched a real object (no false alarms)
  Color acc. - matched objects whose color label is correct
  Shape acc. - matched objects whose shape label is correct
  Both       - matched objects with color AND shape correct

Run:  python evaluate.py
"""

import json
import os
from collections import Counter

import cv2

from shape_detector import PROCESS_WIDTH, process_frame

SAMPLES = "samples"
MATCH_RADIUS = 40  # pixels, in the resized (800 px wide) frame


def evaluate(prefix=None):
    with open(os.path.join(SAMPLES, "ground_truth.json")) as f:
        truth = json.load(f)

    stats = Counter()
    confusions = Counter()
    for name, objs in truth.items():
        if prefix and not name.startswith(prefix):
            continue
        img = cv2.imread(os.path.join(SAMPLES, name))
        scale = PROCESS_WIDTH / img.shape[1]
        _, dets, _, _, _ = process_frame(img)
        used = set()

        for o in objs:
            tx, ty = o["center"][0] * scale, o["center"][1] * scale
            best, best_d = None, MATCH_RADIUS
            for i, d in enumerate(dets):
                if i in used:
                    continue
                dist = ((d["center"][0] - tx) ** 2 + (d["center"][1] - ty) ** 2) ** 0.5
                if dist < best_d:
                    best, best_d = i, dist
            stats["truth"] += 1
            if best is None:
                confusions[f'missed {o["color"]} {o["shape"]}'] += 1
                continue
            used.add(best)
            d = dets[best]
            stats["matched"] += 1
            stats["color_ok"] += d["color"] == o["color"]
            stats["shape_ok"] += d["shape"] == o["shape"]
            stats["both_ok"] += d["color"] == o["color"] and d["shape"] == o["shape"]
            if d["shape"] != o["shape"]:
                confusions[f'{o["shape"]} -> {d["shape"]}'] += 1
            if d["color"] != o["color"]:
                confusions[f'{o["color"]} -> {d["color"]}'] += 1
        stats["detections"] += len(dets)
        stats["false_pos"] += len(dets) - len(used)
        stats["images"] += 1

    m = max(stats["matched"], 1)
    return {
        "images": stats["images"],
        "objects": stats["truth"],
        "recall": stats["matched"] / max(stats["truth"], 1),
        "precision": stats["matched"] / max(stats["detections"], 1),
        "color_acc": stats["color_ok"] / m,
        "shape_acc": stats["shape_ok"] / m,
        "both_acc": stats["both_ok"] / m,
        "false_pos": stats["false_pos"],
        "errors": dict(confusions.most_common()),
    }


def main():
    for label, prefix in (("Clean images", "clean_"), ("Hard images", "hard_"),
                          ("Stress images", "stress_"), ("All images", None)):
        r = evaluate(prefix)
        print(f"\n{label}: {r['images']} images, {r['objects']} objects")
        print(f"  Recall     {r['recall']:.1%}")
        print(f"  Precision  {r['precision']:.1%}  (false positives: {r['false_pos']})")
        print(f"  Color acc. {r['color_acc']:.1%}")
        print(f"  Shape acc. {r['shape_acc']:.1%}")
        print(f"  Both right {r['both_acc']:.1%}")
        if r["errors"]:
            print("  Errors:", ", ".join(f"{k} x{v}" for k, v in r["errors"].items()))


if __name__ == "__main__":
    main()
