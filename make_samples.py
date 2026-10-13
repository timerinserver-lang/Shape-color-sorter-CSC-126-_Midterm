"""
make_samples.py
---------------
Generates labelled test images and a short demo video in samples/.

Why synthetic samples? Each generated image comes with a ground-truth
file (samples/ground_truth.json) listing the true color, shape and
position of every object. evaluate.py compares the detector's output
against it to give an honest accuracy number for the documentation.

Two difficulty levels are generated:
  clean_XX.jpg  - flat background, even lighting
  hard_XX.jpg   - uneven lighting, drop shadows, sensor noise, blur and
                  JPEG compression, to imitate a phone photo of paper cutouts
  stress_XX.jpg - small, faded (low saturation) shapes under very uneven
                  lighting and heavy noise, to find where the method breaks

Run:  python make_samples.py
"""

import json
import math
import os

import cv2
import numpy as np

OUT_DIR = "samples"
W, H = 1200, 900
SHAPES = ["Triangle", "Square", "Rectangle", "Pentagon", "Circle"]

# Hue (OpenCV 0-179 scale) the generator paints each color with.
HUES = {"Red": [2, 176], "Orange": [14], "Yellow": [27], "Green": [60], "Blue": [108], "Purple": [145]}


def hsv_to_bgr(h, s, v):
    px = np.uint8([[[h, s, v]]])
    return tuple(int(c) for c in cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0])


def random_color(rng, name, faded=False):
    h = int(rng.choice(HUES[name])) + int(rng.integers(-2, 3))
    s = int(rng.integers(80, 170)) if faded else int(rng.integers(170, 245))
    v = int(rng.integers(110, 220)) if faded else int(rng.integers(160, 240))
    return hsv_to_bgr(h % 180, s, v)


def shape_points(shape, cx, cy, size, angle):
    """Corner points for a polygon shape, rotated by `angle` degrees."""
    if shape == "Triangle":
        pts = [(math.cos(math.radians(a)), math.sin(math.radians(a))) for a in (-90, 30, 150)]
        pts = [(x * size, y * size) for x, y in pts]
    elif shape == "Square":
        s = size * 0.8
        pts = [(-s, -s), (s, -s), (s, s), (-s, s)]
    elif shape == "Rectangle":
        pts = [(-size * 1.05, -size * 0.55), (size * 1.05, -size * 0.55),
               (size * 1.05, size * 0.55), (-size * 1.05, size * 0.55)]
    elif shape == "Pentagon":
        pts = [(math.cos(math.radians(-90 + 72 * i)) * size,
                math.sin(math.radians(-90 + 72 * i)) * size) for i in range(5)]
    else:
        return None
    a = math.radians(angle)
    rot = [(x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a)) for x, y in pts]
    return np.array([(int(cx + x), int(cy + y)) for x, y in rot], dtype=np.int32)


def draw_shape(img, shape, cx, cy, size, angle, color):
    if shape == "Circle":
        cv2.circle(img, (cx, cy), int(size * 0.85), color, -1, cv2.LINE_AA)
    else:
        cv2.fillPoly(img, [shape_points(shape, cx, cy, size, angle)], color, cv2.LINE_AA)


def make_image(rng, level):
    """Draw 8-12 shapes in a jittered grid so they never overlap.
    level: 0 = clean, 1 = hard (photo-like), 2 = stress (small/faded/dim)."""
    hard = level >= 1
    bg = np.full((H, W, 3), (236, 238, 240), dtype=np.uint8)
    cols, rows = 4, 3
    cw, ch = W // cols, H // rows
    cells = rng.permutation(cols * rows)[: int(rng.integers(8, 13))]
    objects = []

    shadow = np.zeros((H, W), dtype=np.uint8)
    for cell in cells:
        r, c = divmod(int(cell), cols)
        size = int(rng.integers(28, 60)) if level == 2 else int(rng.integers(60, 105))
        cx = c * cw + cw // 2 + int(rng.integers(-25, 26))
        cy = r * ch + ch // 2 + int(rng.integers(-15, 16))
        shape = str(rng.choice(SHAPES))
        color_name = str(rng.choice(list(HUES)))
        angle = float(rng.uniform(0, 360))
        objects.append({"shape": shape, "color": color_name, "center": [cx, cy],
                        "size": size, "angle": angle, "bgr": random_color(rng, color_name, faded=level == 2)})
        if hard:
            draw_shape(shadow, shape, cx + 9, cy + 11, size, angle, 255)

    img = bg.copy()
    if hard:
        # Soft grey drop shadow underneath each paper cutout.
        shadow = cv2.GaussianBlur(shadow, (31, 31), 0).astype(np.float32) / 255.0
        img = (img.astype(np.float32) * (1 - 0.35 * shadow[..., None])).astype(np.uint8)

    for o in objects:
        draw_shape(img, o["shape"], o["center"][0], o["center"][1], o["size"], o["angle"], o["bgr"])

    if hard:
        # Uneven lighting: bright in one corner, darker in the opposite one.
        dark = 0.5 if level == 2 else 0.72
        gx = np.linspace(1.08, dark, W)[None, :]
        gy = np.linspace(1.0, 0.85, H)[:, None]
        img = np.clip(img.astype(np.float32) * (gx * gy)[..., None], 0, 255)
        img += rng.normal(0, 12 if level == 2 else 7, img.shape)  # sensor noise
        img = np.clip(img, 0, 255).astype(np.uint8)
        k = 5 if level == 2 else 3
        img = cv2.GaussianBlur(img, (k, k), 0)                    # defocus
        _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)                 # JPEG artifacts

    truth = [{"shape": o["shape"], "color": o["color"], "center": o["center"]} for o in objects]
    return img, truth


def make_video(rng, path, seconds=12, fps=25):
    """Shapes moving left to right, like items on a conveyor belt."""
    vw, vh = 960, 540
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (vw, vh))
    items = []
    for i in range(14):
        items.append({
            "shape": str(rng.choice(SHAPES)), "color": str(rng.choice(list(HUES))),
            "x": -150 - i * 170, "y": int(rng.integers(110, vh - 110)),
            "size": int(rng.integers(50, 75)), "angle": float(rng.uniform(0, 360)),
            "spin": float(rng.uniform(-2, 2)),
        })
        items[-1]["bgr"] = random_color(rng, items[-1]["color"])

    for f in range(seconds * fps):
        frame = np.full((vh, vw, 3), (225, 228, 230), dtype=np.uint8)
        # Belt edges drawn in dark grey (low saturation, so never detected).
        cv2.rectangle(frame, (0, 60), (vw, 70), (90, 90, 90), -1)
        cv2.rectangle(frame, (0, vh - 70), (vw, vh - 60), (90, 90, 90), -1)
        for it in items:
            x = int(it["x"] + f * 6)
            if -120 < x < vw + 120:
                draw_shape(frame, it["shape"], x, it["y"], it["size"], it["angle"] + f * it["spin"], it["bgr"])
        frame = np.clip(frame + rng.normal(0, 4, frame.shape), 0, 255).astype(np.uint8)
        writer.write(frame)
    writer.release()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(126)  # fixed seed -> same samples every run
    truth = {}
    for kind, level, n in (("clean", 0, 10), ("hard", 1, 10), ("stress", 2, 10)):
        for i in range(1, n + 1):
            name = f"{kind}_{i:02d}.jpg"
            img, t = make_image(rng, level)
            cv2.imwrite(os.path.join(OUT_DIR, name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            truth[name] = t
    with open(os.path.join(OUT_DIR, "ground_truth.json"), "w") as f:
        json.dump(truth, f, indent=1)
    make_video(rng, os.path.join(OUT_DIR, "conveyor.mp4"))
    total = sum(len(v) for v in truth.values())
    print(f"Wrote {len(truth)} images ({total} objects), ground_truth.json and conveyor.mp4 to {OUT_DIR}/")


if __name__ == "__main__":
    main()
