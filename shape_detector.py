"""
shape_detector.py
-----------------
Core computer-vision logic for the Shape and Color Sorter.

The pipeline covers three topics from the CSC 126 laboratory:

  (a) Image Operations         -> resizing, Gaussian blur, color-space
                                  conversion, drawing boxes/labels/overlays
  (b) Image Segmentation       -> HSV color thresholding (cv2.inRange) and
                                  morphological opening/closing to clean masks
  (d) Image Analysis and       -> Canny edge detection, contour extraction,
      Transformation              polygon approximation, moments (centroid),
                                  minAreaRect (rotation-invariant aspect ratio)
                                  and circularity to classify each shape

Every function is small and does one job, so each step can be shown and
explained separately during the live demo.
"""

import math

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Width every frame is resized to. A fixed size keeps the area thresholds
# below meaningful no matter what camera or image resolution is used.
PROCESS_WIDTH = 800

# Contours smaller than this (in pixels, after resizing) are treated as noise.
MIN_AREA = 600

# HSV ranges for each color we sort.
# OpenCV stores Hue as 0-179 (not 0-359), Saturation and Value as 0-255.
# Red sits at both ends of the hue wheel, so it needs two ranges.
COLOR_RANGES = {
    "Red":    [((0, 90, 70), (8, 255, 255)), ((170, 90, 70), (179, 255, 255))],
    "Orange": [((9, 90, 70), (20, 255, 255))],
    "Yellow": [((21, 90, 70), (34, 255, 255))],
    "Green":  [((35, 70, 50), (85, 255, 255))],
    "Blue":   [((86, 90, 50), (128, 255, 255))],
    "Purple": [((129, 60, 50), (169, 255, 255))],
}

# BGR colors used when drawing each label (OpenCV uses BGR, not RGB).
DRAW_COLORS = {
    "Red": (40, 40, 220),
    "Orange": (0, 140, 255),
    "Yellow": (0, 200, 230),
    "Green": (60, 170, 60),
    "Blue": (220, 110, 30),
    "Purple": (170, 60, 150),
}


# ---------------------------------------------------------------------------
# (a) Image Operations: preprocessing
# ---------------------------------------------------------------------------

def resize_to_width(image, width=PROCESS_WIDTH):
    """Resize an image to a fixed width while keeping its aspect ratio."""
    h, w = image.shape[:2]
    if w == width:
        return image
    scale = width / float(w)
    return cv2.resize(image, (width, int(h * scale)), interpolation=cv2.INTER_AREA)


def preprocess(frame):
    """
    Prepare a frame for segmentation.

    1. Resize to a fixed width.
    2. Gaussian blur (5x5) to remove sensor noise and paper texture,
       which would otherwise create jagged contours.
    3. Convert BGR -> HSV. HSV separates color (Hue) from brightness
       (Value), so a shadow on a red shape is still "red".

    Returns the resized BGR frame and its blurred HSV version.
    """
    resized = resize_to_width(frame)
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    return resized, hsv


# ---------------------------------------------------------------------------
# (b) Image Segmentation: one binary mask per color
# ---------------------------------------------------------------------------

# 5x5 elliptical kernel for morphology. An ellipse avoids squaring off
# the edges of round shapes.
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def color_mask(hsv, ranges):
    """
    Build a clean binary mask for one color.

    cv2.inRange keeps pixels whose H, S and V all fall inside a range.
    Opening (erode then dilate) removes small specks; closing
    (dilate then erode) fills small holes inside the shape.
    """
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=2)
    return mask


def segment_colors(hsv):
    """Return a dict {color_name: mask} for every configured color."""
    return {name: color_mask(hsv, ranges) for name, ranges in COLOR_RANGES.items()}


# ---------------------------------------------------------------------------
# (d) Image Analysis and Transformation: classify each contour
# ---------------------------------------------------------------------------

def classify_shape(contour):
    """
    Decide what shape a contour is.

    - approxPolyDP simplifies the outline to its main corners. The
      tolerance is 4% of the perimeter, which ignores small bumps
      but keeps real corners.
    - 3 corners -> Triangle
    - 4 corners -> Square or Rectangle. We use minAreaRect, which fits a
      rotated rectangle, so a tilted square is still a square.
    - 5 corners -> Pentagon
    - Otherwise use circularity = 4*pi*Area / Perimeter^2.
      A perfect circle scores 1.0; anything above 0.80 is a Circle.
    """
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    corners = len(approx)

    if corners == 3:
        return "Triangle"

    if corners == 4:
        (_, _), (w, h), _ = cv2.minAreaRect(contour)
        aspect = max(w, h) / max(1.0, min(w, h))
        return "Square" if aspect <= 1.15 else "Rectangle"

    if corners == 5:
        return "Pentagon"

    area = cv2.contourArea(contour)
    circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter else 0
    if circularity >= 0.80:
        return "Circle"
    return "Unknown"


def centroid(contour):
    """Center of mass of a contour, from its image moments."""
    m = cv2.moments(contour)
    if m["m00"] == 0:
        x, y, w, h = cv2.boundingRect(contour)
        return x + w // 2, y + h // 2
    return int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])


def detect_objects(hsv):
    """
    Run segmentation + analysis and return a list of detections.

    Each detection is a dict with:
      color, shape, contour, bbox (x, y, w, h), center (cx, cy), area
    Also returns the per-color masks so they can be displayed.
    """
    masks = segment_colors(hsv)
    detections = []
    for color, mask in masks.items():
        # RETR_EXTERNAL: only outer outlines, so holes don't count twice.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA:
                continue
            detections.append({
                "color": color,
                "shape": classify_shape(cnt),
                "contour": cnt,
                "bbox": cv2.boundingRect(cnt),
                "center": centroid(cnt),
                "area": area,
            })
    return detections, masks


def edge_map(frame):
    """Canny edge map, shown as a debug view of the object outlines."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(gray, 50, 150)


# ---------------------------------------------------------------------------
# (a) Image Operations: visualization
# ---------------------------------------------------------------------------

def draw_detections(frame, detections):
    """Draw contours, bounding boxes, labels and centroids on a copy of the frame."""
    out = frame.copy()

    # Semi-transparent fill so the detected region (the mask) is visible.
    overlay = out.copy()
    for d in detections:
        cv2.drawContours(overlay, [d["contour"]], -1, DRAW_COLORS[d["color"]], -1)
    out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)

    for d in detections:
        color = DRAW_COLORS[d["color"]]
        x, y, w, h = d["bbox"]
        cx, cy = d["center"]
        label = f'{d["color"]} {d["shape"]}'

        cv2.drawContours(out, [d["contour"]], -1, color, 2)
        cv2.rectangle(out, (x, y), (x + w, y + h), (40, 40, 40), 1)
        cv2.circle(out, (cx, cy), 4, (255, 255, 255), -1)
        cv2.circle(out, (cx, cy), 4, (0, 0, 0), 1)

        # Label with a filled background so it stays readable on any color.
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(y - 6, th + 6)
        cv2.rectangle(out, (x, ty - th - 5), (x + tw + 6, ty + base - 2), color, -1)
        cv2.putText(out, label, (x + 3, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out


def summarize(detections):
    """Count detections by 'Color Shape' label, sorted by count then name."""
    counts = {}
    for d in detections:
        key = f'{d["color"]} {d["shape"]}'
        counts[key] = counts.get(key, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def draw_summary_panel(frame, detections, fps=None):
    """Add a side panel listing the totals per label (and FPS for video)."""
    h = frame.shape[0]
    panel = np.full((h, 230, 3), 255, dtype=np.uint8)
    cv2.putText(panel, "DETECTED", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)
    cv2.putText(panel, f"Total objects: {len(detections)}", (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    y = 88
    for label, n in summarize(detections):
        color = DRAW_COLORS[label.split()[0]]
        cv2.rectangle(panel, (12, y - 11), (24, y + 1), color, -1)
        cv2.putText(panel, f"{label}: {n}", (32, y), cv2.FONT_HERSHEY_SIMPLEX, 0.47,
                    (30, 30, 30), 1, cv2.LINE_AA)
        y += 22
        if y > h - 30:
            break
    if fps is not None:
        cv2.putText(panel, f"FPS: {fps:.1f}", (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (90, 90, 90), 1, cv2.LINE_AA)
    cv2.line(panel, (0, 0), (0, h), (200, 200, 200), 1)
    return np.hstack([frame, panel])


def combined_mask(masks):
    """Merge all color masks into one, for the 'Segmentation' debug view."""
    out = None
    for m in masks.values():
        out = m.copy() if out is None else cv2.bitwise_or(out, m)
    return out


def build_grid(original, masks, edges, result):
    """
    2x2 view used for the demo and the documentation:
      Original | Segmentation mask
      Canny edges | Final result
    """
    def tag(img, text):
        img = img.copy()
        cv2.rectangle(img, (0, 0), (len(text) * 11 + 16, 28), (30, 30, 30), -1)
        cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    mask_bgr = cv2.cvtColor(combined_mask(masks), cv2.COLOR_GRAY2BGR)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    top = np.hstack([tag(original, "1. Original"), tag(mask_bgr, "2. HSV mask")])
    bottom = np.hstack([tag(edges_bgr, "3. Canny edges"), tag(result, "4. Result")])
    return np.vstack([top, bottom])


def process_frame(frame):
    """
    Full pipeline for one frame. Returns:
      resized frame, detections, masks, edges, annotated result
    """
    resized, hsv = preprocess(frame)
    detections, masks = detect_objects(hsv)
    edges = edge_map(resized)
    result = draw_detections(resized, detections)
    return resized, detections, masks, edges, result
