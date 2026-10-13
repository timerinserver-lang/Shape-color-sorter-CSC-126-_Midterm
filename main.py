"""
main.py
-------
Shape and Color Sorter - entry point.

Run on a static image, a recorded video, or a live webcam feed:

    python main.py --image samples/shapes_01.jpg
    python main.py --video samples/conveyor.mp4
    python main.py --webcam            (default camera 0)
    python main.py --webcam 1          (second camera)

Keyboard controls while the window is open:
    v   switch view: Result  ->  4-panel pipeline (Original/Mask/Edges/Result)
    s   save a screenshot to the output/ folder
    p   pause / resume (video and webcam)
    q   quit (Esc also works)

Add --no-display to process without opening a window (results are saved
to output/). Useful for batch runs or machines without a screen.
"""

import argparse
import os
import sys
import time

import cv2

from shape_detector import build_grid, draw_summary_panel, process_frame, summarize

OUTPUT_DIR = "output"
WINDOW = "Shape and Color Sorter"


def render(frame, view, fps=None):
    """Run the pipeline on one frame and build the image to show."""
    resized, detections, masks, edges, result = process_frame(frame)
    if view == "grid":
        canvas = build_grid(resized, masks, edges, result)
    else:
        canvas = draw_summary_panel(result, detections, fps)
    return canvas, detections


def save_image(image, name):
    """Save an image into output/ and return the path."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    cv2.imwrite(path, image)
    print(f"Saved {path}")
    return path


def print_counts(detections):
    """Print the per-label totals to the terminal."""
    print(f"Total objects: {len(detections)}")
    for label, n in summarize(detections):
        print(f"  {label}: {n}")


# ---------------------------------------------------------------------------
# Static image mode
# ---------------------------------------------------------------------------

def run_image(path, show=True):
    frame = cv2.imread(path)
    if frame is None:
        sys.exit(f"Could not read image: {path}")

    base = os.path.splitext(os.path.basename(path))[0]
    result, detections = render(frame, "result")
    grid, _ = render(frame, "grid")
    print_counts(detections)

    if not show:
        save_image(result, f"{base}_result.png")
        save_image(grid, f"{base}_pipeline.png")
        return

    view = "result"
    while True:
        cv2.imshow(WINDOW, result if view == "result" else grid)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("v"):
            view = "grid" if view == "result" else "result"
        elif key == ord("s"):
            save_image(result if view == "result" else grid, f"{base}_{view}.png")
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Video / webcam mode
# ---------------------------------------------------------------------------

def run_stream(source, show=True, label="stream"):
    """Process a recorded video (path) or a webcam (integer index)."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Could not open video source: {source}")

    writer = None
    if not show:
        # Headless: write the annotated video to output/.
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, f"{label}_result.mp4")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25

    view, paused, shot = "result", False, 0
    fps, last = 0.0, time.time()
    canvas = None
    frames = 0

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break  # end of video
            canvas, detections = render(frame, view, fps)
            frames += 1

            # Smoothed frames-per-second, so the number doesn't flicker.
            now = time.time()
            inst = 1.0 / max(now - last, 1e-6)
            fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst
            last = now

            if writer is None and not show:
                h, w = canvas.shape[:2]
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))
            if writer is not None:
                writer.write(canvas)

        if show:
            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("v"):
                view = "grid" if view == "result" else "result"
            elif key == ord("p"):
                paused = not paused
            elif key == ord("s"):
                shot += 1
                save_image(canvas, f"{label}_{view}_{shot:02d}.png")

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved {out_path} ({frames} frames, avg {fps:.1f} FPS)")
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Detect, label and count colored shapes with OpenCV.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="path to an image file")
    src.add_argument("--video", help="path to a video file")
    src.add_argument("--webcam", nargs="?", const=0, type=int, help="webcam index (default 0)")
    p.add_argument("--no-display", action="store_true", help="don't open a window; save results to output/")
    return p.parse_args()


def main():
    args = parse_args()
    show = not args.no_display
    if args.image:
        run_image(args.image, show)
    elif args.video:
        label = os.path.splitext(os.path.basename(args.video))[0]
        run_stream(args.video, show, label)
    else:
        run_stream(args.webcam, show, "webcam")


if __name__ == "__main__":
    main()
