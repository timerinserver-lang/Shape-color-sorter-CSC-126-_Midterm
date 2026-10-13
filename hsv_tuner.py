"""
hsv_tuner.py
------------
Helper for calibrating the HSV color ranges to YOUR paper and YOUR lights.

Real colored paper under classroom lights rarely matches the default
ranges exactly. Run this before the demo, point the webcam (or an image)
at one color, and drag the sliders until only that color is white in the
mask. Then copy the printed range into COLOR_RANGES in shape_detector.py.

    python hsv_tuner.py --webcam
    python hsv_tuner.py --image samples/hard_03.jpg

Keys:  p = print current range   q / Esc = quit
"""

import argparse

import cv2
import numpy as np

from shape_detector import preprocess

WIN = "HSV Tuner"


def nothing(_):
    pass


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image")
    src.add_argument("--webcam", nargs="?", const=0, type=int)
    args = ap.parse_args()

    cap = None if args.image else cv2.VideoCapture(args.webcam)
    still = cv2.imread(args.image) if args.image else None

    cv2.namedWindow(WIN)
    for name, maxv, init in (("H min", 179, 0), ("H max", 179, 179), ("S min", 255, 90),
                             ("S max", 255, 255), ("V min", 255, 70), ("V max", 255, 255)):
        cv2.createTrackbar(name, WIN, init, maxv, nothing)

    while True:
        frame = still if still is not None else cap.read()[1]
        if frame is None:
            break
        resized, hsv = preprocess(frame)
        lo = np.array([cv2.getTrackbarPos(n, WIN) for n in ("H min", "S min", "V min")])
        hi = np.array([cv2.getTrackbarPos(n, WIN) for n in ("H max", "S max", "V max")])
        mask = cv2.inRange(hsv, lo, hi)
        kept = cv2.bitwise_and(resized, resized, mask=mask)
        cv2.imshow(WIN, np.hstack([resized, kept]))

        key = cv2.waitKey(30) & 0xFF
        if key == ord("p"):
            print(f"(({lo[0]}, {lo[1]}, {lo[2]}), ({hi[0]}, {hi[1]}, {hi[2]}))")
        elif key in (ord("q"), 27):
            print(f"Final range: (({lo[0]}, {lo[1]}, {lo[2]}), ({hi[0]}, {hi[1]}, {hi[2]}))")
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
