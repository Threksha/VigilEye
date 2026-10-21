"""Generate a synthetic CCTV-style test video (no camera or dataset needed).

Timeline (15 fps, 50 s):
   5-10 s  person walks in
  10 s     person drops a bag and walks away          -> ABANDONED alert expected
  28-34 s  another person walks to the monitor on the desk and takes it
                                                      -> REMOVED alert expected
"""
import sys

import cv2
import numpy as np

W, H, FPS, DUR = 640, 480, 15, 50


def background():
    rng = np.random.RandomState(1)
    yy, xx = np.mgrid[0:H, 0:W]
    base = 205 - (yy / H) * 35 + (xx / W) * 8                  # smooth gradient
    base = base + cv2.GaussianBlur(rng.randn(H, W).astype(np.float32) * 6, (0, 0), 6)
    img = np.dstack([base, base + 3, base + 8]).clip(0, 255).astype(np.uint8)
    cv2.rectangle(img, (0, 400), (W, H), (150, 160, 170), -1)  # floor strip
    return img


def person(img, x, y, scale=1.0):
    cv2.rectangle(img, (int(x), int(y + 25 * scale)), (int(x + 36 * scale), int(y + 120 * scale)), (70, 50, 130), -1)
    cv2.circle(img, (int(x + 18 * scale), int(y + 12 * scale)), int(14 * scale), (110, 140, 190), -1)


def lerp(a, b, t0, t1, t):
    return a + (b - a) * min(1.0, max(0.0, (t - t0) / (t1 - t0)))


def main(out="test_video.mp4"):
    bg = background()
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    rng = np.random.RandomState(7)
    for i in range(FPS * DUR):
        t = i / FPS
        f = bg.copy()
        # pre-existing object (a monitor on a desk) removed at t=31
        if t < 31:
            cv2.rectangle(f, (470, 300), (530, 350), (40, 40, 45), -1)
            cv2.rectangle(f, (490, 350), (510, 360), (60, 60, 60), -1)
        # person 1: walks in, drops bag at t=10, walks off
        if 5 <= t < 17:
            if t < 10:
                px = lerp(-40, 200, 5, 10, t)
            elif t < 12:
                px = 200
            else:
                px = lerp(200, 700, 12, 17, t)
            person(f, px, 280)
        if t >= 10:  # the bag
            cv2.rectangle(f, (215, 375), (265, 410), (30, 30, 150), -1)
            cv2.rectangle(f, (225, 365), (255, 375), (30, 30, 120), -1)
        # person 2: walks to the monitor, takes it, walks off
        if 28 <= t < 40:
            if t < 31:
                px = lerp(700, 520, 28, 31, t)
            elif t < 33:
                px = 520
            else:
                px = lerp(520, -60, 33, 40, t)
            person(f, px, 270)
        noise = rng.randn(H, W, 3) * 2.0
        f = np.clip(f.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        vw.write(f)
    vw.release()
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test_video.mp4")
