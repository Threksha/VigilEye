"""Glue: video source -> detector -> alerts.  Used by main.py and dashboard.py."""
import time

import cv2

from alerts import AlertManager
from detector import AbandonedObjectDetector


class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        src = int(cfg.source) if str(cfg.source).isdigit() else cfg.source
        self.is_file = not isinstance(src, int) and not str(src).startswith(("rtsp://", "http"))
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {cfg.source}")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 1 else 25.0
        self.frame_idx = 0
        self.t0 = time.time()

        person_detector = None
        if cfg.use_yolo:
            from person_detector import PersonDetector
            person_detector = PersonDetector(cfg.yolo_model, cfg.yolo_conf, cfg.yolo_every)
        self.detector = AbandonedObjectDetector(cfg, person_detector)
        self.alerts = AlertManager(cfg)

    def step(self):
        """Read + process one frame. Returns (annotated_frame, events) or (None, [])."""
        ok, frame = self.cap.read()
        if not ok:
            return None, []
        # Video files use frame-based time so tests run faster than real time.
        t = self.frame_idx / self.fps if self.is_file else time.time() - self.t0
        self.frame_idx += 1
        vis, events = self.detector.process(frame, t)
        for ev in events:
            self.alerts.handle(ev)
        return vis, events

    def reset(self):
        self.detector.reset()

    def release(self):
        self.cap.release()
