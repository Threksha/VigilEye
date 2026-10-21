"""Core algorithm: abandoned-object and removed-object detection.

Idea (dual background model)
----------------------------
* bg_long  : slow, selectively-updated model = the "reference" scene.
* bg_short : fast model that quickly absorbs anything that stops moving.

    foreground in bg_long  AND  NOT foreground in bg_short
        => something changed vs. the reference scene and has been STATIC
           for a few seconds  => "static candidate"

Candidates are tracked over time.  Each candidate is classified by comparing
edge strength along its border in the current frame vs. the reference scene:

    stronger border in current frame   -> object was ADDED   (abandoned)
    stronger border in reference scene -> object was REMOVED

Abandoned alarm : candidate stays static for `dwell_seconds` with nobody near.
Removed alarm   : scene stays changed for `removed_dwell` seconds.
"""
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np

ABANDONED = "ABANDONED"
REMOVED = "REMOVED"
UNKNOWN = "UNKNOWN"


@dataclass
class Track:
    id: int
    bbox: tuple
    first_seen: float
    last_seen: float
    static_since: float
    votes_add: int = 0
    votes_rem: int = 0
    alerted: bool = False
    alerted_at: float = 0.0

    @property
    def kind(self):
        if self.votes_add + self.votes_rem == 0:
            return UNKNOWN
        return ABANDONED if self.votes_add >= self.votes_rem else REMOVED

    @property
    def votes(self):
        return self.votes_add + self.votes_rem


def _inter(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    return max(0, x1 - x0) * max(0, y1 - y0)


def _iou(a, b):
    i = _inter(a, b)
    u = a[2] * a[3] + b[2] * b[3] - i
    return i / u if u > 0 else 0.0


def _gradient(gray_f32):
    gx = cv2.Sobel(gray_f32, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f32, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


class AbandonedObjectDetector:
    def __init__(self, cfg, person_detector=None):
        self.cfg = cfg
        self.person_detector = person_detector
        self.k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self.reset()

    def reset(self):
        self.bg_long = None
        self.bg_short = None
        self.n = 0
        self.tracks = {}
        self.next_id = 1

    # ------------------------------------------------------------------
    def process(self, frame, t):
        """Process one BGR frame at time t (seconds).

        Returns (annotated_frame, events).  Each event is a dict with keys:
        id, type, bbox, time, timestamp, snapshot.
        """
        cfg = self.cfg
        h0, w0 = frame.shape[:2]
        if w0 != cfg.width:
            frame = cv2.resize(frame, (cfg.width, int(h0 * cfg.width / w0)))
        H, W = frame.shape[:2]
        vis = frame.copy()
        events = []

        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        gf = gray.astype(np.float32)

        # ---- 1. learn the empty scene -----------------------------------
        self.n += 1
        if self.bg_long is None:
            self.bg_long = np.zeros_like(gf)
        if self.n <= cfg.warmup_frames:
            cv2.accumulateWeighted(gf, self.bg_long, 1.0 / self.n)  # running mean
            self.bg_short = self.bg_long.copy()
            self._hud(vis, f"Learning background {self.n}/{cfg.warmup_frames}")
            return vis, events

        # ---- 2. foreground masks ----------------------------------------
        thr = cfg.diff_threshold
        fg_long = ((np.abs(gf - self.bg_long) > thr).astype(np.uint8)) * 255
        fg_short = ((np.abs(gf - self.bg_short) > thr).astype(np.uint8)) * 255
        fg_long = cv2.morphologyEx(fg_long, cv2.MORPH_OPEN, self.k3)
        fg_long = cv2.morphologyEx(fg_long, cv2.MORPH_CLOSE, self.k9)
        fg_short = cv2.morphologyEx(fg_short, cv2.MORPH_OPEN, self.k3)

        # global change (lights switched, camera bumped) -> relearn scene
        if np.count_nonzero(fg_long) / float(fg_long.size) > cfg.global_change_ratio:
            self.bg_long = gf.copy()
            self.bg_short = gf.copy()
            self.tracks.clear()
            self._hud(vis, "Global scene change - relearning")
            return vis, events

        moving = cv2.dilate(fg_short, self.k9)               # recent motion
        static = cv2.bitwise_and(fg_long, cv2.bitwise_not(moving))
        static = cv2.morphologyEx(static, cv2.MORPH_OPEN, self.k5)
        static = cv2.morphologyEx(static, cv2.MORPH_CLOSE, self.k9)

        # ---- 3. update the models ---------------------------------------
        cv2.accumulateWeighted(gf, self.bg_short, cfg.alpha_short)
        keep = cv2.bitwise_not(cv2.dilate(fg_long, self.k9))  # don't learn changed areas
        cv2.accumulateWeighted(gf, self.bg_long, cfg.alpha_long, mask=keep)

        # ---- 4. candidate blobs -----------------------------------------
        persons = []
        if self.person_detector is not None:
            persons = self.person_detector.detect(frame)

        contours, _ = cv2.findContours(static, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = cfg.max_area_ratio * W * H
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < cfg.min_area or area > max_area:
                continue
            box = cv2.boundingRect(c)
            bw, bh = box[2], box[3]
            if any(_inter(box, p) > 0.5 * bw * bh for p in persons):
                continue  # a stationary person is not an object
            candidates.append(box)

        # ---- 5. tracking -------------------------------------------------
        grad_cur = grad_bg = None
        if candidates:
            grad_cur = _gradient(gf)
            grad_bg = _gradient(self.bg_long)

        unmatched = set(self.tracks)
        new_alerts = []
        for box in candidates:
            best, best_iou = None, 0.2
            for tid in unmatched:
                iou = _iou(box, self.tracks[tid].bbox)
                if iou > best_iou:
                    best, best_iou = tid, iou
            if best is not None:
                unmatched.discard(best)
                tr = self.tracks[best]
                tr.bbox, tr.last_seen = box, t
            else:
                tr = Track(self.next_id, box, t, t, t)
                self.tracks[tr.id] = tr
                self.next_id += 1

            # add / remove vote
            c_edge, b_edge = self._border_edge(grad_cur, grad_bg, tr.bbox)
            if c_edge > b_edge:
                tr.votes_add += 1
            else:
                tr.votes_rem += 1

            # is a person (the owner) close by?  -> restart the unattended timer
            if self._person_near(moving, persons, tr.bbox, W, H):
                tr.static_since = t

            if not tr.alerted and tr.votes >= cfg.min_votes:
                if tr.kind == ABANDONED and t - tr.static_since >= cfg.dwell_seconds:
                    new_alerts.append(tr)
                elif tr.kind == REMOVED and t - tr.first_seen >= cfg.removed_dwell:
                    new_alerts.append(tr)

        for tid in unmatched:  # tracks not seen this frame
            if t - self.tracks[tid].last_seen > cfg.lost_grace:
                del self.tracks[tid]

        # ---- 6. alarms ---------------------------------------------------
        for tr in new_alerts:
            tr.alerted, tr.alerted_at = True, t

        # merge already-reported objects into the background after a while
        for tid in list(self.tracks):
            tr = self.tracks[tid]
            if tr.alerted and t - tr.alerted_at >= cfg.absorb_after:
                x, y, w, h = tr.bbox
                self.bg_long[y:y + h, x:x + w] = gf[y:y + h, x:x + w]
                self.bg_short[y:y + h, x:x + w] = gf[y:y + h, x:x + w]
                del self.tracks[tid]

        # ---- 7. draw -----------------------------------------------------
        for tr in self.tracks.values():
            x, y, w, h = tr.bbox
            if tr.alerted:
                color, label = (0, 0, 255), f"ALERT #{tr.id} {tr.kind}"
            elif tr.kind == REMOVED:
                color, label = (0, 165, 255), f"#{tr.id} removed? {t - tr.first_seen:.0f}/{cfg.removed_dwell:.0f}s"
            else:
                color, label = (0, 255, 255), f"#{tr.id} static {t - tr.static_since:.0f}/{cfg.dwell_seconds:.0f}s"
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            cv2.putText(vis, label, (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        for (px, py, pw, ph) in persons:
            cv2.rectangle(vis, (px, py), (px + pw, py + ph), (0, 200, 0), 1)
        self._hud(vis, f"Candidates: {len(self.tracks)}")

        for tr in new_alerts:
            events.append({
                "id": tr.id,
                "type": tr.kind,
                "bbox": tr.bbox,
                "time": t,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "snapshot": vis.copy(),
            })
        return vis, events

    # ------------------------------------------------------------------
    @staticmethod
    def _border_edge(grad_cur, grad_bg, bbox, band=3):
        """Mean gradient on a band around the bbox in current vs. reference."""
        x, y, w, h = bbox
        mask = np.zeros(grad_cur.shape, np.uint8)
        cv2.rectangle(mask, (x - band, y - band), (x + w + band, y + h + band), 255, -1)
        cv2.rectangle(mask, (x + band, y + band), (x + w - band, y + h - band), 0, -1)
        sel = mask > 0
        return float(grad_cur[sel].mean()), float(grad_bg[sel].mean())

    def _person_near(self, moving, persons, bbox, W, H):
        cfg = self.cfg
        x, y, w, h = bbox
        m = cfg.owner_margin
        x0, y0 = max(0, x - m), max(0, y - m)
        x1, y1 = min(W, x + w + m), min(H, y + h + m)
        if any(_inter((x0, y0, x1 - x0, y1 - y0), p) > 0 for p in persons):
            return True
        roi = moving[y0:y1, x0:x1].copy()
        roi[y - y0:y - y0 + h, x - x0:x - x0 + w] = 0  # ignore the object itself
        return cv2.countNonZero(roi) > cfg.owner_min_pixels

    @staticmethod
    def _hud(img, text):
        cv2.putText(img, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3)
        cv2.putText(img, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
