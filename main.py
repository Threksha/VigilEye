"""Command-line / desktop-window runner.

Examples
--------
  python main.py                              # webcam
  python main.py --source video.mp4           # video file
  python main.py --source rtsp://user:pw@ip/stream --yolo
  python main.py --source test_video.mp4 --headless --dwell 8
Keys in the window:  q = quit,  r = relearn background
"""
import argparse

import cv2

from config import Config
from pipeline import Pipeline


def parse_args():
    p = argparse.ArgumentParser(description="Abandoned / removed object detection")
    p.add_argument("--source", default="0", help="0 for webcam, or video path / RTSP URL")
    p.add_argument("--dwell", type=float, default=10.0, help="seconds before an unattended object triggers an alert")
    p.add_argument("--removed-dwell", type=float, default=2.0)
    p.add_argument("--min-area", type=int, default=400)
    p.add_argument("--threshold", type=int, default=30, help="pixel difference threshold")
    p.add_argument("--yolo", action="store_true", help="use YOLO person detection (needs ultralytics)")
    p.add_argument("--headless", action="store_true", help="no window")
    p.add_argument("--save-video", default="", help="write annotated video to this path")
    p.add_argument("--snapshot-dir", default="snapshots")
    p.add_argument("--db", default="events.db")
    p.add_argument("--no-sound", action="store_true")
    p.add_argument("--email", action="store_true", help="send email alerts (see README)")
    p.add_argument("--telegram", action="store_true", help="send Telegram alerts (see README)")
    return p.parse_args()


def main():
    a = parse_args()
    cfg = Config(
        source=a.source, dwell_seconds=a.dwell, removed_dwell=a.removed_dwell,
        min_area=a.min_area, diff_threshold=a.threshold, use_yolo=a.yolo,
        snapshot_dir=a.snapshot_dir, db_path=a.db, sound=not a.no_sound,
        email_enabled=a.email, telegram_enabled=a.telegram,
    )
    pipe = Pipeline(cfg)
    writer = None
    total_events = 0
    try:
        while True:
            vis, events = pipe.step()
            if vis is None:
                break
            total_events += len(events)
            if a.save_video:
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(a.save_video, cv2.VideoWriter_fourcc(*"mp4v"), pipe.fps, (w, h))
                writer.write(vis)
            if not a.headless:
                cv2.imshow("Abandoned / Removed Object Detection", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    pipe.reset()
    finally:
        pipe.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
    print(f"Done. {total_events} alert(s) raised. Log: {cfg.db_path}, snapshots: {cfg.snapshot_dir}/")


if __name__ == "__main__":
    main()
