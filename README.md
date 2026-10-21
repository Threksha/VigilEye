# Abandoned & Removed Object Detection System

Detects motion in a monitored area and raises real-time alerts when an object is
**left behind** or **removed**. Built with Python + OpenCV.

## Files

| File | Purpose |
|---|---|
| `config.py` | All tunable settings + alert credentials (from environment variables) |
| `detector.py` | Core algorithm (dual background model, tracking, add/remove classification) |
| `person_detector.py` | Optional YOLO person detector (reduces false alarms) |
| `alerts.py` | Snapshot, SQLite log, console, sound, email, Telegram |
| `pipeline.py` | Video source -> detector -> alerts |
| `main.py` | Desktop window / command-line runner |
| `dashboard.py` | Web dashboard (live feed + alert table) for security staff |
| `make_test_video.py` | Creates a synthetic test video (no camera needed) |

## Setup

```bash
pip install -r requirements.txt
```
Python 3.9+ recommended. For the optional YOLO mode: `pip install ultralytics`
(downloads `yolov8n.pt` on first use).

## Quick test (no camera needed)

```bash
python make_test_video.py                       # creates test_video.mp4
python main.py --source test_video.mp4 --dwell 8
```
Expected: one **ABANDONED** alert (the dropped bag) and one **REMOVED** alert
(the monitor taken from the desk). Snapshots go to `snapshots/`, events to `events.db`.

## Run on real cameras

```bash
python main.py                                   # laptop webcam
python main.py --source video.mp4                # recorded video
python main.py --source rtsp://user:pass@192.168.1.10/stream1 --yolo
python dashboard.py --source 0                   # web UI at http://localhost:5000
```
Window keys: `q` quit, `r` relearn the background.

**Important:** when the program starts, the first ~30 frames should show the
scene *without* the objects you want to protect being moved. They are learned as
the reference scene.

## Alert channels

* Always on: console message, saved snapshot, SQLite record, beep.
* **Email** (`--email`): set `SMTP_USER`, `SMTP_PASS` (Gmail needs an *App Password*), `ALERT_TO`.
* **Telegram** (`--telegram`): create a bot with @BotFather, set `TELEGRAM_TOKEN` and `TELEGRAM_CHAT_ID`.

```bash
export SMTP_USER=you@gmail.com SMTP_PASS=app-password ALERT_TO=guard@example.com
python main.py --source 0 --email
```

## How it works

1. **Learn** the empty scene for the first 30 frames.
2. Two background models:
   * `bg_long` (slow, only updated where nothing changed) = reference scene
   * `bg_short` (fast) = absorbs anything that stops moving within ~3 s
3. `changed in bg_long  AND  NOT changed in bg_short` = **static candidate**
   (something differs from the reference and has stopped moving).
4. Candidates are **tracked** between frames (IoU matching).
5. **Added vs removed:** compare edge strength along the candidate's border in the
   current frame vs the reference. Stronger in the current frame = object added
   (abandoned); stronger in the reference = object removed.
6. **Alarm rules**
   * Abandoned: static for `dwell_seconds` with no person nearby (timer restarts while the owner is close).
   * Removed: scene stays changed for `removed_dwell` seconds.
7. After alerting, the object is merged into the background so it is not re-reported.
8. A global change (lights switched, camera bumped) triggers automatic relearning.

## Tuning (see `config.py`)

| Problem | Adjust |
|---|---|
| Too many false alarms | raise `diff_threshold`, `min_area`, `dwell_seconds`; enable `--yolo` |
| Small objects missed | lower `min_area` and `diff_threshold` |
| Alerts too slow / fast | `dwell_seconds`, `removed_dwell` |
| Owner standing near bag still triggers | raise `owner_margin` or use `--yolo` |

## Known limitations (good to mention in your report)

* Without `--yolo`, a person who stands completely still for a long time can be
  mistaken for an abandoned object. YOLO mode fixes this.
* Strong shadows, flickering lights or reflective floors can cause false alarms.
* Fixed cameras only; a moving/PTZ camera needs the background relearned.
* Heavy crowds that constantly occlude an object can delay or hide detection.

## Ideas for extension

* Replace the person filter with DeepSORT to link an object to its owner.
* Classify the left object (bag/box/suitcase) with YOLO.
* Define polygon zones (e.g. only monitor the platform edge).
* Evaluate on PETS2006 / AVSS / ABODA datasets and report precision and recall.
