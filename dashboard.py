"""Web dashboard for security personnel: live annotated video + alert history.

  python dashboard.py --source 0
  python dashboard.py --source test_video.mp4
Then open http://localhost:5000
"""
import argparse
import threading
import time

import cv2
from flask import Flask, Response, jsonify, send_from_directory

from config import Config
from pipeline import Pipeline

app = Flask(__name__)
state = {"jpeg": None, "pipe": None, "cfg": None}

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Object Monitoring Dashboard</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;background:#111;color:#eee}
 header{padding:12px 16px;background:#1b1b1b;font-weight:600}
 main{display:flex;flex-wrap:wrap;gap:16px;padding:16px}
 .box{background:#1b1b1b;border-radius:8px;padding:12px;flex:1 1 360px}
 img.live{width:100%;border-radius:6px}
 table{width:100%;border-collapse:collapse;font-size:14px}
 td,th{padding:6px;border-bottom:1px solid #333;text-align:left}
 .ABANDONED{color:#ff6b6b;font-weight:600}.REMOVED{color:#ffa94d;font-weight:600}
 a{color:#74c0fc}
</style></head><body>
<header>Abandoned / Removed Object Monitoring</header>
<main>
 <div class="box"><h3>Live feed</h3><img class="live" src="/video"></div>
 <div class="box"><h3>Alerts</h3>
  <table><thead><tr><th>Time</th><th>Type</th><th>Track</th><th>Snapshot</th></tr></thead>
  <tbody id="rows"></tbody></table></div>
</main>
<script>
async function load(){
  const r = await fetch('/api/events'); const ev = await r.json();
  document.getElementById('rows').innerHTML = ev.map(e =>
    `<tr><td>${e.ts}</td><td class="${e.type}">${e.type}</td><td>#${e.track_id}</td>
     <td><a href="/snapshots/${e.snapshot}" target="_blank">view</a></td></tr>`).join('');
}
load(); setInterval(load, 3000);
</script></body></html>"""


def worker():
    pipe = state["pipe"]
    delay = 1.0 / pipe.fps if pipe.is_file else 0
    while True:
        vis, _ = pipe.step()
        if vis is None:
            break
        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            state["jpeg"] = buf.tobytes()
        if delay:
            time.sleep(delay)


def mjpeg():
    while True:
        if state["jpeg"] is not None:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + state["jpeg"] + b"\r\n"
        time.sleep(0.04)


@app.route("/")
def index():
    return PAGE


@app.route("/video")
def video():
    return Response(mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/events")
def events():
    return jsonify(state["pipe"].alerts.recent(50))


@app.route("/snapshots/<path:name>")
def snapshot(name):
    return send_from_directory(state["cfg"].snapshot_dir, name)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0")
    p.add_argument("--dwell", type=float, default=10.0)
    p.add_argument("--yolo", action="store_true")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--no-sound", action="store_true")
    a = p.parse_args()
    cfg = Config(source=a.source, dwell_seconds=a.dwell, use_yolo=a.yolo, sound=not a.no_sound)
    state["cfg"] = cfg
    state["pipe"] = Pipeline(cfg)
    threading.Thread(target=worker, daemon=True).start()
    app.run(host="0.0.0.0", port=a.port, threaded=True)
