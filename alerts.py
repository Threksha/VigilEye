"""Alerting: snapshot + SQLite log + console + sound + optional email / Telegram."""
import os
import smtplib
import sqlite3
import sys
import threading
from datetime import datetime
from email.message import EmailMessage

import cv2


class AlertManager:
    def __init__(self, cfg):
        self.cfg = cfg
        os.makedirs(cfg.snapshot_dir, exist_ok=True)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(cfg.db_path, check_same_thread=False)
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS events (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   ts TEXT, event_type TEXT, track_id INTEGER,
                   x INTEGER, y INTEGER, w INTEGER, h INTEGER, snapshot TEXT)"""
        )
        self.db.commit()

    # ------------------------------------------------------------------
    def handle(self, event):
        """Called once per new alarm."""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{stamp}_{event['type']}_{event['id']}.jpg"
        path = os.path.join(self.cfg.snapshot_dir, fname)
        cv2.imwrite(path, event["snapshot"])

        x, y, w, h = event["bbox"]
        with self.lock:
            self.db.execute(
                "INSERT INTO events (ts, event_type, track_id, x, y, w, h, snapshot) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (event["timestamp"], event["type"], event["id"], x, y, w, h, fname),
            )
            self.db.commit()

        msg = f"[{event['timestamp']}] {event['type']} object detected (track #{event['id']}) at x={x}, y={y}"
        print("\033[91m" + msg + "\033[0m", flush=True)

        if self.cfg.sound:
            threading.Thread(target=self._beep, daemon=True).start()
        if self.cfg.email_enabled:
            threading.Thread(target=self._email, args=(msg, path), daemon=True).start()
        if self.cfg.telegram_enabled:
            threading.Thread(target=self._telegram, args=(msg, path), daemon=True).start()

    def recent(self, limit=50):
        with self.lock:
            cur = self.db.execute(
                "SELECT ts, event_type, track_id, x, y, w, h, snapshot FROM events "
                "ORDER BY id DESC LIMIT ?", (limit,))
            rows = cur.fetchall()
        keys = ["ts", "type", "track_id", "x", "y", "w", "h", "snapshot"]
        return [dict(zip(keys, r)) for r in rows]

    # ------------------------------------------------------------------
    @staticmethod
    def _beep():
        try:
            if sys.platform.startswith("win"):
                import winsound
                for _ in range(3):
                    winsound.Beep(1200, 300)
            else:
                sys.stdout.write("\a")
                sys.stdout.flush()
        except Exception:
            pass

    def _email(self, msg, image_path):
        c = self.cfg
        if not (c.smtp_user and c.smtp_pass and c.alert_to):
            print("Email not configured (SMTP_USER / SMTP_PASS / ALERT_TO)")
            return
        try:
            m = EmailMessage()
            m["Subject"] = "Security alert: " + msg[:60]
            m["From"], m["To"] = c.smtp_user, c.alert_to
            m.set_content(msg)
            with open(image_path, "rb") as f:
                m.add_attachment(f.read(), maintype="image", subtype="jpeg",
                                 filename=os.path.basename(image_path))
            with smtplib.SMTP(c.smtp_host, c.smtp_port, timeout=20) as s:
                s.starttls()
                s.login(c.smtp_user, c.smtp_pass)
                s.send_message(m)
        except Exception as e:
            print("Email failed:", e)

    def _telegram(self, msg, image_path):
        c = self.cfg
        if not (c.telegram_token and c.telegram_chat_id):
            print("Telegram not configured (TELEGRAM_TOKEN / TELEGRAM_CHAT_ID)")
            return
        try:
            import requests
            url = f"https://api.telegram.org/bot{c.telegram_token}/sendPhoto"
            with open(image_path, "rb") as f:
                requests.post(url, data={"chat_id": c.telegram_chat_id, "caption": msg},
                              files={"photo": f}, timeout=20)
        except Exception as e:
            print("Telegram failed:", e)
