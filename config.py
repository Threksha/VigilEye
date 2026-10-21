"""Central configuration for the abandoned / removed object detection system.

Secrets (email / Telegram) are read from environment variables so they never
end up in source control.
"""
import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # ---- video ---------------------------------------------------------
    source: str = "0"            # "0" = webcam, or a video path / RTSP URL
    width: int = 640             # frames are resized to this width

    # ---- background models --------------------------------------------
    warmup_frames: int = 30      # frames used to learn the empty scene
    alpha_short: float = 0.02    # fast model: absorbs new things in ~3-4 s
    alpha_long: float = 0.005    # slow model: remembers the "reference" scene
    diff_threshold: int = 30     # gray-level difference that counts as change
    global_change_ratio: float = 0.60  # >60% of pixels changed -> light change / camera bump

    # ---- candidate filtering ------------------------------------------
    min_area: int = 400          # px^2, ignore smaller blobs
    max_area_ratio: float = 0.25 # ignore blobs bigger than 25% of the frame

    # ---- decision logic -----------------------------------------------
    dwell_seconds: float = 10.0  # abandoned: object must be unattended this long
    removed_dwell: float = 2.0   # removed: scene must stay changed this long
    owner_margin: int = 80       # px around the object checked for people
    owner_min_pixels: int = 300  # moving pixels near object => owner present
    lost_grace: float = 2.0      # seconds a track may disappear (occlusion)
    min_votes: int = 5           # frames needed before classifying add/remove
    absorb_after: float = 8.0    # after alerting, merge object into background

    # ---- optional YOLO person detector --------------------------------
    use_yolo: bool = False
    yolo_model: str = "yolov8n.pt"
    yolo_conf: float = 0.4
    yolo_every: int = 3          # run YOLO every N frames

    # ---- alerts --------------------------------------------------------
    snapshot_dir: str = "snapshots"
    db_path: str = "events.db"
    sound: bool = True
    email_enabled: bool = False
    telegram_enabled: bool = False
    smtp_host: str = field(default_factory=lambda: os.getenv("SMTP_HOST", "smtp.gmail.com"))
    smtp_port: int = field(default_factory=lambda: int(os.getenv("SMTP_PORT", "587")))
    smtp_user: str = field(default_factory=lambda: os.getenv("SMTP_USER", ""))
    smtp_pass: str = field(default_factory=lambda: os.getenv("SMTP_PASS", ""))
    alert_to: str = field(default_factory=lambda: os.getenv("ALERT_TO", ""))
    telegram_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_TOKEN", ""))
    telegram_chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))
