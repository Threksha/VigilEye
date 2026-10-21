"""Optional YOLO person detector (pip install ultralytics).

Used to (a) ignore stationary people so they are not reported as objects and
(b) know when the owner is still standing near a bag.
"""


class PersonDetector:
    def __init__(self, model="yolov8n.pt", conf=0.4, every=3):
        from ultralytics import YOLO  # imported lazily: optional dependency

        self.model = YOLO(model)
        self.conf = conf
        self.every = max(1, every)
        self.count = 0
        self.last = []

    def detect(self, frame):
        """Return a list of (x, y, w, h) person boxes for a BGR frame."""
        self.count += 1
        if self.count % self.every == 1 or self.every == 1:
            res = self.model.predict(frame, classes=[0], conf=self.conf, verbose=False)[0]
            self.last = [
                (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                for x1, y1, x2, y2 in res.boxes.xyxy.cpu().numpy()
            ]
        return self.last
