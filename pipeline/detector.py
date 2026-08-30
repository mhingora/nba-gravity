"""Detector wrappers for Stage 1.

Milestone 1 in docs/06-roadmap.md says to start with an off-the-shelf model
before investing in fine-tuning, so `YoloDetector` accepts stock COCO weights
and maps `person` -> `player` and `sports ball` -> `ball`. Point it at custom
weights later and the same mapping picks up native `player` / `ball` classes
with no other change.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import supervision as sv

from pipeline.common import CLASS_BALL, CLASS_PLAYER

# Stock COCO class names that stand in for our classes until a fine-tuned
# basketball detector exists.
CLASS_ALIASES = {
    "player": CLASS_PLAYER,
    "person": CLASS_PLAYER,
    "ball": CLASS_BALL,
    "sports ball": CLASS_BALL,
    "basketball": CLASS_BALL,
}


def _remap_class_names(detections: sv.Detections, names: dict[int, str]) -> sv.Detections:
    """Relabel detections to project classes and drop everything unmapped."""
    if len(detections) == 0:
        return detections

    raw_names = detections.data.get("class_name")
    if raw_names is None:
        raw_names = np.array([names.get(int(i), "") for i in detections.class_id])

    mapped = np.array(
        [CLASS_ALIASES.get(str(name).lower(), "") for name in raw_names], dtype=object
    )
    keep = mapped != ""
    detections = detections[keep]
    detections.data["class_name"] = mapped[keep]
    return detections


class YoloDetector:
    """Ultralytics detector producing `player` / `ball` detections."""

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        confidence: float = 0.25,
        device: str | None = None,
        imgsz: int = 640,
        ball_weights: str | None = None,
        ball_imgsz: int = 1280,
        ball_confidence: float = 0.15,
    ):
        from ultralytics import YOLO  # noqa: PLC0415 — heavy, import on use

        self.model = YOLO(weights)
        self.confidence = confidence
        self.device = device
        self.imgsz = imgsz

        # The ball is the hardest object in the frame (docs/07-known-challenges.md).
        # An optional second pass at higher resolution usually recovers far more
        # of it than widening the player model's confidence floor does.
        self.ball_model = YOLO(ball_weights) if ball_weights else None
        self.ball_imgsz = ball_imgsz
        self.ball_confidence = ball_confidence

    def _predict(self, model, frame: np.ndarray, imgsz: int, confidence: float):
        result = model.predict(
            frame,
            imgsz=imgsz,
            conf=confidence,
            device=self.device,
            verbose=False,
        )[0]
        detections = sv.Detections.from_ultralytics(result)
        return _remap_class_names(detections, result.names)

    def detect(self, frame: np.ndarray) -> sv.Detections:
        detections = self._predict(self.model, frame, self.imgsz, self.confidence)

        if self.ball_model is not None:
            ball = self._predict(
                self.ball_model, frame, self.ball_imgsz, self.ball_confidence
            )
            ball = ball[ball.data["class_name"] == CLASS_BALL]
            detections = detections[detections.data["class_name"] != CLASS_BALL]
            detections = sv.Detections.merge([detections, ball])

        return detections


class ColorBlobDetector:
    """Smoke-test detector for the synthetic clip from `tools/make_test_clip.py`.

    Finds saturated colored discs by HSV threshold. This exists only so the
    stage-1 -> stage-2 -> UI path can be exercised end to end without GPU
    weights; it is useless on real broadcast footage.
    """

    def __init__(self, min_player_area: int = 120, max_ball_area: int = 400):
        self.min_player_area = min_player_area
        self.max_ball_area = max_ball_area

    def detect(self, frame: np.ndarray) -> sv.Detections:
        import cv2

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 120, 90), (180, 255, 255))
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        boxes, class_names, confidences = [], [], []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 40:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            class_name = CLASS_BALL if area <= self.max_ball_area else CLASS_PLAYER
            if class_name == CLASS_PLAYER and area < self.min_player_area:
                continue
            boxes.append([x, y, x + w, y + h])
            class_names.append(class_name)
            confidences.append(min(0.5 + area / 2000.0, 0.99))

        if not boxes:
            return sv.Detections.empty()

        return sv.Detections(
            xyxy=np.asarray(boxes, dtype=np.float32),
            confidence=np.asarray(confidences, dtype=np.float32),
            class_id=np.zeros(len(boxes), dtype=int),
            data={"class_name": np.asarray(class_names, dtype=object)},
        )


def build_detector(name: str, **kwargs):
    """Factory used by `01_detect.py --detector`."""
    if name == "yolo":
        return YoloDetector(**kwargs)
    if name == "colorblob":
        return ColorBlobDetector()
    raise ValueError(f"Unknown detector '{name}'. Expected 'yolo' or 'colorblob'.")
