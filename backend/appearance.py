"""Recognising a player again: appearance fingerprints and shirt numbers.

Fingerprinter: turns each player's crop into a 256-number "fingerprint" with a
small image model (YOLO11n-cls, downloaded automatically the first time). Two
crops of the same person give similar fingerprints; compare with cosine
similarity (1.0 = identical).

NumberReader: reads the shirt number from a player's back with EasyOCR, only on
players big enough on screen to have a readable number (80+ px tall). Optional:
if EasyOCR isn't installed it quietly reads nothing.
"""

import cv2
import numpy as np

EMBED_MODEL = "yolo11n-cls.pt"
MIN_LOOK_HEIGHT = 24      # px: smaller boxes are too blurry to fingerprint
MIN_NUMBER_HEIGHT = 80    # px: below this the number is a few blurry pixels


def _crop(frame, box, pad=0.05):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = int(max(0, x1 - pad * bw)), int(min(w, x2 + pad * bw))
    y1, y2 = int(max(0, y1 - pad * bh)), int(min(h, y2 + pad * bh))
    return frame[y1:y2, x1:x2]


class Fingerprinter:
    def __init__(self, model_name=EMBED_MODEL):
        self.model_name = model_name
        self.model = None

    def __call__(self, frame, boxes):
        """Returns (vectors, indices) for the boxes big enough to fingerprint."""
        crops, keep = [], []
        for i, b in enumerate(boxes):
            if b[3] - b[1] >= MIN_LOOK_HEIGHT and b[2] - b[0] >= 6:
                c = _crop(frame, b)
                if c.size:
                    crops.append(c)
                    keep.append(i)
        if not crops:
            return [], []
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO(self.model_name)
        vecs = self.model.embed(crops, verbose=False)
        out = []
        for v in vecs:
            v = v.detach().cpu().numpy().astype(np.float32).ravel()
            out.append(v / (np.linalg.norm(v) + 1e-9))
        return out, keep


class NumberReader:
    def __init__(self):
        try:
            import easyocr
            self.reader = easyocr.Reader(["en"], gpu=_has_gpu(), verbose=False)
        except Exception:
            self.reader = None  # EasyOCR not installed: no numbers, everything else still works

    def __call__(self, frame, boxes, min_conf=0.5):
        """Returns [(box_index, (number, confidence))] for numbers it could read."""
        if self.reader is None:
            return []
        out = []
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            h = y2 - y1
            if h < MIN_NUMBER_HEIGHT:
                continue
            torso = frame[int(y1 + 0.12 * h):int(y1 + 0.55 * h), int(max(0, x1)):int(x2)]
            if torso.size == 0:
                continue
            scale = max(1.0, 64 / torso.shape[0])
            big = cv2.resize(torso, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            try:
                res = self.reader.readtext(big, allowlist="0123456789", detail=1)
            except Exception:
                continue
            if not res:
                continue
            _, text, p = max(res, key=lambda r: r[2])
            if text.isdigit() and 1 <= int(text) <= 99 and p >= min_conf:
                out.append((i, (int(text), float(p))))
        return out


def _has_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
