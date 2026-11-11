import base64
import cv2
import numpy as np
import config


class InvalidImage(ValueError):
    pass


def validate_and_decode(data: bytes, filename="upload.jpg"):
    """Check extension, size and that the bytes really decode as an image."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in config.ALLOWED_EXT:
        raise InvalidImage("Only JPG, JPEG and PNG files are allowed")
    if not data or len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise InvalidImage(f"File is empty or larger than {config.MAX_UPLOAD_MB} MB")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise InvalidImage("File is not a valid image")
    return resize(img)


def resize(img, max_side=config.MAX_SIDE):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s))) if s < 1 else img


def decode_data_url(s: str):
    try:
        raw = base64.b64decode(s.split(",", 1)[-1])
    except Exception:
        raise InvalidImage("Bad frame data")
    return validate_and_decode(raw, "frame.jpg")
