"""Face detection + embedding using DeepFace (pretrained FaceNet512). Loaded lazily."""
import threading
import numpy as np
import config

_lock = threading.Lock()
_ready = False


def warmup():
    """Downloads/loads model weights on first call (needs internet once)."""
    global _ready
    with _lock:
        if _ready:
            return
        from deepface import DeepFace
        DeepFace.build_model(config.MODEL_NAME)
        _ready = True


def extract(img_bgr):
    """Return [{box:(x,y,w,h), embedding:np.ndarray}] - one per detected face."""
    warmup()
    from deepface import DeepFace
    try:
        with _lock:
            res = DeepFace.represent(img_path=img_bgr, model_name=config.MODEL_NAME,
                                     detector_backend=config.DETECTOR,
                                     enforce_detection=True, align=True)
    except ValueError:      # raised by DeepFace when no face is found
        return []
    out = []
    for r in res:
        a = r["facial_area"]
        out.append({"box": (int(a["x"]), int(a["y"]), int(a["w"]), int(a["h"])),
                    "embedding": np.asarray(r["embedding"], dtype=np.float32)})
    return out


def is_ready():
    return _ready
