"""Central configuration. Host/port follow the PORT env var used by cloud IDEs."""
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
DB_PATH = os.environ.get("FACEREC_DB", os.path.join(STORAGE_DIR, "facerec.db"))
EVAL_DIR = os.path.join(STORAGE_DIR, "eval")
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 5000))
MODEL_NAME = "Facenet512"          # pretrained; ArcFace also supported
DETECTOR = "opencv"
# Facenet512 cosine-distance threshold in DeepFace is 0.30 -> similarity 0.70
MATCH_THRESHOLD = float(os.environ.get("MATCH_THRESHOLD", 0.70))
MIN_REG_IMAGES = 3
MAX_UPLOAD_MB = 8
ALLOWED_EXT = {"jpg", "jpeg", "png"}
MAX_SIDE = 1024
LIVE_LOG_COOLDOWN = 30             # seconds between history rows per person in live mode
