import numpy as np
import config


def cosine(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d else 0.0


def match(query, gallery, threshold=None):
    """gallery: [(person_id, name, vector)]. Highest cosine similarity wins;
    below threshold -> Unknown."""
    threshold = config.MATCH_THRESHOLD if threshold is None else threshold
    best = (None, "Unknown", -1.0)
    for pid, name, vec in gallery:
        s = cosine(query, vec)
        if s > best[2]:
            best = (pid, name, s)
    if best[0] is None:
        return {"person_id": None, "name": "Unknown", "similarity": 0.0, "known": False}
    known = best[2] >= threshold
    return {"person_id": best[0] if known else None, "name": best[1] if known else "Unknown",
            "similarity": round(best[2], 4), "known": known}
