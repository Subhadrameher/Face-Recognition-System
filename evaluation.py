"""Evaluation on labelled images: storage/eval/<person_id>/*.jpg and
storage/eval/unknown/*.jpg (people NOT registered, used to measure false acceptance)."""
import os
import cv2
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import config
from recognition import embedder, matcher

UNKNOWN = "unknown"


def compute_metrics(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                 average="macro", zero_division=0)
    unk = [q for t, q in zip(y_true, y_pred) if t == UNKNOWN]
    kn = [(t, q) for t, q in zip(y_true, y_pred) if t != UNKNOWN]
    far = sum(q != UNKNOWN for q in unk) / len(unk) if unk else None
    frr = sum(t != q for t, q in kn) / len(kn) if kn else None
    return {"samples": len(y_true), "accuracy": accuracy_score(y_true, y_pred),
            "precision": p, "recall": r, "f1": f, "far": far, "frr": frr,
            "labels": labels,
            "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist()}


def run(db, eval_dir=config.EVAL_DIR):
    gallery = db.gallery()
    if not gallery:
        return {"available": False, "reason": "No people are registered yet."}
    y_true, y_pred, skipped = [], [], 0
    folders = sorted(os.listdir(eval_dir)) if os.path.isdir(eval_dir) else []
    for label in folders:
        folder = os.path.join(eval_dir, label)
        if not os.path.isdir(folder):
            continue
        for fn in sorted(os.listdir(folder)):
            if fn.rsplit(".", 1)[-1].lower() not in config.ALLOWED_EXT:
                continue
            img = cv2.imread(os.path.join(folder, fn))
            faces = embedder.extract(img) if img is not None else []
            if not faces:
                skipped += 1
                continue
            f = max(faces, key=lambda x: x["box"][2] * x["box"][3])
            m = matcher.match(f["embedding"], gallery)
            y_true.append(label)
            y_pred.append(m["person_id"] if m["known"] else UNKNOWN)
    if not y_true:
        return {"available": False, "reason": "No labelled test images with a detectable "
                "face found in storage/eval/<person_id>/ (see README)."}
    out = compute_metrics(y_true, y_pred)
    out.update(available=True, skipped=skipped, threshold=config.MATCH_THRESHOLD)
    return out
