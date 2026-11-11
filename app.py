import os, time, threading
from flask import Flask, jsonify, render_template, request
import config
from database.db import Database
from recognition import embedder, matcher, evaluation
from recognition.preprocessing import validate_and_decode, decode_data_url, InvalidImage

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024
os.makedirs(config.STORAGE_DIR, exist_ok=True)
db = Database()
_last_log = {}


def faces_with_matches(img):
    gallery = db.gallery()
    out = []
    for f in embedder.extract(img):
        m = matcher.match(f["embedding"], gallery)
        m["box"] = f["box"]
        out.append(m)
    h, w = img.shape[:2]
    return {"faces": out, "width": w, "height": h}


@app.errorhandler(InvalidImage)
def bad_image(e):
    return jsonify(error=str(e)), 400


@app.errorhandler(Exception)
def server_error(e):
    if hasattr(e, "code") and e.code:
        return jsonify(error=e.description), e.code
    app.logger.exception(e)
    return jsonify(error="Model error. Check the server log. On first run the model "
                         "must download (internet needed once)."), 500


PAGES = {"": ("dashboard", "Dashboard"), "live": ("live", "Live Recognition"),
         "register": ("register", "Register Face"), "image": ("image", "Image Recognition"),
         "faces": ("faces", "Registered Faces"), "history": ("history", "History"),
         "evaluation": ("evaluation", "Model Evaluation"), "about": ("about", "About")}


def make_page(tpl, title):
    def view():
        return render_template(f"{tpl}.html", title=title, pages=PAGES, model=config.MODEL_NAME,
                               threshold=config.MATCH_THRESHOLD, min_images=config.MIN_REG_IMAGES)
    view.__name__ = "page_" + tpl
    return view


for path, (tpl, title) in PAGES.items():
    app.add_url_rule("/" + path, view_func=make_page(tpl, title))


@app.get("/api/health")
def health():
    return jsonify(model_loaded=embedder.is_ready(), model=config.MODEL_NAME)


@app.get("/api/stats")
def stats():
    return jsonify(db.stats() | {"model": config.MODEL_NAME, "threshold": config.MATCH_THRESHOLD,
                                 "model_loaded": embedder.is_ready()})


@app.post("/api/register")
def register():
    pid = request.form.get("person_id", "").strip()
    name = request.form.get("name", "").strip()
    if request.form.get("consent") != "true":
        return jsonify(error="Consent is required before registering a face."), 400
    if not pid or not name or len(pid) > 40 or len(name) > 80:
        return jsonify(error="Enter a name and a person ID (max 40 chars)."), 400
    if any(p["person_id"] == pid for p in db.list_persons()):
        return jsonify(error=f"Person ID '{pid}' already exists."), 409
    vectors, problems = [], []
    for i, f in enumerate(request.files.getlist("images"), 1):
        img = validate_and_decode(f.read(), f.filename or "capture.jpg")
        faces = embedder.extract(img)
        if len(faces) == 1:
            vectors.append(faces[0]["embedding"])
        else:
            problems.append(f"Image {i}: {len(faces)} faces found (need exactly 1)")
    if len(vectors) < config.MIN_REG_IMAGES:
        return jsonify(error=f"Need at least {config.MIN_REG_IMAGES} usable images "
                             f"(got {len(vectors)}).", details=problems), 400
    db.add_person(pid, name, vectors)
    return jsonify(ok=True, stored=len(vectors), skipped=problems)


@app.post("/api/recognize-image")
def recognize_image():
    f = request.files.get("image")
    if not f:
        return jsonify(error="No image uploaded"), 400
    img = validate_and_decode(f.read(), f.filename or "")
    res = faces_with_matches(img)
    for m in res["faces"]:
        db.log(m["person_id"], m["name"], m["similarity"], m["known"], "image")
    return jsonify(res)


@app.post("/api/recognize-frame")
def recognize_frame():
    img = decode_data_url((request.get_json(silent=True) or {}).get("frame", ""))
    res = faces_with_matches(img)
    now = time.time()
    for m in res["faces"]:                       # throttle live logging
        key = m["person_id"] or "unknown"
        if now - _last_log.get(key, 0) > config.LIVE_LOG_COOLDOWN:
            _last_log[key] = now
            db.log(m["person_id"], m["name"], m["similarity"], m["known"], "live")
    return jsonify(res)


@app.get("/api/persons")
def persons():
    return jsonify(db.list_persons())          # no embeddings exposed


@app.delete("/api/persons/<person_id>")
def delete_person(person_id):
    if not db.delete_person(person_id):
        return jsonify(error="Not found"), 404
    return jsonify(ok=True)


@app.get("/api/history")
def history():
    return jsonify(db.history())


@app.post("/api/evaluate")
def evaluate():
    return jsonify(evaluation.run(db))


def _safe_warmup():
    try:
        embedder.warmup()
    except Exception as e:
        print("Model warm-up failed:", e)


if __name__ == "__main__":
    threading.Thread(target=lambda: _safe_warmup(), daemon=True).start()
    app.run(host=config.HOST, port=config.PORT, debug=False, threaded=True)
