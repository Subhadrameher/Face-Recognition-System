# AI-Based Real-Time Face Recognition System
Flask + OpenCV + DeepFace (Facenet512) + SQLite. Cosine-similarity matching, Unknown rejection.

## Run (cloud IDE or local)
```
pip install -r requirements.txt
python app.py
```
Open the preview/forwarded URL for the port shown (default 5000, or `$PORT`). The server binds `0.0.0.0`.
First recognition downloads the Facenet512 weights once (internet needed); a background warm-up starts at launch.
Webcam needs HTTPS or localhost: cloud preview URLs are HTTPS; locally use http://localhost:5000.

## Pages
Dashboard, Live Recognition, Register Face, Image Recognition, Registered Faces, History, Model Evaluation, About.

## Evaluation data
Put labelled images in `storage/eval/<person_id>/` (people you registered) and `storage/eval/unknown/` (people you did NOT register). Use different photos from the ones used for registration. Then click "Run evaluation".

## Tests
`python -m unittest discover -s tests -t .` (database, matching, unknown rejection, validation, cascade delete, metrics; no model download needed).

## Troubleshooting
- Camera blocked: allow it in the address bar; page must be HTTPS or localhost.
- Model error on first use: check internet, then retry; see terminal log.
- TensorFlow install fails: use Python 3.10-3.12 and `pip install --upgrade pip`.
- Tune strictness with env var `MATCH_THRESHOLD` (default 0.70).
