import sqlite3
import numpy as np
import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS persons(
  person_id TEXT PRIMARY KEY, name TEXT NOT NULL,
  consent INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS embeddings(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  person_id TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
  vector BLOB NOT NULL, model TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS history(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  person_id TEXT, name TEXT, similarity REAL, known INTEGER NOT NULL,
  source TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
"""


class Database:
    def __init__(self, path=None):
        self.path = path or config.DB_PATH
        c = self._c()
        try:
            c.executescript(SCHEMA)
        finally:
            c.close()

    def _c(self):
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys=ON")
        return c

    def _run(self, sql, args=(), fetch=False):
        c = self._c()
        try:
            cur = c.execute(sql, args)
            rows = [dict(r) for r in cur.fetchall()] if fetch else None
            c.commit()
            return rows
        finally:
            c.close()

    def add_person(self, person_id, name, vectors, model=config.MODEL_NAME):
        c = self._c()
        try:
            if c.execute("SELECT 1 FROM persons WHERE person_id=?", (person_id,)).fetchone():
                raise ValueError("Person ID already exists")
            c.execute("INSERT INTO persons(person_id,name) VALUES(?,?)", (person_id, name))
            for v in vectors:
                c.execute("INSERT INTO embeddings(person_id,vector,model) VALUES(?,?,?)",
                          (person_id, np.asarray(v, dtype=np.float32).tobytes(), model))
            c.commit()
        finally:
            c.close()

    def list_persons(self):
        return self._run("""SELECT p.person_id,p.name,p.created_at,COUNT(e.id) AS images
            FROM persons p LEFT JOIN embeddings e ON e.person_id=p.person_id
            GROUP BY p.person_id ORDER BY p.created_at DESC""", fetch=True)

    def delete_person(self, person_id):
        c = self._c()
        try:
            n = c.execute("DELETE FROM persons WHERE person_id=?", (person_id,)).rowcount
            c.commit()
            return n > 0
        finally:
            c.close()

    def count_embeddings(self, person_id):
        return self._run("SELECT COUNT(*) AS n FROM embeddings WHERE person_id=?",
                         (person_id,), fetch=True)[0]["n"]

    def gallery(self):
        """Internal use only: [(person_id, name, vector)]. Never returned by an API."""
        rows = self._run("""SELECT e.person_id,p.name,e.vector FROM embeddings e
            JOIN persons p ON p.person_id=e.person_id""", fetch=True)
        return [(r["person_id"], r["name"], np.frombuffer(r["vector"], dtype=np.float32)) for r in rows]

    def log(self, person_id, name, similarity, known, source):
        self._run("INSERT INTO history(person_id,name,similarity,known,source) VALUES(?,?,?,?,?)",
                  (person_id, name, similarity, 1 if known else 0, source))

    def history(self, limit=200):
        return self._run("SELECT * FROM history ORDER BY id DESC LIMIT ?", (limit,), fetch=True)

    def stats(self):
        s = self._run("""SELECT (SELECT COUNT(*) FROM persons) AS people,
            COUNT(*) AS attempts, COALESCE(SUM(known),0) AS known FROM history""", fetch=True)[0]
        s["unknown"] = s["attempts"] - s["known"]
        s["daily"] = self._run("""SELECT date(created_at) AS day, COUNT(*) AS total,
            SUM(known) AS known FROM history GROUP BY day ORDER BY day DESC LIMIT 14""", fetch=True)[::-1]
        s["recent"] = self.history(8)
        return s
