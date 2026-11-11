import os, tempfile, unittest
import cv2, numpy as np
from database.db import Database
from recognition import matcher
from recognition.evaluation import compute_metrics
from recognition.preprocessing import validate_and_decode, InvalidImage


def vec(seed, noise=0.0):
    r = np.random.RandomState(seed)
    v = r.randn(512).astype(np.float32)
    return v + noise * np.random.RandomState(seed + 99).randn(512).astype(np.float32)


class Core(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = Database(os.path.join(self.tmp, "t.db"))

    def test_register_store_and_duplicate(self):
        self.db.add_person("A1", "Asha", [vec(1), vec(1, .1), vec(1, .1)])
        self.assertEqual(self.db.list_persons()[0]["images"], 3)
        with self.assertRaises(ValueError):
            self.db.add_person("A1", "Other", [vec(2)])

    def test_embedding_roundtrip(self):
        self.db.add_person("A1", "Asha", [vec(1)])
        self.assertTrue(np.allclose(self.db.gallery()[0][2], vec(1)))

    def test_match_known_and_unknown(self):
        self.db.add_person("A1", "Asha", [vec(1)])
        g = self.db.gallery()
        self.assertEqual(matcher.match(vec(1, .2), g)["name"], "Asha")
        r = matcher.match(vec(7), g)
        self.assertFalse(r["known"]); self.assertEqual(r["name"], "Unknown")
        self.assertEqual(matcher.match(vec(7), [])["name"], "Unknown")

    def test_cascade_delete(self):
        self.db.add_person("A1", "Asha", [vec(1), vec(1)])
        self.assertTrue(self.db.delete_person("A1"))
        self.assertEqual(self.db.count_embeddings("A1"), 0)
        self.assertFalse(self.db.delete_person("A1"))

    def test_history_stats(self):
        self.db.log("A1", "Asha", .9, True, "image"); self.db.log(None, "Unknown", .2, False, "live")
        s = self.db.stats()
        self.assertEqual((s["attempts"], s["known"], s["unknown"]), (2, 1, 1))

    def test_image_validation(self):
        ok, buf = cv2.imencode(".png", np.zeros((20, 20, 3), np.uint8))
        self.assertEqual(validate_and_decode(buf.tobytes(), "a.png").shape[:2], (20, 20))
        with self.assertRaises(InvalidImage): validate_and_decode(buf.tobytes(), "a.gif")
        with self.assertRaises(InvalidImage): validate_and_decode(b"not an image", "a.jpg")
        with self.assertRaises(InvalidImage): validate_and_decode(b"", "a.jpg")

    def test_metrics(self):
        m = compute_metrics(["a", "a", "unknown", "unknown"], ["a", "unknown", "unknown", "a"])
        self.assertAlmostEqual(m["accuracy"], .5)
        self.assertAlmostEqual(m["far"], .5); self.assertAlmostEqual(m["frr"], .5)


if __name__ == "__main__":
    unittest.main()
