"""Offline face detection / recognition for Picture Viewer.

Detection: OpenCV YuNet. Recognition: OpenCV SFace (128-d embeddings, cosine similarity).
Everything is stored locally in a SQLite database; nothing leaves the PC.
"""
import os
import sqlite3
import sys
import threading

import numpy as np

try:
    import cv2
    cv2.setNumThreads(2)
except ImportError:  # the viewer still works without face support
    cv2 = None

MATCH = 0.363        # SFace's recommended cosine threshold: "same person" (search)
SUGGEST = 0.40       # unnamed face -> "looks like <name>"
GROUP = 0.42         # unnamed faces closer than this are put in the same group
MIN_SCORE = 0.85     # detector confidence
MAX_SIDE = 1280      # detect on a downscaled copy
MIN_FACE_PX = 40     # ignore tiny faces (unreliable embeddings), measured on the scaled copy
THUMB_PX = 96

DB_PATH = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
                       "PictureViewer", "faces.db")


def model_dir():
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "models")


def available():
    d = model_dir()
    return cv2 is not None and all(os.path.exists(os.path.join(d, f)) for f in (
        "face_detection_yunet_2023mar.onnx", "face_recognition_sface_2021dec.onnx"))


class FaceEngine:
    """One instance per thread: OpenCV's DNN objects are not thread-safe."""

    def __init__(self):
        d = model_dir()
        self.det = cv2.FaceDetectorYN.create(
            os.path.join(d, "face_detection_yunet_2023mar.onnx"), "", (320, 320),
            MIN_SCORE, 0.3, 5000)
        self.rec = cv2.FaceRecognizerSF.create(
            os.path.join(d, "face_recognition_sface_2021dec.onnx"), "")

    def analyze(self, bgr):
        """bgr: HxWx3 uint8. Returns [(x, y, w, h fractions, embedding, thumb jpeg bytes)]."""
        h0, w0 = bgr.shape[:2]
        scale = min(1.0, MAX_SIDE / max(h0, w0))
        img = cv2.resize(bgr, (round(w0 * scale), round(h0 * scale)),
                         interpolation=cv2.INTER_AREA) if scale < 1 else bgr
        h, w = img.shape[:2]
        self.det.setInputSize((w, h))
        _, rows = self.det.detect(img)
        out = []
        for r in (rows if rows is not None else []):
            x, y, bw, bh = r[:4]
            if min(bw, bh) < MIN_FACE_PX:
                continue
            try:
                feat = self.rec.feature(self.rec.alignCrop(img, r))[0].astype(np.float32)
            except cv2.error:
                continue
            feat /= (np.linalg.norm(feat) or 1.0)
            # Square-ish crop with a margin for the people dialog.
            m = 0.25
            x0, y0 = max(0, int(x - m * bw)), max(0, int(y - m * bh))
            x1, y1 = min(w, int(x + bw * (1 + m))), min(h, int(y + bh * (1 + m)))
            crop = cv2.resize(img[y0:y1, x0:x1], (THUMB_PX, THUMB_PX), interpolation=cv2.INTER_AREA)
            ok, jpg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            out.append((max(0, x) / w, max(0, y) / h, min(bw, w - x) / w, min(bh, h - y) / h,
                        feat, jpg.tobytes() if ok else b""))
        return out


class FaceStore:
    def __init__(self, path=DB_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS photos(path TEXT PRIMARY KEY, mtime INTEGER, size INTEGER);
            CREATE TABLE IF NOT EXISTS people(id INTEGER PRIMARY KEY, name TEXT UNIQUE COLLATE NOCASE);
            CREATE TABLE IF NOT EXISTS faces(
                id INTEGER PRIMARY KEY, path TEXT, x REAL, y REAL, w REAL, h REAL,
                emb BLOB, thumb BLOB, person_id INTEGER, cluster_id INTEGER, ignored INTEGER DEFAULT 0);
            CREATE INDEX IF NOT EXISTS faces_path ON faces(path);
            CREATE INDEX IF NOT EXISTS faces_person ON faces(person_id);
        """)
        self._centroids = None

    # ---- scanning ----
    def is_current(self, path, mtime, size):
        with self.lock:
            r = self.db.execute("SELECT mtime,size FROM photos WHERE path=?", (path,)).fetchone()
        return r == (mtime, size)

    def save(self, path, mtime, size, faces):
        with self.lock, self.db:
            self.db.execute("DELETE FROM faces WHERE path=?", (path,))
            self.db.execute("INSERT OR REPLACE INTO photos VALUES(?,?,?)", (path, mtime, size))
            self.db.executemany(
                "INSERT INTO faces(path,x,y,w,h,emb,thumb) VALUES(?,?,?,?,?,?,?)",
                [(path, x, y, w, h, emb.tobytes(), thumb) for x, y, w, h, emb, thumb in faces])
            self._centroids = None

    # ---- lookups ----
    def faces_in(self, path):
        """[{id,x,y,w,h,person_id,name,emb}] for one photo, with a suggested name if unnamed."""
        with self.lock:
            rows = self.db.execute(
                "SELECT f.id,f.x,f.y,f.w,f.h,f.person_id,p.name,f.emb FROM faces f "
                "LEFT JOIN people p ON p.id=f.person_id WHERE f.path=? AND f.ignored=0",
                (path,)).fetchall()
        out = []
        for i, x, y, w, h, pid, name, emb in rows:
            e = np.frombuffer(emb, np.float32)
            d = dict(id=i, x=x, y=y, w=w, h=h, person_id=pid, name=name, emb=e, suggest=None)
            if pid is None:
                d["suggest"] = self.suggest(e)
            out.append(d)
        return out

    def centroids(self):
        with self.lock:
            if self._centroids is None:
                cents = {}
                for pid, name, emb in self.db.execute(
                        "SELECT p.id,p.name,f.emb FROM faces f JOIN people p ON p.id=f.person_id"):
                    cents.setdefault((pid, name), []).append(np.frombuffer(emb, np.float32))
                self._centroids = []
                for (pid, name), v in cents.items():
                    c = np.mean(v, axis=0)
                    self._centroids.append((pid, name, c / (np.linalg.norm(c) or 1.0)))
            return self._centroids

    def suggest(self, emb, thr=SUGGEST):
        best = None
        for pid, name, c in self.centroids():
            s = float(emb @ c)
            if s >= thr and (best is None or s > best[2]):
                best = (pid, name, s)
        return best  # (person_id, name, score) or None

    def names(self):
        with self.lock:
            return [r[0] for r in self.db.execute(
                "SELECT p.name FROM people p WHERE EXISTS(SELECT 1 FROM faces f WHERE f.person_id=p.id) "
                "ORDER BY p.name")]

    def paths_for_name(self, name, include_suggested=True):
        with self.lock:
            paths = {r[0] for r in self.db.execute(
                "SELECT DISTINCT f.path FROM faces f JOIN people p ON p.id=f.person_id "
                "WHERE p.name=? COLLATE NOCASE", (name,))}
            if include_suggested:
                pid = self.db.execute("SELECT id FROM people WHERE name=? COLLATE NOCASE",
                                      (name,)).fetchone()
                if pid:
                    for path, emb in self.db.execute(
                            "SELECT path,emb FROM faces WHERE person_id IS NULL AND ignored=0"):
                        s = self.suggest(np.frombuffer(emb, np.float32))
                        if s and s[0] == pid[0]:
                            paths.add(path)
        return sorted(paths)

    def paths_like(self, emb, thr=MATCH):
        with self.lock:
            rows = self.db.execute("SELECT path,emb FROM faces WHERE ignored=0").fetchall()
        if not rows:
            return []
        mat = np.stack([np.frombuffer(e, np.float32) for _, e in rows])
        sims = mat @ emb
        return sorted({rows[i][0] for i in np.nonzero(sims >= thr)[0]})

    # ---- grouping / naming ----
    def regroup(self):
        """Cluster unnamed faces greedily by centroid similarity."""
        with self.lock:
            rows = self.db.execute(
                "SELECT id,emb FROM faces WHERE person_id IS NULL AND ignored=0").fetchall()
            cents, sums, counts, assign = [], [], [], []
            for fid, emb in rows:
                e = np.frombuffer(emb, np.float32)
                k = -1
                if cents:
                    sims = np.stack(cents) @ e
                    k = int(np.argmax(sims))
                    if sims[k] < GROUP:
                        k = -1
                if k < 0:
                    cents.append(e.copy()); sums.append(e.copy()); counts.append(1)
                    k = len(cents) - 1
                else:
                    sums[k] += e; counts[k] += 1
                    cents[k] = sums[k] / (np.linalg.norm(sums[k]) or 1.0)
                assign.append((k + 1, fid))
            with self.db:
                self.db.execute("UPDATE faces SET cluster_id=NULL WHERE person_id IS NULL")
                self.db.executemany("UPDATE faces SET cluster_id=? WHERE id=?", assign)

    def groups(self, sample=24):
        """People first, then unnamed groups, biggest first."""
        out = []
        with self.lock:
            for pid, name, n in self.db.execute(
                    "SELECT p.id,p.name,COUNT(f.id) FROM people p JOIN faces f ON f.person_id=p.id "
                    "WHERE f.ignored=0 GROUP BY p.id ORDER BY p.name"):
                thumbs = [r[0] for r in self.db.execute(
                    "SELECT thumb FROM faces WHERE person_id=? AND ignored=0 LIMIT ?", (pid, sample))]
                out.append(dict(kind="person", id=pid, name=name, count=n, thumbs=thumbs, suggest=None))
            unnamed = []
            for cid, n in self.db.execute(
                    "SELECT cluster_id,COUNT(*) FROM faces WHERE person_id IS NULL AND ignored=0 "
                    "AND cluster_id IS NOT NULL GROUP BY cluster_id ORDER BY COUNT(*) DESC"):
                rows = self.db.execute(
                    "SELECT thumb,emb FROM faces WHERE person_id IS NULL AND cluster_id=? AND ignored=0 "
                    "LIMIT ?", (cid, sample)).fetchall()
                m = np.mean([np.frombuffer(e, np.float32) for _, e in rows], axis=0) if rows else None
                sug = self.suggest(m / (np.linalg.norm(m) or 1.0)) if rows else None
                unnamed.append(dict(kind="cluster", id=cid, name=None, count=n,
                                    thumbs=[t for t, _ in rows], suggest=sug[1] if sug else None))
            out += unnamed
        return out

    def _person_id(self, name):
        name = name.strip()
        self.db.execute("INSERT OR IGNORE INTO people(name) VALUES(?)", (name,))
        return self.db.execute("SELECT id FROM people WHERE name=?", (name,)).fetchone()[0]

    def name_group(self, kind, gid, name):
        with self.lock, self.db:
            pid = self._person_id(name)
            if kind == "cluster":
                self.db.execute("UPDATE faces SET person_id=? WHERE cluster_id=? AND person_id IS NULL "
                                "AND ignored=0", (pid, gid))
            else:  # rename / merge a person into another
                self.db.execute("UPDATE faces SET person_id=? WHERE person_id=?", (pid, gid))
                self.db.execute("DELETE FROM people WHERE id=? AND id<>?", (gid, pid))
            self._centroids = None

    def name_face(self, face_id, name):
        with self.lock, self.db:
            self.db.execute("UPDATE faces SET person_id=? WHERE id=?", (self._person_id(name), face_id))
            self._centroids = None

    def unname_face(self, face_id):
        with self.lock, self.db:
            self.db.execute("UPDATE faces SET person_id=NULL WHERE id=?", (face_id,))
            self._centroids = None

    def ignore_face(self, face_id):
        with self.lock, self.db:
            self.db.execute("UPDATE faces SET ignored=1, person_id=NULL WHERE id=?", (face_id,))
            self._centroids = None

    def ignore_group(self, kind, gid):
        with self.lock, self.db:
            col = "person_id" if kind == "person" else "cluster_id"
            self.db.execute(f"UPDATE faces SET ignored=1, person_id=NULL WHERE {col}=?", (gid,))
            self._centroids = None

    def group_faces(self, kind, gid, limit=400):
        """[(face_id, thumb)] for one person / unnamed group, so single faces can be picked."""
        col = "person_id" if kind == "person" else "cluster_id"
        extra = " AND person_id IS NULL" if kind == "cluster" else ""
        with self.lock:
            return self.db.execute(f"SELECT id,thumb FROM faces WHERE {col}=? AND ignored=0{extra} "
                                   "ORDER BY id LIMIT ?", (gid, limit)).fetchall()

    def face_path(self, face_id):
        with self.lock:
            r = self.db.execute("SELECT path FROM faces WHERE id=?", (face_id,)).fetchone()
        return r[0] if r else None

    def name_faces(self, face_ids, name):
        with self.lock, self.db:
            pid = self._person_id(name)
            self.db.executemany("UPDATE faces SET person_id=?, ignored=0 WHERE id=?",
                                [(pid, i) for i in face_ids])
            self._centroids = None

    def unname_faces(self, face_ids):
        with self.lock, self.db:
            self.db.executemany("UPDATE faces SET person_id=NULL WHERE id=?", [(i,) for i in face_ids])
            self._centroids = None

    def ignore_faces(self, face_ids):
        with self.lock, self.db:
            self.db.executemany("UPDATE faces SET ignored=1, person_id=NULL WHERE id=?",
                                [(i,) for i in face_ids])
            self._centroids = None

    def ignored_faces(self, limit=500):
        """[(face_id, thumb)] for faces hidden with 'not a face'."""
        with self.lock:
            return self.db.execute("SELECT id,thumb FROM faces WHERE ignored=1 ORDER BY id LIMIT ?",
                                   (limit,)).fetchall()

    def ignored_count(self):
        with self.lock:
            return self.db.execute("SELECT COUNT(*) FROM faces WHERE ignored=1").fetchone()[0]

    def unignore(self, face_ids=None):
        """Restore the given faces (all ignored faces if None). They return as unnamed faces."""
        with self.lock, self.db:
            if face_ids is None:
                self.db.execute("UPDATE faces SET ignored=0 WHERE ignored=1")
            else:
                self.db.executemany("UPDATE faces SET ignored=0 WHERE id=?", [(i,) for i in face_ids])
            self._centroids = None

    def clear_all(self):
        with self.lock, self.db:
            for t in ("faces", "photos", "people"):
                self.db.execute(f"DELETE FROM {t}")
            self._centroids = None
        self.db.execute("VACUUM")

    def stats(self):
        with self.lock:
            return self.db.execute("SELECT (SELECT COUNT(*) FROM photos),(SELECT COUNT(*) FROM faces),"
                                   "(SELECT COUNT(*) FROM people)").fetchone()
