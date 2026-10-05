"""Picture viewer: main image on top, scrolling thumbnail strip at the bottom."""
import hashlib
import os
import random
import subprocess
import sys
import threading

import numpy as np
from collections import OrderedDict

from PySide6.QtCore import (QAbstractListModel, QModelIndex, QObject, QRunnable, QSize, Qt,
                            QPoint, QRectF, QSettings, QThread, QThreadPool, QTimer, Signal)
from PySide6.QtGui import (QAction, QActionGroup, QColor, QIcon, QImage, QPalette, QImageReader, QKeySequence,
                           QBrush, QPainter, QPen, QPixmap, QTransform)
from PySide6.QtWidgets import (QApplication, QGraphicsRectItem, QGraphicsSimpleTextItem,
                               QInputDialog, QMenu, QFileDialog, QGraphicsPixmapItem, QGraphicsScene,
                               QGraphicsView, QLabel, QMainWindow,
                               QMessageBox, QSplitter, QToolBar, QListView)

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pillow_heif = None
try:
    import rawpy
except ImportError:
    rawpy = None
try:
    import faces as fc
    from facesui import PeopleDialog
    if not fc.available():
        fc = None
except Exception:  # OpenCV or models missing: the viewer still works without face support
    fc = None
try:
    from send2trash import send2trash
except ImportError:
    send2trash = None

QT_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tif", ".tiff", ".ico"}
HEIC_EXTS = {".heic", ".heif"} if pillow_heif else set()
RAW_EXTS = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".raf"} if rawpy else set()
ALL_EXTS = QT_EXTS | HEIC_EXTS | RAW_EXTS
APP_NAME = "Picture Viewer"
VERSION = "1.1"
BUILD_DATE = "dev"  # stamped by the build step
THUMB = 110
THUMB_PX = 160  # decoded thumbnail size (sharper on HiDPI)
CELL = THUMB + 14
IMG_CACHE = 5
THUMB_CACHE = 1500


def load_image(path, max_size=None):
    """Return a QImage for any supported file (or a null QImage)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in RAW_EXTS:
        try:
            with rawpy.imread(path) as raw:
                try:
                    t = raw.extract_thumb()
                    if t.format == rawpy.ThumbFormat.JPEG and not max_size is None:
                        img = QImage.fromData(t.data)
                        if not img.isNull():
                            return img
                except Exception:
                    pass
                rgb = raw.postprocess(use_camera_wb=True, half_size=max_size is not None)
            h, w, _ = rgb.shape
            return QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format_RGB888).copy()
        except Exception:
            return QImage()
    if ext in HEIC_EXTS:
        try:
            from PIL import Image, ImageOps
            im = ImageOps.exif_transpose(Image.open(path)).convert("RGBA")
            return QImage(im.tobytes(), im.width, im.height, 4 * im.width,
                          QImage.Format_RGBA8888).copy()
        except Exception:
            return QImage()
    reader = QImageReader(path)
    reader.setAutoTransform(True)
    return reader.read()


class Signals(QObject):
    thumb = Signal(str, QImage, bool)   # path, image, skipped
    image = Signal(str, QImage, bool)   # path, image, skipped


CACHE_DIR = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
                         "PictureViewer", "thumbs")


def cache_file(path):
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = hashlib.sha1(f"{path}|{st.st_mtime_ns}|{st.st_size}|{THUMB_PX}".encode()).hexdigest()
    return os.path.join(CACHE_DIR, key[:2], key + ".jpg")


def make_thumb(path):
    cf = cache_file(path)
    if cf and os.path.exists(cf):
        img = QImage(cf)
        if not img.isNull():
            return img
    ext = os.path.splitext(path)[1].lower()
    if ext in QT_EXTS:
        r = QImageReader(path)
        r.setAutoTransform(True)
        sz = r.size()
        if sz.isValid():
            r.setScaledSize(sz.scaled(THUMB_PX, THUMB_PX, Qt.KeepAspectRatio))
        img = r.read()
    else:
        img = load_image(path, max_size=THUMB_PX)
    if not img.isNull() and max(img.width(), img.height()) > THUMB_PX:
        img = img.scaled(THUMB_PX, THUMB_PX, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    if cf and not img.isNull():
        try:
            os.makedirs(os.path.dirname(cf), exist_ok=True)
            img.save(cf, "JPG", 85)
        except OSError:
            pass
    return img


class ThumbLoader:
    """Thumbnail scheduler. Holds only what is wanted *right now*: every scroll replaces the
    to-do list, so slow decodes never pile up stale work, and workers always take the
    thumbnail closest to the middle of the strip first."""

    def __init__(self, signals, threads):
        self.signals = signals
        self.lock = threading.Lock()
        self.todo = {}        # path -> row
        self.inflight = set()
        self.center = 0
        self.workers = 0
        self.max_workers = threads
        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(threads)

    def request(self, wanted, center):
        """wanted: {path: row}. Replaces the to-do list; running decodes are left alone."""
        with self.lock:
            self.center = center
            self.todo = {p: r for p, r in wanted.items() if p not in self.inflight}
            n_new = min(len(self.todo), self.max_workers - self.workers)
            self.workers += max(0, n_new)
        for _ in range(max(0, n_new)):
            self.pool.start(_ThumbWorker(self))

    def clear(self):
        with self.lock:
            self.todo = {}

    def take(self):
        with self.lock:
            if not self.todo:
                self.workers -= 1
                return None
            path = min(self.todo, key=lambda p: abs(self.todo[p] - self.center))
            del self.todo[path]
            self.inflight.add(path)
            return path

    def done(self, path):
        with self.lock:
            self.inflight.discard(path)


class _ThumbWorker(QRunnable):
    def __init__(self, loader):
        super().__init__()
        self.loader = loader

    def run(self):
        ld = self.loader
        while True:
            path = ld.take()
            if path is None:
                return
            try:
                img = make_thumb(path)
            except Exception:
                img = QImage()
            ld.done(path)
            ld.signals.thumb.emit(path, img, False)


def load_for_faces(path):
    """BGR numpy array (EXIF-rotated, downscaled for speed) or None."""
    ext = os.path.splitext(path)[1].lower()
    if ext in QT_EXTS:
        r = QImageReader(path)
        r.setAutoTransform(True)
        sz = r.size()
        if sz.isValid() and max(sz.width(), sz.height()) > fc.MAX_SIDE:
            r.setScaledSize(sz.scaled(fc.MAX_SIDE, fc.MAX_SIDE, Qt.KeepAspectRatio))
        img = r.read()
    else:
        img = load_image(path, max_size=fc.MAX_SIDE)
    if img.isNull():
        return None
    img = img.convertToFormat(QImage.Format_RGB888)
    w, h, bpl = img.width(), img.height(), img.bytesPerLine()
    arr = np.frombuffer(img.constBits(), np.uint8).reshape(h, bpl)[:, :w * 3].reshape(h, w, 3)
    return np.ascontiguousarray(arr[:, :, ::-1])  # RGB -> BGR


class FaceSignals(QObject):
    progress = Signal(int, int)
    finished = Signal(bool)   # cancelled?


class FaceScanJob(QRunnable):
    def __init__(self, paths, store, signals):
        super().__init__()
        self.paths, self.store, self.signals = paths, store, signals
        self.cancelled = False

    def run(self):
        try:
            engine = fc.FaceEngine()
            total = len(self.paths)
            for i, p in enumerate(self.paths, 1):
                if self.cancelled:
                    break
                try:
                    st = os.stat(p)
                    key = (int(st.st_mtime), st.st_size)
                    if not self.store.is_current(p, *key):
                        bgr = load_for_faces(p)
                        found = engine.analyze(bgr) if bgr is not None else []
                        self.store.save(p, *key, found)
                except Exception:
                    pass
                if i % 3 == 0 or i == total:
                    self.signals.progress.emit(i, total)
            self.store.regroup()
        finally:
            self.signals.finished.emit(self.cancelled)


class ImageJob(QRunnable):
    def __init__(self, path, state, signals, prefetch=False):
        super().__init__()
        self.path, self.state, self.signals, self.prefetch = path, state, signals, prefetch

    def run(self):
        # Skip stale requests (user already moved on); prefetches always run.
        if not self.prefetch and self.state.path != self.path:
            self.signals.image.emit(self.path, QImage(), True)
            return
        self.signals.image.emit(self.path, load_image(self.path), False)


class ThumbModel(QAbstractListModel):
    """Virtual model: only visible thumbnails are ever decoded or held in memory."""

    def __init__(self):
        super().__init__()
        self.files = []
        self.cache = OrderedDict()
        self.blank = QPixmap(THUMB, THUMB)
        self.blank.fill(Qt.darkGray)
        self.rows = {}

    def set_files(self, files):
        self.beginResetModel()
        self.files = files
        self.cache.clear()
        self.rows = {p: i for i, p in enumerate(files)}
        self.endResetModel()

    def remove_row(self, row):
        self.beginRemoveRows(QModelIndex(), row, row)
        p = self.files.pop(row)
        self.cache.pop(p, None)
        self.rows = {q: i for i, q in enumerate(self.files)}
        self.endRemoveRows()

    def put(self, path, pix):
        self.cache[path] = pix
        while len(self.cache) > THUMB_CACHE:
            self.cache.popitem(last=False)
        row = self.rows.get(path)
        if row is not None:
            i = self.index(row)
            self.dataChanged.emit(i, i, [Qt.DecorationRole])

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.files)

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DecorationRole:
            return self.cache.get(self.files[index.row()], self.blank)
        if role == Qt.ToolTipRole:
            return os.path.basename(self.files[index.row()])
        if role == Qt.SizeHintRole:
            return QSize(CELL, CELL)
        return None


class ImageView(QGraphicsView):
    zoomChanged = Signal(float)
    faceClicked = Signal(int)
    faceMenu = Signal(int, QPoint)

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.item = QGraphicsPixmapItem()
        self.scene().addItem(self.item)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setBackgroundBrush(Qt.black)
        self.setFrameShape(QGraphicsView.NoFrame)
        self.fit_mode = True
        self.face_items = []
        self._press = None

    def set_faces(self, faces):
        """faces: dicts with fractional x,y,w,h plus id, label, color. Drawn on top of the picture."""
        self.clear_faces()
        pm = self.item.pixmap()
        if pm.isNull():
            return
        W, H = pm.width(), pm.height()
        for f in faces:
            r = QGraphicsRectItem(QRectF(f["x"] * W, f["y"] * H, f["w"] * W, f["h"] * H), self.item)
            pen = QPen(QColor(f["color"]), 2)
            pen.setCosmetic(True)
            r.setPen(pen)
            r.setData(0, f["id"])
            self.face_items.append(r)
            if f["label"]:
                t = QGraphicsSimpleTextItem(f["label"], r)
                t.setBrush(QBrush(QColor(f["color"])))
                t.setFlag(t.GraphicsItemFlag.ItemIgnoresTransformations)
                t.setPos(f["x"] * W, (f["y"] + f["h"]) * H)
                t.setData(0, f["id"])

    def clear_faces(self):
        for r in self.face_items:
            if r.scene() is not None:
                self.scene().removeItem(r)
        self.face_items = []

    def face_at(self, pos):
        for it in self.items(pos):
            fid = it.data(0)
            if fid is not None:
                return fid
        return None

    def mousePressEvent(self, e):
        self._press = e.position().toPoint()
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        if (self._press is not None and e.button() == Qt.LeftButton
                and (e.position().toPoint() - self._press).manhattanLength() < 4):
            fid = self.face_at(e.position().toPoint())
            if fid is not None:
                self.faceClicked.emit(fid)

    def contextMenuEvent(self, e):
        fid = self.face_at(e.pos())
        if fid is not None:
            self.faceMenu.emit(fid, e.globalPos())

    def set_image(self, pix, keep_zoom=False):
        self.clear_faces()
        self.item.setPixmap(pix)
        self.scene().setSceneRect(self.item.boundingRect())
        if not keep_zoom:
            self.fit()

    def scale_now(self):
        return self.transform().m11()

    def update_quality(self):
        # Smooth filtering only pays off when shrinking; nearest is much faster when zoomed in.
        self.setRenderHint(QPainter.SmoothPixmapTransform, self.scale_now() < 1)

    def fit(self):
        self.fit_mode = True
        if self.item.pixmap().isNull():
            return
        self.resetTransform()
        self.fitInView(self.item, Qt.KeepAspectRatio)
        # Don't upscale small images past 100% when fitting.
        if self.scale_now() > 1:
            self.resetTransform()
        self.update_quality()
        self.zoomChanged.emit(self.scale_now())

    def actual(self):
        self.fit_mode = False
        self.resetTransform()
        self.update_quality()
        self.zoomChanged.emit(1.0)

    def zoom_by(self, factor):
        self.fit_mode = False
        new = self.scale_now() * factor
        if 0.02 <= new <= 50:
            self.scale(factor, factor)
            self.update_quality()
            self.zoomChanged.emit(self.scale_now())

    def wheelEvent(self, e):
        self.zoom_by(1.25 if e.angleDelta().y() > 0 else 1 / 1.25)

    def mouseDoubleClickEvent(self, e):
        if self.fit_mode:
            self.actual()
        else:
            self.fit()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.fit_mode:
            self.fit()


class Viewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Picture Viewer")
        self.resize(1200, 800)
        self.setAcceptDrops(True)
        self.folder = None
        self.search_title = None
        self.files = []
        self.index = -1
        self.image_pool = QThreadPool()  # separate so the main picture never waits on thumbnails
        self.image_pool.setMaxThreadCount(2)
        self.img_cache = OrderedDict()
        self.inflight = set()
        self.shown_path = None
        self.load_seq = 0
        self.signals = Signals()
        self.signals.thumb.connect(self.on_thumb)
        self.loader = ThumbLoader(self.signals, max(2, min(4, (os.cpu_count() or 4) // 2)))
        self.loader.pool.setThreadPriority(QThread.LowPriority)  # never starve the UI thread
        self.signals.image.connect(self.on_image)
        self.recursive = False

        self.view = ImageView()
        self.view.zoomChanged.connect(self.on_zoom)
        self.model = ThumbModel()
        self.strip = QListView()
        self.strip.setModel(self.model)
        self.strip.setViewMode(QListView.ListMode)
        self.strip.setFlow(QListView.LeftToRight)
        self.strip.setWrapping(False)
        self.strip.setMovement(QListView.Static)
        self.strip.setSelectionMode(QListView.SingleSelection)
        self.strip.setFrameShape(QListView.NoFrame)
        self.strip.setUniformItemSizes(True)
        self.strip.setIconSize(QSize(THUMB, THUMB))
        self.strip.setGridSize(QSize(CELL, CELL))
        self.strip.setFixedHeight(CELL + 30)
        self.strip.setHorizontalScrollMode(QListView.ScrollPerPixel)
        self.strip.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.strip.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        pal = self.strip.palette()
        pal.setColor(QPalette.Base, QColor("#202020"))
        pal.setColor(QPalette.Highlight, QColor("#3b78d8"))
        self.strip.setPalette(pal)
        # Style sheet only on the scrollbar: a sheet on the item view slows every repaint.
        self.strip.horizontalScrollBar().setStyleSheet(
            "QScrollBar:horizontal{background:#303030;height:18px;margin:0;}"
            "QScrollBar::handle:horizontal{background:#8a8a8a;border-radius:6px;"
            "min-width:40px;margin:2px;}"
            "QScrollBar::handle:horizontal:hover{background:#b0b0b0;}"
            "QScrollBar::add-line:horizontal,QScrollBar::sub-line:horizontal{width:0;}")
        self.strip.selectionModel().currentChanged.connect(lambda cur, _: self.show_index(cur.row()))
        self.thumb_timer = QTimer(self)
        self.thumb_timer.setSingleShot(True)
        self.thumb_timer.setInterval(30)
        self.thumb_timer.timeout.connect(self.request_visible)
        self.strip.horizontalScrollBar().valueChanged.connect(self.queue_thumbs)

        split = QSplitter(Qt.Vertical)
        split.addWidget(self.view)
        split.addWidget(self.strip)
        split.setStretchFactor(0, 1)
        split.setCollapsible(1, False)
        self.setCentralWidget(split)
        self.splitter = split

        self.zoom_label = QLabel(" 100% ")
        self.status = self.statusBar()
        self.status.addPermanentWidget(self.zoom_label)

        self.slide_timer = QTimer(self)
        self.slide_timer.timeout.connect(self.slide_next)
        self.build_actions()

    # ---- UI ----
    def act(self, text, slot, keys=None, checkable=False):
        a = QAction(text, self)
        a.setCheckable(checkable)
        if keys:
            a.setShortcuts([QKeySequence(k) for k in (keys if isinstance(keys, list) else [keys])])
        a.triggered.connect(slot)
        self.addAction(a)
        return a

    def build_actions(self):
        open_a = self.act("Open Folder…", self.choose_folder, "Ctrl+O")
        rec_a = self.act("Include Subfolders", self.toggle_recursive, checkable=True)
        zin = self.act("Zoom In (+)", lambda: self.view.zoom_by(1.25), ["+", "=", "Ctrl+="])
        zout = self.act("Zoom Out (-)", lambda: self.view.zoom_by(0.8), ["-", "Ctrl+-"])
        fit = self.act("Fit", self.view.fit, "F")
        actual = self.act("100%", self.view.actual, ["1", "Ctrl+0"])
        prev = self.act("◀ Prev", lambda: self.step(-1), ["Left", "PgUp"])
        nxt = self.act("Next ▶", lambda: self.step(1), ["Right", "PgDown", "Space"])
        self.act("First", lambda: self.goto(0), "Home")
        self.act("Last", lambda: self.goto(len(self.files) - 1), "End")
        rl = self.act("⟲ Rotate Left", lambda: self.rotate(-90), "L")
        rr = self.act("⟳ Rotate Right", lambda: self.rotate(90), "R")
        dele = self.act("Delete", self.delete_current, "Delete")
        openw = self.act("Open With…", self.open_with, "Ctrl+E")
        about = self.act("About", self.show_about, "F1")
        self.slide_a = self.act("Slideshow", self.toggle_slideshow, "F5", checkable=True)
        self.full_a = self.act("Fullscreen", self.toggle_fullscreen, ["F11", "Return"],
                               checkable=True)
        self.act("Exit Fullscreen", lambda: self.full_a.isChecked() and self.toggle_fullscreen(),
                 "Escape")

        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.addToolBar(tb)
        for a in (open_a, rec_a, None, prev, nxt, None, zout, zin, fit, actual, None,
                  rl, rr, None, self.slide_a, self.full_a, None, openw, dele, None, about):
            tb.addSeparator() if a is None else tb.addAction(a)
        self.toolbar = tb

        self.build_slideshow_menu()
        self.build_people_menu()

    # ---- Faces ----
    def build_people_menu(self):
        self.store = fc.FaceStore() if fc else None
        self.face_job = None
        self.face_signals = None
        menu = self.menuBar().addMenu("&People")
        if not self.store:
            menu.addAction("Face recognition unavailable (OpenCV or models missing)").setEnabled(False)
            self.boxes_a = QAction(self)
            return
        self.face_signals = FaceSignals()
        self.face_signals.progress.connect(lambda i, n: self.status.showMessage(
            f"Scanning faces… {i} / {n}"))
        self.face_signals.finished.connect(self.on_scan_finished)
        self.face_pool = QThreadPool()
        self.face_pool.setMaxThreadCount(1)
        self.face_pool.setThreadPriority(QThread.LowPriority)
        self.scan_a = menu.addAction("&Scan this folder for faces", self.scan_faces)
        self.cancel_a = menu.addAction("Cancel scan", self.cancel_scan)
        self.cancel_a.setEnabled(False)
        menu.addSeparator()
        self.boxes_a = QAction("Show face &boxes", self, checkable=True)
        self.boxes_a.setShortcut(QKeySequence("F2"))
        self.boxes_a.toggled.connect(lambda _: self.refresh_faces())
        menu.addAction(self.boxes_a)
        menu.addAction("&People…", self.show_people)
        menu.addAction("&Find person…", self.find_person)
        self.back_a = menu.addAction("&Back to folder", self.back_to_folder)
        self.back_a.setEnabled(False)
        menu.addSeparator()
        menu.addAction("&Delete all face data…", self.clear_faces_data)
        self.view.faceClicked.connect(self.name_face_dialog)
        self.view.faceMenu.connect(self.face_context_menu)

    def scan_faces(self):
        if not self.files or self.face_job:
            return
        self.face_job = FaceScanJob(list(self.files), self.store, self.face_signals)
        self.face_job.setAutoDelete(False)
        self.scan_a.setEnabled(False)
        self.cancel_a.setEnabled(True)
        self.face_pool.start(self.face_job)

    def cancel_scan(self):
        if self.face_job:
            self.face_job.cancelled = True

    def on_scan_finished(self, cancelled):
        self.face_job = None
        self.scan_a.setEnabled(True)
        self.cancel_a.setEnabled(False)
        n_photos, n_faces, _ = self.store.stats()
        self.status.showMessage(("Scan cancelled. " if cancelled else "Scan complete. ")
                                + f"{n_faces} faces in {n_photos} photos.")
        self.refresh_faces()

    def refresh_faces(self):
        """Redraw the name boxes for the picture on screen."""
        if not self.store or not self.boxes_a.isChecked() or not self.shown_path:
            self.view.clear_faces()
            return
        boxes = []
        for f in self.store.faces_in(self.shown_path):
            if f["name"]:
                label, color = f["name"], "#41d36b"
            elif f["suggest"]:
                label, color = f"{f['suggest'][1]}?", "#ffb02e"
            else:
                label, color = "", "#ffffff"
            boxes.append(dict(id=f["id"], x=f["x"], y=f["y"], w=f["w"], h=f["h"],
                              label=label, color=color))
        self.view.set_faces(boxes)

    def faces_changed(self):
        self.refresh_faces()

    def _face(self, face_id):
        return next((f for f in self.store.faces_in(self.shown_path) if f["id"] == face_id), None)

    def name_face_dialog(self, face_id):
        f = self._face(face_id)
        if not f:
            return
        names = self.store.names()
        default = f["name"] or (f["suggest"][1] if f["suggest"] else "")
        name, ok = QInputDialog.getItem(self, "Who is this?", "Name:", names,
                                        names.index(default) if default in names else 0, True)
        if ok and name.strip():
            self.store.name_face(face_id, name)
            self.store.regroup()
            self.refresh_faces()

    def face_context_menu(self, face_id, pos):
        f = self._face(face_id)
        if not f:
            return
        m = QMenu(self)
        m.addAction("Name this face…", lambda: self.name_face_dialog(face_id))
        if f["suggest"] and not f["name"]:
            n = f["suggest"][1]
            m.addAction(f"Yes, this is {n}", lambda: (self.store.name_face(face_id, n),
                                                       self.refresh_faces()))
        if f["name"]:
            m.addAction("Remove name", lambda: (self.store.unname_face(face_id), self.refresh_faces()))
        m.addAction("Not a face", lambda: (self.store.ignore_face(face_id), self.refresh_faces()))
        m.addSeparator()
        m.addAction("Find more photos of this person",
                    lambda: self.show_results(self.store.paths_like(f["emb"]), "similar faces"))
        m.exec(pos)

    def show_people(self):
        if self.face_job:
            self.status.showMessage("A scan is running; the list will be incomplete until it finishes.")
        self.store.regroup()
        PeopleDialog(self.store, self).exec()

    def find_person(self):
        names = self.store.names()
        if not names:
            QMessageBox.information(self, "Find person", "No one is named yet. Scan a folder, "
                                    "then name faces with People… or by clicking a face box.")
            return
        name, ok = QInputDialog.getItem(self, "Find person", "Name:", names, 0, True)
        if ok and name.strip():
            self.show_results(self.store.paths_for_name(name.strip()), name.strip())

    def show_results(self, paths, title):
        paths = [p for p in paths if os.path.isfile(p)]
        if not paths:
            QMessageBox.information(self, "Search", f"No photos found for {title}.")
            return
        self.search_title = f"Search: {title}"
        self.set_file_list(paths)

    def back_to_folder(self):
        if self.folder:
            self.open_path(self.folder, select=self.current_path())

    def clear_faces_data(self):
        if QMessageBox.question(self, "Delete face data",
                                "Delete all saved faces, groups and names? Photos are not touched."
                                ) == QMessageBox.Yes:
            self.cancel_scan()
            self.store.clear_all()
            self.refresh_faces()
            self.status.showMessage("Face data deleted.")

    def build_slideshow_menu(self):
        cfg = QSettings("PictureViewer", "PictureViewer")
        try:
            self.slide_secs = min(10, max(1, int(cfg.value("slideshow/delay", 3))))
        except (TypeError, ValueError):
            self.slide_secs = 3
        self.slide_random = str(cfg.value("slideshow/random", "false")).lower() == "true"
        self.shuffle_bag = []

        menu = self.menuBar().addMenu("&Slideshow")
        menu.addAction(self.slide_a)
        delay = menu.addMenu("&Delay")
        group = QActionGroup(self)
        group.setExclusive(True)
        for n in range(1, 11):
            a = QAction(f"{n} second{'s' if n > 1 else ''}", self, checkable=True)
            a.setChecked(n == self.slide_secs)
            a.triggered.connect(lambda _=False, n=n: self.set_slide_delay(n))
            group.addAction(a)
            delay.addAction(a)
        rnd = QAction("&Random order", self, checkable=True)
        rnd.setChecked(self.slide_random)
        rnd.triggered.connect(self.set_slide_random)
        menu.addAction(rnd)

    def set_slide_delay(self, secs):
        self.slide_secs = secs
        QSettings("PictureViewer", "PictureViewer").setValue("slideshow/delay", secs)
        if self.slide_timer.isActive():
            self.slide_timer.start(secs * 1000)

    def set_slide_random(self, on):
        self.slide_random = on
        self.shuffle_bag = []
        QSettings("PictureViewer", "PictureViewer").setValue("slideshow/random", on)

    def slide_next(self):
        if not self.files:
            return
        if not self.slide_random:
            self.step(1, wrap=True)
            return
        # Shuffled bag: every picture is shown once before any repeats.
        if not self.shuffle_bag:
            self.shuffle_bag = [i for i in range(len(self.files)) if i != self.index]
            random.shuffle(self.shuffle_bag)
        if self.shuffle_bag:
            self.goto(self.shuffle_bag.pop())

    def show_about(self):
        QMessageBox.about(
            self, f"About {APP_NAME}",
            f"<h3>{APP_NAME}</h3>"
            f"<p>Version {VERSION}<br>Build date: {BUILD_DATE}</p>"
            "<p>A fast picture viewer with a thumbnail strip, zoom and pan.<br>"
            "Supports JPG, PNG, GIF, BMP, WebP, TIFF, HEIC and RAW.</p>")

    def on_zoom(self, z):
        self.zoom_label.setText(f" {z * 100:.0f}% ")

    # ---- Folder / files ----
    def choose_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Choose a picture folder", self.folder or "")
        if d:
            self.open_path(d)

    def toggle_recursive(self, on):
        self.recursive = on
        if self.folder:
            self.open_path(self.folder, select=self.current_path())

    def scan(self, folder):
        out = []
        stack = [folder]
        while stack:
            d = stack.pop()
            try:
                with os.scandir(d) as it:
                    for e in it:  # DirEntry caches type info: no extra stat per file
                        if e.is_dir(follow_symlinks=False):
                            if self.recursive:
                                stack.append(e.path)
                        elif os.path.splitext(e.name)[1].lower() in ALL_EXTS:
                            out.append(e.path)
            except OSError:
                pass
        n = len(folder) + 1
        out.sort(key=lambda p: p[n:].lower())
        return out

    def open_path(self, path, select=None):
        path = os.path.abspath(path)
        if os.path.isfile(path):
            select, path = path, os.path.dirname(path)
        if not os.path.isdir(path):
            return
        self.folder = path
        self.search_title = None
        self.set_file_list(self.scan(path), select, f"No pictures in {path}")

    def set_file_list(self, files, select=None, empty_msg="Nothing to show"):
        self.img_cache.clear()
        self.shown_path = None
        self.loader.clear()
        self.shuffle_bag = []
        self.files = files
        self.model.set_files(self.files)
        if hasattr(self, "back_a"):
            self.back_a.setEnabled(bool(self.search_title))
        if not self.files:
            self.index = -1
            self.view.set_image(QPixmap())
            self.setWindowTitle("Picture Viewer")
            self.status.showMessage(empty_msg)
            return
        start = self.files.index(select) if select in self.files else 0
        self.strip.setCurrentIndex(self.model.index(start))

    def queue_thumbs(self, _=None):
        if not self.thumb_timer.isActive():  # throttle: keep loading while the slider is dragged
            self.thumb_timer.start()

    def request_visible(self):
        """Queue thumbnail decoding for the visible cells (plus a margin), nearest first."""
        n = len(self.files)
        if not n:
            return
        first = self.strip.horizontalScrollBar().value() // CELL
        count = self.strip.viewport().width() // CELL + 2
        lo, hi = max(0, first - count // 2), min(n - 1, first + count + count // 2)
        mid = first + count // 2
        wanted = {self.files[r]: r for r in range(lo, hi + 1) if self.files[r] not in self.model.cache}
        self.loader.request(wanted, mid)

    def on_thumb(self, path, img, skipped):
        pix = QPixmap.fromImage(img) if not img.isNull() else self.model.blank
        self.model.put(path, pix)

    def on_image(self, path, img, skipped):
        self.inflight.discard(path)
        if skipped:
            return
        if not img.isNull():
            self.img_cache[path] = img
            while len(self.img_cache) > IMG_CACHE:
                self.img_cache.popitem(last=False)
        if path == self.current_path() and path != self.shown_path:
            self.display(path, img)

    def display(self, path, img):
        self.shown_path = path
        if img.isNull():
            self.view.set_image(QPixmap())
            self.status.showMessage(f"Can't open {os.path.basename(path)}")
            return
        self.view.set_image(QPixmap.fromImage(img), keep_zoom=not self.view.fit_mode)
        tag = f"   [{self.search_title}]" if self.search_title else ""
        self.status.showMessage(f"{os.path.basename(path)}   {img.width()}×{img.height()}   "
                                f"{self.index + 1} / {len(self.files)}{tag}")
        self.refresh_faces()

    def prefetch(self, i):
        for j in (i + 1, i - 1, i + 2):
            if 0 <= j < len(self.files):
                p = self.files[j]
                if p not in self.img_cache and p not in self.inflight:
                    self.inflight.add(p)
                    self.image_pool.start(ImageJob(p, self, self.signals, prefetch=True), -(1 << 20))

    # ---- Navigation ----
    def current_path(self):
        return self.files[self.index] if 0 <= self.index < len(self.files) else None

    @property
    def path(self):  # read by ImageJob worker threads to detect stale requests
        return self.current_path()

    def show_index(self, i):
        if not (0 <= i < len(self.files)):
            return
        self.index = i
        path = self.files[i]
        self.setWindowTitle(f"{os.path.basename(path)} — Picture Viewer")
        if path in self.img_cache:
            self.img_cache.move_to_end(path)
            self.display(path, self.img_cache[path])
        else:
            self.status.showMessage(f"Loading {os.path.basename(path)}…   {i + 1} / {len(self.files)}")
            if path not in self.inflight:
                self.inflight.add(path)
                self.load_seq += 1
                self.image_pool.start(ImageJob(path, self, self.signals), self.load_seq)
        self.prefetch_timer_start(i)
        if self.strip.currentIndex().row() != i:
            self.strip.setCurrentIndex(self.model.index(i))
        self.strip.scrollTo(self.model.index(i), QListView.PositionAtCenter)
        self.queue_thumbs()

    def prefetch_timer_start(self, i):
        # Wait briefly so rapid key-repeat doesn't queue prefetches for every picture passed.
        if not hasattr(self, "_pf"):
            self._pf = QTimer(self)
            self._pf.setSingleShot(True)
            self._pf.setInterval(120)
            self._pf.timeout.connect(lambda: self.prefetch(self.index))
        self._pf.start()

    def goto(self, i):
        if self.files:
            self.show_index(max(0, min(i, len(self.files) - 1)))

    def step(self, d, wrap=False):
        if not self.files:
            return
        n = self.index + d
        if wrap:
            n %= len(self.files)
        self.goto(n)

    # ---- Actions ----
    def rotate(self, deg):
        pm = self.view.item.pixmap()
        if pm.isNull():
            return
        self.view.set_image(pm.transformed(QTransform().rotate(deg), Qt.SmoothTransformation),
                            keep_zoom=not self.view.fit_mode)  # boxes no longer line up: cleared

    def delete_current(self):
        p = self.current_path()
        if not p:
            return
        if QMessageBox.question(self, "Delete", f"Move to Recycle Bin?\n\n{os.path.basename(p)}"
                                ) != QMessageBox.Yes:
            return
        try:
            if send2trash:
                send2trash(os.path.normpath(p))
            else:
                raise RuntimeError("send2trash is not installed")
        except Exception as e:
            QMessageBox.warning(self, "Delete failed", str(e))
            return
        row = self.index
        self.img_cache.pop(p, None)
        self.shown_path = None
        self.model.remove_row(row)  # shares its list with self.files
        if self.files:
            self.strip.setCurrentIndex(self.model.index(min(row, len(self.files) - 1)))
            self.show_index(min(row, len(self.files) - 1))
        else:
            self.index = -1
            self.view.set_image(QPixmap())

    def open_with(self):
        p = self.current_path()
        if p:
            # Windows "Open with" chooser.
            subprocess.Popen(["rundll32.exe", "shell32.dll,OpenAs_RunDLL", os.path.normpath(p)])

    def toggle_slideshow(self, on=None):
        on = self.slide_a.isChecked() if on is None else on
        if on:
            self.slide_timer.start(self.slide_secs * 1000)
        else:
            self.slide_timer.stop()

    def toggle_fullscreen(self, _=None):
        full = not self.isFullScreen()
        self.full_a.setChecked(full)
        self.toolbar.setVisible(not full)
        self.menuBar().setVisible(not full)
        self.statusBar().setVisible(not full)
        self.showFullScreen() if full else self.showNormal()

    # ---- Drag and drop ----
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if urls:
            self.open_path(urls[0].toLocalFile())


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Picture Viewer")
    w = Viewer()
    w.show()
    if len(sys.argv) > 1:
        w.open_path(sys.argv[1])
    else:
        QTimer.singleShot(0, w.choose_folder)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
