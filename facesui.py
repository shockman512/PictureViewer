"""People dialog for the face feature."""
import os

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QInputDialog, QLabel, QListView, QListWidget,
                               QListWidgetItem, QMessageBox, QPushButton, QVBoxLayout)


def pix(jpg):
    p = QPixmap()
    p.loadFromData(jpg or b"")
    return p


class PeopleDialog(QDialog):
    def __init__(self, store, owner):
        super().__init__(owner)
        self.store = store
        self.owner = owner
        self.setWindowTitle("People")
        self.resize(820, 520)
        self.groups = []
        self.base_hint = ""

        self.glist = QListWidget()
        self.glist.setIconSize(QSize(48, 48))
        self.glist.setFixedWidth(280)
        self.glist.currentRowChanged.connect(self.show_group)
        self.faces = QListWidget()
        self.faces.setViewMode(QListView.IconMode)
        self.faces.setIconSize(QSize(80, 80))
        self.faces.setGridSize(QSize(92, 92))
        self.faces.setResizeMode(QListView.Adjust)
        self.faces.setMovement(QListView.Static)
        self.faces.setSelectionMode(QListWidget.ExtendedSelection)
        self.faces.itemSelectionChanged.connect(self.update_buttons)
        self.faces.itemDoubleClicked.connect(self.open_photo)
        self.hint = QLabel()
        self.hint.setWordWrap(True)

        self.name_b = name_b = QPushButton("Name group…")
        name_b.clicked.connect(self.name_group)
        self.unname_b = QPushButton("Remove name from selected")
        self.unname_b.clicked.connect(self.unname_selected)
        self.accept_b = QPushButton("Accept suggestion")
        self.accept_b.clicked.connect(self.accept_suggestion)
        self.ign_b = ign_b = QPushButton("Ignore whole group")
        ign_b.clicked.connect(self.ignore_group)
        self.restore_b = QPushButton("Restore selected")
        self.restore_b.setToolTip("Bring hidden faces back. Select some, or leave none selected to restore all.")
        self.restore_b.clicked.connect(self.restore)
        regroup_b = QPushButton("Regroup")
        regroup_b.clicked.connect(lambda: (self.store.regroup(), self.reload()))
        close_b = QPushButton("Close")
        close_b.clicked.connect(self.accept)

        right = QVBoxLayout()
        right.addWidget(self.hint)
        right.addWidget(self.faces, 1)
        btns = QHBoxLayout()
        self.main_btns = (name_b, self.accept_b, ign_b)
        for b in (name_b, self.accept_b, ign_b, self.unname_b, self.restore_b, regroup_b):
            btns.addWidget(b)
        btns.addStretch(1)
        btns.addWidget(close_b)
        right.addLayout(btns)
        lay = QHBoxLayout(self)
        lay.addWidget(self.glist)
        lay.addLayout(right, 1)
        self.reload()

    def reload(self, keep=None):
        self.groups = self.store.groups()
        self.glist.blockSignals(True)
        self.glist.clear()
        n_unnamed = 0
        for g in self.groups:
            if g["kind"] == "person":
                label = f"{g['name']}  ({g['count']})"
            else:
                n_unnamed += 1
                label = f"Unnamed {n_unnamed}  ({g['count']})"
                if g["suggest"]:
                    label += f"\nlooks like {g['suggest']}?"
            g["label"] = label
            self.glist.addItem(QListWidgetItem(
                QIcon(pix(g["thumbs"][0] if g["thumbs"] else b"")), label))
        n_ign = self.store.ignored_count()
        if n_ign:
            self.groups.append(dict(kind="ignored", id=0, name=None, count=n_ign, thumbs=[],
                                    suggest=None, label=f"Ignored faces  ({n_ign})"))
            self.glist.addItem(QListWidgetItem(self.groups[-1]["label"]))
        self.glist.blockSignals(False)
        if self.groups:
            row = 0
            if keep:
                row = next((i for i, g in enumerate(self.groups)
                            if (g["kind"], g["id"]) == keep), 0)
            self.glist.setCurrentRow(row)
            self.show_group(row)
        else:
            self.faces.clear()
            self.hint.setText("No faces yet. Use People > Scan this folder first.")

    def current(self):
        r = self.glist.currentRow()
        return self.groups[r] if 0 <= r < len(self.groups) else None

    def show_group(self, _row):
        g = self.current()
        self.faces.clear()
        is_ign = bool(g) and g["kind"] == "ignored"
        self.restore_b.setVisible(is_ign)
        for b in self.main_btns:
            b.setVisible(not is_ign)
        if not g:
            return
        if is_ign:
            self.unname_b.setVisible(False)
            for fid, t in self.store.ignored_faces():
                it = QListWidgetItem(QIcon(pix(t)), "")
                it.setData(Qt.UserRole, fid)
                self.faces.addItem(it)
            self.hint.setText(f"{g['count']} hidden faces. Select the ones to bring back and press "
                              "Restore selected (select none to restore all).")
            return
        shown = self.store.group_faces(g["kind"], g["id"])
        for fid, t in shown:
            it = QListWidgetItem(QIcon(pix(t)), "")
            it.setData(Qt.UserRole, fid)
            self.faces.addItem(it)
        extra = f" Showing the first {len(shown)}." if g["count"] > len(shown) else ""
        self.base_hint = (g["label"].replace("\n", " - ") + "." + extra +
                          " Click faces to select them (Ctrl/Shift for several); double-click to open the photo.")
        self.accept_b.setEnabled(bool(g["suggest"]))
        self.update_buttons()

    def selected_ids(self):
        return [it.data(Qt.UserRole) for it in self.faces.selectedItems()]

    def update_buttons(self):
        g = self.current()
        if not g or g["kind"] == "ignored":
            self.unname_b.setVisible(False)
            return
        n = len(self.selected_ids())
        self.name_b.setText(f"Name {n} selected…" if n else "Name group…")
        self.ign_b.setText(f"Ignore {n} selected" if n else "Ignore whole group")
        self.unname_b.setVisible(g["kind"] == "person" and n > 0)
        self.accept_b.setText("Accept suggestion" + (" (whole group)" if n else ""))
        self.hint.setText(self.base_hint)

    def open_photo(self, item):
        path = self.store.face_path(item.data(Qt.UserRole))
        if path and os.path.exists(path):
            self.owner.open_path(os.path.dirname(path), select=path)
            self.accept()

    def unname_selected(self):
        ids = self.selected_ids()
        if ids:
            self.store.unname_faces(ids)
            self._changed()

    def _changed(self):
        self.store.regroup()
        self.reload()
        self.owner.faces_changed()

    def restore(self):
        ids = [it.data(Qt.UserRole) for it in self.faces.selectedItems()]
        self.store.unignore(ids or None)
        self._changed()

    def name_group(self):
        g = self.current()
        if not g or g['kind'] == 'ignored':
            return
        ids = self.selected_ids()
        names = self.store.names()
        default = (g["name"] or g["suggest"] or "") if not ids else ""
        name, ok = QInputDialog.getItem(self, "Name this person", "Name:", names,
                                        names.index(default) if default in names else 0, True)
        if ok and name.strip():
            if ids:
                self.store.name_faces(ids, name.strip())
            else:
                self.store.name_group(g["kind"], g["id"], name.strip())
            self._changed()

    def accept_suggestion(self):
        g = self.current()
        if g and g["suggest"]:
            self.store.name_group(g["kind"], g["id"], g["suggest"])
            self._changed()

    def ignore_group(self):
        g = self.current()
        if not g:
            return
        ids = self.selected_ids()
        what = f"{len(ids)} selected face(s)" if ids else "this whole group"
        if QMessageBox.question(self, "Ignore", f"Hide {what} (not real faces)?\n"
                                "You can bring them back from 'Ignored faces'.") != QMessageBox.Yes:
            return
        if ids:
            self.store.ignore_faces(ids)
        else:
            self.store.ignore_group(g["kind"], g["id"])
        self._changed()
