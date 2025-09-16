# Plugin: transform_rt
# Adds Tools ▸ Rotation/Translation… to interactively translate/rotate datasets
# and to load a .conf file with per-file transforms (bmesh lines).
#
# Assumptions about the host:
# - window.viewer3d holds a Plotter-like object (pyvista) and a datasets registry
#   at window.viewer3d._datasets: List[dict], where each record may contain:
#     - 'kind': 'points' | 'mesh'
#     - 'actor_points' or 'actor_mesh' (pyvista Actor or vtkActor)
#     - 'source_path' (absolute or relative), optional but recommended
#     - 'visible': bool
# - You already persist source_path when importing (recommended for matching .conf).
#
# We always apply transforms as Actor UserMatrix (non destructive).

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import os
import math

from PySide6 import QtCore, QtGui, QtWidgets

# -------------------------
# Progress helper (best-effort integration with MainWindow)
# -------------------------

class _ProgressHelper:
    """
    Best-effort progress integration:
    - Try MainWindow methods if they exist (progress_begin/progress_set/progress_end or _progress_* variants)
    - Else try a status-bar QProgressBar on `window.progressBar` or `window._progressBar`
    - Else fallback to a non-modal QProgressDialog.
    Use as:
        pg = _ProgressHelper(window, "Applying...", total=100)
        pg.start()
        pg.set(10)
        ...
        pg.finish()
    Or as context manager:
        with _ProgressHelper(window, "Doing X", total=n) as pg:
            for i in range(n): pg.step()
    """
    def __init__(self, window, text: str, total: int = 100):
        self.window = window
        self.text = text
        self.total = max(1, int(total))
        self._val = 0
        self._mode = None   # 'methods' | 'bar' | 'dialog'
        self._bar = None
        self._dlg = None

    # -- low-level probes --
    def _call(self, names, *args, **kwargs):
        for name in names:
            fn = getattr(self.window, name, None)
            if callable(fn):
                try:
                    return fn(*args, **kwargs)
                except Exception:
                    pass
        return None

    def start(self):
        # Try MainWindow explicit methods
        if self._call(("progress_begin", "_progress_begin"), self.text, self.total) is not None:
            self._mode = "methods"
            return self

        # Try to reuse an existing QProgressBar on the window
        for attr in ("progressBar", "_progressBar", "prgMain", "prgProgress"):
            bar = getattr(self.window, attr, None)
            if bar is not None:
                try:
                    bar.setRange(0, self.total)
                    bar.setValue(0)
                    # Some styles do not support setFormat; guard it.
                    if hasattr(bar, "setFormat"):
                        try:
                            bar.setFormat(self.text + " %p%")
                        except Exception:
                            pass
                    bar.setVisible(True)
                    self._bar = bar
                    self._mode = "bar"
                    return self
                except Exception:
                    pass

        # Fallback: progress dialog (non modal)
        try:
            dlg = QtWidgets.QProgressDialog(self.text, None, 0, self.total, self.window)
            dlg.setWindowTitle("Working…")
            dlg.setWindowModality(QtCore.Qt.NonModal)
            dlg.setAutoClose(False); dlg.setAutoReset(False)
            dlg.setValue(0)
            dlg.show()
            self._dlg = dlg
            self._mode = "dialog"
        except Exception:
            self._mode = None
        return self

    def set(self, value: int):
        self._val = max(0, min(self.total, int(value)))
        if self._mode == "methods":
            self._call(("progress_set", "_progress_set"), self._val)
        elif self._mode == "bar" and self._bar is not None:
            try:
                self._bar.setValue(self._val)
            except Exception:
                pass
        elif self._mode == "dialog" and self._dlg is not None:
            try:
                self._dlg.setValue(self._val)
            except Exception:
                pass

    def step(self, inc: int = 1):
        self.set(self._val + int(inc))

    def finish(self):
        if self._mode == "methods":
            self._call(("progress_end", "_progress_end"))
        elif self._mode == "bar" and self._bar is not None:
            try:
                self._bar.setVisible(False)
            except Exception:
                pass
        elif self._mode == "dialog" and self._dlg is not None:
            try:
                self._dlg.close()
            except Exception:
                pass
        self._mode = None
        self._bar = None
        self._dlg = None

    # Context manager sugar
    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.finish()
        # do not suppress exceptions
        return False

try:
    import numpy as np
except Exception:
    np = None

# -------------------------
# Math helpers
# -------------------------

def quat_to_mat4(qx: float, qy: float, qz: float, qw: float) -> List[List[float]]:
    """Convert quaternion (x, y, z, w) to a 3x3 rotation matrix embedded in 4x4."""
    # Normalize quaternion for safety
    norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if norm == 0:
        qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
    else:
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm

    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz

    m00 = 1.0 - 2.0*(yy + zz)
    m01 = 2.0*(xy - wz)
    m02 = 2.0*(xz + wy)

    m10 = 2.0*(xy + wz)
    m11 = 1.0 - 2.0*(xx + zz)
    m12 = 2.0*(yz - wx)

    m20 = 2.0*(xz - wy)
    m21 = 2.0*(yz + wx)
    m22 = 1.0 - 2.0*(xx + yy)

    return [
        [m00, m01, m02, 0.0],
        [m10, m11, m12, 0.0],
        [m20, m21, m22, 0.0],
        [0.0,  0.0,  0.0,  1.0],
    ]


def euler_deg_to_quat(rx_deg: float, ry_deg: float, rz_deg: float, order: str = "XYZ") -> Tuple[float, float, float, float]:
    """Convert Euler angles in degrees to quaternion (x, y, z, w).
    Uses intrinsic rotations and order 'XYZ' by default (tweak if needed)."""
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)

    # Basic XYZ -> quaternion composition
    # q = qx(rx) * qy(ry) * qz(rz)
    def qx(a): return (math.sin(a/2), 0.0, 0.0, math.cos(a/2))
    def qy(a): return (0.0, math.sin(a/2), 0.0, math.cos(a/2))
    def qz(a): return (0.0, 0.0, math.sin(a/2), math.cos(a/2))

    def qmul(q1, q2):
        x1,y1,z1,w1 = q1
        x2,y2,z2,w2 = q2
        return (
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        )

    seq = []
    for c,a in zip(order.upper(), (rx, ry, rz)):
        if c == "X": seq.append(qx(a))
        elif c == "Y": seq.append(qy(a))
        else: seq.append(qz(a))

    q = seq[0]
    q = qmul(q, seq[1])
    q = qmul(q, seq[2])
    return q  # (x, y, z, w)


def mat4_mul(A, B):
    """4x4 matrix multiply: return A @ B."""
    return [
        [
            A[r][0]*B[0][c] + A[r][1]*B[1][c] + A[r][2]*B[2][c] + A[r][3]*B[3][c]
            for c in range(4)
        ]
        for r in range(4)
    ]


def compose_trs_scale_pivot(
    tx: float, ty: float, tz: float,
    qx: float, qy: float, qz: float, qw: float,
    sx: float, sy: float, sz: float,
    px: float, py: float, pz: float,
) -> list[list[float]]:
    """
    Build a 4x4 matrix for: Translate * Tpivot * Rotate * Scale * T(-pivot).

    Order rationale:
      - Scale happens in local object space first
      - Rotate around the same pivot
      - Then translate in world
    """
    # Rotation (4x4)
    R = quat_to_mat4(qx, qy, qz, qw)

    # Scale (4x4)
    S = [
        [float(sx), 0.0,       0.0,       0.0],
        [0.0,       float(sy), 0.0,       0.0],
        [0.0,       0.0,       float(sz), 0.0],
        [0.0,       0.0,       0.0,       1.0],
    ]

    # Translation matrices for pivot and its inverse
    T_p = [
        [1.0, 0.0, 0.0, float(px)],
        [0.0, 1.0, 0.0, float(py)],
        [0.0, 0.0, 1.0, float(pz)],
        [0.0, 0.0, 0.0, 1.0],
    ]
    T_n = [
        [1.0, 0.0, 0.0, -float(px)],
        [0.0, 1.0, 0.0, -float(py)],
        [0.0, 0.0, 1.0, -float(pz)],
        [0.0, 0.0, 0.0, 1.0],
    ]

    # World translation
    T = [
        [1.0, 0.0, 0.0, float(tx)],
        [0.0, 1.0, 0.0, float(ty)],
        [0.0, 0.0, 1.0, float(tz)],
        [0.0, 0.0, 0.0, 1.0],
    ]

    # Compose: T * Tpivot * R * S * T(-pivot)
    M = mat4_mul(T_p, S)
    M = mat4_mul(R, M)
    M = mat4_mul(T, mat4_mul(T_p, mat4_mul(R, mat4_mul(S, T_n))))  # clarity version
    # The previous line re-expands to emphasize order; keep as-is for readability.
    return M


def to_vtk_matrix4x4(M: List[List[float]]):
    """Create a vtkMatrix4x4 from a python 4x4 list."""
    try:
        import vtk
    except Exception:
        return None
    mat = vtk.vtkMatrix4x4()
    for r in range(4):
        for c in range(4):
            mat.SetElement(r, c, float(M[r][c]))
    return mat

# -------------------------
# Actor transform helpers
# -------------------------

def _get_record_actor(rec: dict):
    """Return the primary actor from a dataset record."""
    for key in ("actor_mesh", "actor_points"):
        act = rec.get(key)
        if act is not None:
            return act
    return None


def apply_actor_user_matrix(actor, M: List[List[float]]) -> None:
    """Set an actor UserMatrix to apply TRS without altering the dataset."""
    if actor is None:
        return
    mat = to_vtk_matrix4x4(M)
    if mat is None:
        return
    try:
        actor.SetUserMatrix(mat)
    except Exception:
        # Some wrappers expose .user_matrix
        try:
            actor.user_matrix = mat
        except Exception:
            pass


def clear_actor_user_matrix(actor) -> None:
    """Remove any user matrix (reset transform)."""
    if actor is None:
        return
    try:
        actor.SetUserMatrix(None)
    except Exception:
        try:
            actor.user_matrix = None
        except Exception:
            pass

# -------------------------
# Dialog UI
# -------------------------

@dataclass
class TargetEntry:
    key: str          # dataset index as string or logical key
    label: str        # shown in combo
    rec: dict         # original record for direct access


class TransformDialog(QtWidgets.QDialog):
    """Dialog to transform datasets and load a .conf placement file."""

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.viewer = window.viewer3d
        self.setWindowTitle("Rotation / Translation")
        self.setModal(False)

        self.targets: List[TargetEntry] = self._collect_targets()
        self._build_ui()
        self._fill_targets()

    # ---- data collect ----
    def _collect_targets(self) -> List[TargetEntry]:
        out: List[TargetEntry] = []
        dsets = getattr(self.viewer, "_datasets", [])
        for i, rec in enumerate(dsets):
            # Build a readable label including filename if available
            base = os.path.basename(rec.get("source_path") or f"dataset_{i}")
            kind = rec.get("kind", "?")
            label = f"[{i}] {base}  ({kind})"
            out.append(TargetEntry(str(i), label, rec))
        return out

    def _notify_tree_added(self, parent_index: int | None, rec_new: dict, label: str) -> None:
        """Best-effort: notify the host to add/update the tree for a new dataset.

        We try a few host methods in order. If none is present, we fall back to
        forcing a full tree refresh.
        """
        w = self.window

        # Prefer explicit "add child" signatures if available
        for name in ("treeAddChild", "_tree_add_child", "tree_add_child"):
            fn = getattr(w, name, None)
            if callable(fn):
                try:
                    fn(parent_index, rec_new, label)
                    return
                except Exception:
                    pass

        # Try a generic "add dataset"
        for name in ("treeAddDataset", "_tree_add_dataset", "tree_add_dataset"):
            fn = getattr(w, name, None)
            if callable(fn):
                try:
                    fn(rec_new, parent_index)
                    return
                except Exception:
                    pass

        # Fall back to full rebuild/refresh
        for name in ("_rebuild_tree_from_datasets", "rebuild_tree_from_datasets", "_refresh_tree_visibility"):
            fn = getattr(w, name, None)
            if callable(fn):
                try:
                    fn()
                    return
                except Exception:
                    pass

        # Last resort: silent
        try:
            w._append_message("[Transform] Added duplicated dataset; tree update fallback used.")
        except Exception:
            pass

    # ---- UI ----
    def _build_ui(self):
        """Build dialog UI.

        Rows:
          1) Dataset selector
          2) Buttons: Load .conf  |  Reset Transformation
          3) Translation group
          4) Rotations row: Euler (left)  +  Quaternion (right)
          5) Action buttons on two rows:
             - Row A: Show transformation   |   Apply (overwrite)
             - Row B: Duplicate cloud
        """
        layout = QtWidgets.QVBoxLayout(self)

        # --- Row 1: dataset selector ---
        row_target = QtWidgets.QHBoxLayout()
        row_target.addWidget(QtWidgets.QLabel("Dataset"))
        self.cmbTarget = QtWidgets.QComboBox()
        row_target.addWidget(self.cmbTarget, 1)
        layout.addLayout(row_target)

        # --- Row 2: .conf + reset ---
        row_conf = QtWidgets.QHBoxLayout()
        self.btnLoadConf = QtWidgets.QPushButton("Load .conf…")
        self.btnReset = QtWidgets.QPushButton("Reset transformation")
        self.btnLoadConf.setToolTip("Load a placement .conf file and apply transforms by filename")
        self.btnReset.setToolTip("Clear any temporary (UserMatrix) transform on the selected dataset actor")
        row_conf.addWidget(self.btnLoadConf)
        row_conf.addStretch(1)
        row_conf.addWidget(self.btnReset)
        layout.addLayout(row_conf)

        # --- Row 3: translation ---
        grpT = QtWidgets.QGroupBox("Translation")
        formT = QtWidgets.QFormLayout(grpT)
        self.spinTx = QtWidgets.QDoubleSpinBox(); self._cfg_spin(self.spinTx)
        self.spinTy = QtWidgets.QDoubleSpinBox(); self._cfg_spin(self.spinTy)
        self.spinTz = QtWidgets.QDoubleSpinBox(); self._cfg_spin(self.spinTz)
        formT.addRow("Tx", self.spinTx)
        formT.addRow("Ty", self.spinTy)
        formT.addRow("Tz", self.spinTz)
        layout.addWidget(grpT)

        # --- Row 3.5: scale (anisotropic) ---
        grpS = QtWidgets.QGroupBox("Scale (Sx, Sy, Sz)")
        formS = QtWidgets.QFormLayout(grpS)
        self.spinSx = QtWidgets.QDoubleSpinBox(); self._cfg_spin(self.spinSx, 1e-6, 1e6, 0.001); self.spinSx.setValue(1.0)
        self.spinSy = QtWidgets.QDoubleSpinBox(); self._cfg_spin(self.spinSy, 1e-6, 1e6, 0.001); self.spinSy.setValue(1.0)
        self.spinSz = QtWidgets.QDoubleSpinBox(); self._cfg_spin(self.spinSz, 1e-6, 1e6, 0.001); self.spinSz.setValue(1.0)
        formS.addRow("Sx", self.spinSx)
        formS.addRow("Sy", self.spinSy)
        formS.addRow("Sz", self.spinSz)
        layout.addWidget(grpS)

        # --- Row 3.6: pivot (scale/rotate around this point) ---
        grpP = QtWidgets.QGroupBox("Pivot (Px, Py, Pz)")
        formP = QtWidgets.QFormLayout(grpP)
        self.spinPx = QtWidgets.QDoubleSpinBox(); self._cfg_spin(self.spinPx, -1e6, 1e6, 0.01); self.spinPx.setValue(0.0)
        self.spinPy = QtWidgets.QDoubleSpinBox(); self._cfg_spin(self.spinPy, -1e6, 1e6, 0.01); self.spinPy.setValue(0.0)
        self.spinPz = QtWidgets.QDoubleSpinBox(); self._cfg_spin(self.spinPz, -1e6, 1e6, 0.01); self.spinPz.setValue(0.0)
        formP.addRow("Px", self.spinPx)
        formP.addRow("Py", self.spinPy)
        formP.addRow("Pz", self.spinPz)
        layout.addWidget(grpP)

        # --- Row 4: rotations (Euler + Quaternion) side-by-side ---
        row_rot = QtWidgets.QHBoxLayout()

        grpR = QtWidgets.QGroupBox("Rotation (Euler, deg, XYZ)")
        formR = QtWidgets.QFormLayout(grpR)
        self.spinRx = QtWidgets.QDoubleSpinBox(); self._cfg_spin(self.spinRx, -360, 360, 0.1)
        self.spinRy = QtWidgets.QDoubleSpinBox(); self._cfg_spin(self.spinRy, -360, 360, 0.1)
        self.spinRz = QtWidgets.QDoubleSpinBox(); self._cfg_spin(self.spinRz, -360, 360, 0.1)
        formR.addRow("Rx", self.spinRx)
        formR.addRow("Ry", self.spinRy)
        formR.addRow("Rz", self.spinRz)
        row_rot.addWidget(grpR, 1)

        grpQ = QtWidgets.QGroupBox("Rotation (Quaternion x, y, z, w)")
        formQ = QtWidgets.QFormLayout(grpQ)
        self.spinQx = QtWidgets.QDoubleSpinBox(); self._cfg_spin(self.spinQx, -1e6, 1e6, 1e-6)
        self.spinQy = QtWidgets.QDoubleSpinBox(); self._cfg_spin(self.spinQy, -1e6, 1e6, 1e-6)
        self.spinQz = QtWidgets.QDoubleSpinBox(); self._cfg_spin(self.spinQz, -1e6, 1e6, 1e-6)
        self.spinQw = QtWidgets.QDoubleSpinBox(); self._cfg_spin(self.spinQw, -1e6, 1e6, 1e-6); self.spinQw.setValue(1.0)
        formQ.addRow("Qx", self.spinQx)
        formQ.addRow("Qy", self.spinQy)
        formQ.addRow("Qz", self.spinQz)
        formQ.addRow("Qw", self.spinQw)
        row_rot.addWidget(grpQ, 1)

        layout.addLayout(row_rot)

        # --- Row 5: action buttons on two rows ---
        # Row A: Show | Apply
        rowA = QtWidgets.QHBoxLayout()
        self.btnShow = QtWidgets.QPushButton("Show transformation")
        self.btnApply = QtWidgets.QPushButton("Apply (overwrite)")
        self.btnShow.setToolTip("Temporarily show the transform using the actor UserMatrix (non-destructive).")
        self.btnApply.setToolTip("Bake the transform into the dataset geometry and clear the temporary matrix.")
        rowA.addWidget(self.btnShow)
        rowA.addStretch(1)
        rowA.addWidget(self.btnApply)

        # Row B: Duplicate
        rowB = QtWidgets.QHBoxLayout()
        self.btnDuplicate = QtWidgets.QPushButton("Duplicate cloud")
        self.btnDuplicate.setToolTip("Create a new dataset with the transformed geometry (original untouched).")
        rowB.addStretch(1)
        rowB.addWidget(self.btnDuplicate)

        layout.addLayout(rowA)
        layout.addLayout(rowB)

        # --- Connections ---
        # Live sync: when Euler changes, update quaternion
        for sp in (self.spinRx, self.spinRy, self.spinRz):
            sp.valueChanged.connect(self._sync_quat_from_euler)

        self.btnReset.clicked.connect(self._on_reset_transform)
        self.btnLoadConf.clicked.connect(self._on_load_conf)
        self.btnShow.clicked.connect(self._on_show_only)
        self.btnApply.clicked.connect(self._on_apply_baked)
        self.btnDuplicate.clicked.connect(self._on_duplicate)

        # Width for side panel friendliness
        self.setMinimumWidth(380)
        self.resize(420, self.sizeHint().height())

    def _cfg_spin(self, sp: QtWidgets.QDoubleSpinBox, mn=-1e6, mx=1e6, step=0.01):
        sp.setRange(mn, mx)
        sp.setDecimals(6)
        sp.setSingleStep(step)

    def _fill_targets(self):
        self.cmbTarget.clear()
        for t in self.targets:
            self.cmbTarget.addItem(t.label, t.key)

    # ---- actions ----
    def _sync_quat_from_euler(self):
        """Recompute quaternion from current Euler and update the 4 spin boxes.
        We block signals to avoid feedback loops."""
        qx, qy, qz, qw = euler_deg_to_quat(
            self.spinRx.value(), self.spinRy.value(), self.spinRz.value(), "XYZ"
        )
        for sp, val in ((self.spinQx, qx), (self.spinQy, qy), (self.spinQz, qz), (self.spinQw, qw)):
            sp.blockSignals(True)
            sp.setValue(val)
            sp.blockSignals(False)

    def _current_params(self):
        """Return current (T, Q, S, P) tuples from the UI."""
        T = (self.spinTx.value(), self.spinTy.value(), self.spinTz.value())
        Q = (self.spinQx.value(), self.spinQy.value(), self.spinQz.value(), self.spinQw.value())
        S = (self.spinSx.value(), self.spinSy.value(), self.spinSz.value())
        P = (self.spinPx.value(), self.spinPy.value(), self.spinPz.value())
        return T, Q, S, P

    def _actor_for_current(self):
        idx_str = self.cmbTarget.currentData()
        if idx_str is None:
            return None, None
        try:
            ds = int(idx_str)
        except Exception:
            return None, None
        rec = self._get_record(ds)
        if rec is None:
            return None, None
        act = _get_record_actor(rec)
        return rec, act

    def _on_show_only(self):
        """Apply the transform as a temporary UserMatrix on the actor only (non-destructive)."""
        rec, act = self._actor_for_current()
        if act is None:
            return
        T, Q, S, P = self._current_params()
        M = compose_trs_scale_pivot(*T, *Q, *S, *P)
        with _ProgressHelper(self.window, "Previewing transform…", total=2) as pg:
            apply_actor_user_matrix(act, M); pg.step()
            self._refresh(); pg.step()

    def _on_apply_baked(self):
        """Bake the transform into the dataset geometry and clear the temporary matrix."""
        rec, act = self._actor_for_current()
        if rec is None:
            return
        T, Q, S, P = self._current_params()
        M = compose_trs_scale_pivot(*T, *Q, *S, *P)

        with _ProgressHelper(self.window, "Applying transform…", total=3) as pg:
            ok = self._bake_transform_into_record(rec, M); pg.step()
            clear_actor_user_matrix(act); pg.step()
            self._refresh(); pg.step()

        try:
            self.window._append_message("[Transform] Apply (overwrite): " + ("OK" if ok else "FAILED"))
        except Exception:
            pass

    def _on_duplicate(self):
        """Create a new dataset with transformed geometry, keep original untouched."""
        rec, _act = self._actor_for_current()
        if rec is None:
            return
        T, Q, S, P = self._current_params()
        M = compose_trs_scale_pivot(*T, *Q, *S, *P)

        with _ProgressHelper(self.window, "Duplicating dataset…", total=2) as pg:
            ok = self._duplicate_record_transformed(rec, M); pg.step()
            self._refresh(); pg.step()

        if not ok:
            QtWidgets.QMessageBox.warning(self, "Duplicate cloud", "Could not duplicate the dataset with current viewer API.")
        else:
            try:
                self.window._append_message("[Transform] Duplicate cloud: OK")
            except Exception:
                pass
    # ---- geometry ops ----

    def _apply_affine_to_points(self, pts_np, M4):
        """Return transformed Nx3 points applying 4x4 matrix."""
        try:
            import numpy as _np
            M = _np.asarray(M4, dtype=float)
            P = _np.asarray(pts_np, dtype=float)
            if P.ndim != 2 or P.shape[1] != 3:
                return None
            Ph = _np.c_[P, _np.ones((P.shape[0], 1))]
            P2 = (Ph @ M.T)[:, :3]
            return P2
        except Exception:
            return None

    def _bake_transform_into_record(self, rec: dict, M4) -> bool:
        """Modify the underlying dataset geometry in-place.
        Works for both points-polydata and mesh (unstructured/surface)."""
        try:
            import pyvista as pv
        except Exception:
            return False
        # Points dataset
        if rec.get("kind") == "points":
            pd = rec.get("full_pdata") or rec.get("pdata")
            if pd is None:
                return False
            P2 = self._apply_affine_to_points(pd.points, M4)
            if P2 is None:
                return False
            pd.points = P2
            if rec.get("pdata") is not pd:
                try:
                    rec["pdata"].points = P2
                except Exception:
                    pass
            act = rec.get("actor_points")
            if act is not None:
                try:
                    mapper = getattr(act, "GetMapper", lambda: None)()
                    if mapper is not None and hasattr(mapper, "SetInputData"):
                        mapper.SetInputData(pd)
                except Exception:
                    pass
            return True

        # Mesh dataset
        if rec.get("kind") == "mesh":
            mesh = rec.get("mesh")
            if mesh is None:
                return False
            P2 = self._apply_affine_to_points(mesh.points, M4)
            if P2 is None:
                return False
            mesh.points = P2
            act = rec.get("actor_mesh")
            if act is not None:
                try:
                    mapper = getattr(act, "GetMapper", lambda: None)()
                    if mapper is not None and hasattr(mapper, "SetInputData"):
                        mapper.SetInputData(mesh)
                except Exception:
                    pass
            return True

        return False

    def _append_new_points_dataset(self, poly, label: str, parent_ds: int | None) -> bool:
        """Best-effort: append a new points dataset to the viewer and tree.

        We also try to register a parent/child relation by storing 'parent'
        on the new record when possible, so that the MainWindow can render a
        branch in the tree (if it supports it).
        """
        v = self.viewer

        # First, try explicit viewer APIs if present (optional)
        for name in ("add_points_dataset", "add_points_polydata", "add_polydata"):
            fn = getattr(v, name, None)
            if callable(fn):
                try:
                    fn(poly, label=label)
                except TypeError:
                    # Fallback to single-arg signatures
                    try:
                        fn(poly)
                    except Exception:
                        pass
                break

        # Always keep a local record for consistency with the app's registry
        try:
            actor = v.plotter.add_mesh(
                poly,
                render_points_as_spheres=getattr(v, "_points_as_spheres", False),
                point_size=3,
                name=label,
            )
        except Exception:
            actor = None

        rec_new = {
            "kind": "points",
            "full_pdata": poly,
            "pdata": poly,
            "actor_points": actor,
            "source_path": None,         # new derived geometry, no original file
            "name": label,               # human label used by the tree when available
            "visible": True,
            "point_size": 3,
            "view_percent": 100,
            "color_mode": "Normal Colormap",
            "cmap": "viridis",
            "points_as_spheres": getattr(v, "_points_as_spheres", False),
            "parent": int(parent_ds) if parent_ds is not None else None,
        }
        getattr(v, "_datasets", []).append(rec_new)

        # Tell the UI tree to add a child under parent_ds (best effort)
        try:
            self._notify_tree_added(parent_ds, rec_new, label)
        except Exception:
            pass

        # Refresh view
        try:
            self.window.viewer3d.refresh()
        except Exception:
            pass
        return True

    def _append_new_mesh_dataset(self, mesh, label: str, parent_ds: int | None) -> bool:
        """Best-effort: append a new mesh dataset to the viewer and tree, tracking parent."""
        v = self.viewer

        for name in ("add_mesh_dataset", "add_mesh_polydata", "add_unstructured"):
            fn = getattr(v, name, None)
            if callable(fn):
                try:
                    fn(mesh, label=label)
                except TypeError:
                    try:
                        fn(mesh)
                    except Exception:
                        pass
                break

        try:
            actor = v.plotter.add_mesh(mesh, name=label, show_edges=False, opacity=1.0)
        except Exception:
            actor = None

        rec_new = {
            "kind": "mesh",
            "mesh": mesh,
            "actor_mesh": actor,
            "source_path": None,
            "name": label,
            "visible": True,
            "representation": "surface",
            "opacity": 100,
            "solid_color": (1.0, 1.0, 1.0),
            "parent": int(parent_ds) if parent_ds is not None else None,
        }
        getattr(v, "_datasets", []).append(rec_new)

        try:
            self._notify_tree_added(parent_ds, rec_new, label)
        except Exception:
            pass

        try:
            self.window.viewer3d.refresh()
        except Exception:
            pass
        return True

    def _duplicate_record_transformed(self, rec: dict, M4) -> bool:
        """Create a new dataset as a transformed copy."""
        try:
            import numpy as _np
        except Exception:
            return False

        # Determine parent dataset index for tree relationship (if possible)
        parent_ds: int | None = None
        try:
            dsets = getattr(self.viewer, "_datasets", [])
            parent_ds = dsets.index(rec)
        except Exception:
            parent_ds = None

        base_name = os.path.basename(rec.get("source_path") or rec.get("name") or "dataset")
        # Build a concise suffix as requested: _RT (Rotation/Translation)
        label = f"{base_name}_RT"

        if rec.get("kind") == "points":
            src = rec.get("full_pdata") or rec.get("pdata")
            if src is None:
                return False
            dst = src.copy(deep=True)
            pts2 = self._apply_affine_to_points(dst.points, M4)
            if pts2 is None:
                return False
            dst.points = pts2
            return self._append_new_points_dataset(dst, label, parent_ds)

        if rec.get("kind") == "mesh":
            src = rec.get("mesh")
            if src is None:
                return False
            dst = src.copy(deep=True)
            pts2 = self._apply_affine_to_points(dst.points, M4)
            if pts2 is None:
                return False
            dst.points = pts2
            return self._append_new_mesh_dataset(dst, label, parent_ds)

        return False

    def _on_reset_transform(self):
        idx_str = self.cmbTarget.currentData()
        if idx_str is None:
            return
        try:
            ds = int(idx_str)
        except Exception:
            return
        rec = self._get_record(ds)
        if rec is None:
            return
        act = _get_record_actor(rec)
        clear_actor_user_matrix(act)
        self._refresh()

    def _on_load_conf(self):
        # .conf lines:
        # camera cx cy cz  qx qy qz qw
        # bmesh filename.ply tx ty tz  qx qy qz qw
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open placement .conf", "", "Conf (*.conf *.txt);;All (*)")
        if not path:
            return
        try:
            with open(path, "r") as f:
                lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Load .conf", f"Failed to read file:\n{exc}")
            return

        applied = 0
        with _ProgressHelper(self.window, f"Applying {os.path.basename(path)}…", total=max(1, len(lines))) as pg:
            for ln in lines:
                tokens = ln.split()
                if not tokens:
                    pg.step(); continue
                kind = tokens[0].lower()
                if kind == "camera":
                    if len(tokens) >= 8:
                        cx, cy, cz = map(float, tokens[1:4])
                        qx, qy, qz, qw = map(float, tokens[4:8])
                        self._apply_camera((cx, cy, cz), (qx, qy, qz, qw))
                elif kind in ("bmesh", "mesh", "points"):
                    if len(tokens) >= 9:
                        fname = tokens[1]
                        tx, ty, tz = map(float, tokens[2:5])
                        qx, qy, qz, qw = map(float, tokens[5:9])
                        if self._apply_by_filename(fname, (tx, ty, tz), (qx, qy, qz, qw)):
                            applied += 1
                pg.step()

        self._refresh()
        QtWidgets.QMessageBox.information(self, "Load .conf", f"Applied transforms to {applied} dataset(s).")

    # ---- helpers ----

    def _get_record(self, ds_index: int) -> Optional[dict]:
        dsets = getattr(self.viewer, "_datasets", [])
        if 0 <= ds_index < len(dsets):
            return dsets[ds_index]
        return None

    def _apply_to_dataset_index(self, ds_index: int, T: Tuple[float,float,float], Q: Tuple[float,float,float,float]) -> None:
        rec = self._get_record(ds_index)
        if rec is None:
            return
        act = _get_record_actor(rec)
        if act is None:
            return
        M = compose_trs(T[0], T[1], T[2], Q[0], Q[1], Q[2], Q[3])
        apply_actor_user_matrix(act, M)
        self._refresh()

    def _apply_by_filename(self, filename: str, T: Tuple[float,float,float], Q: Tuple[float,float,float,float]) -> bool:
        """Match by basename against rec['source_path'] or fallback to label."""
        base = os.path.basename(filename)
        dsets = getattr(self.viewer, "_datasets", [])
        hit = False
        for i, rec in enumerate(dsets):
            sp = rec.get("source_path")
            ok = False
            if sp:
                ok = (os.path.basename(sp) == base)
            else:
                # fallback: try label key
                ok = (base in (os.path.basename(sp or f"dataset_{i}")))
            if not ok:
                continue
            act = _get_record_actor(rec)
            if act is None:
                continue
            M = compose_trs(T[0], T[1], T[2], Q[0], Q[1], Q[2], Q[3])
            apply_actor_user_matrix(act, M)
            hit = True
        return hit

    def _apply_camera(self, pos: Tuple[float,float,float], q: Tuple[float,float,float,float]) -> None:
        """Apply camera position+orientation if available from .conf."""
        try:
            # Convert quaternion to view-up and view-direction
            # We rotate basis vectors by R(q): forward = R*[0,0,-1], up = R*[0,1,0]
            M = quat_to_mat4(*q)
            fwd = ( -M[0][2], -M[1][2], -M[2][2] )
            up  = (  M[0][1],  M[1][1],  M[2][1] )
            cam = self.viewer.plotter.camera
            cam.position = pos
            cam.focal_point = (pos[0] + fwd[0], pos[1] + fwd[1], pos[2] + fwd[2])
            cam.up = up
        except Exception:
            pass

    def _refresh(self):
        try:
            self.viewer.plotter.update()
            self.viewer.refresh()
        except Exception:
            pass


# -------------------------
# Plugin wrapper
# -------------------------

class TransformRTPlugin(QtCore.QObject):
    """Wire menu entry and dialog into the host MainWindow."""
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.dialog: Optional[TransformDialog] = None

        self.action = QtGui.QAction(QtGui.QIcon(), "Rotation/Translation…", self)
        self.action.setShortcut(QtGui.QKeySequence("Ctrl+T"))
        self.action.triggered.connect(lambda _checked=False: self.open_dialog())

        # Add to Tools menu (create if missing)
        try:
            mb = window.menuBar()
            m_tools = None
            for act in mb.actions():
                if act.text().replace("&", "") == "Tools":
                    m_tools = act.menu()
                    break
            if m_tools is None:
                m_tools = mb.addMenu("&Tools")
            m_tools.addAction(self.action)
        except Exception:
            pass

    @QtCore.Slot()
    def open_dialog(self):
        if self.dialog is None:
            self.dialog = TransformDialog(self.window)
        # Refresh target list each time (datasets may have changed)
        self.dialog.targets = self.dialog._collect_targets()
        self.dialog._fill_targets()
        try:
            self.dialog._sync_quat_from_euler()
        except Exception:
            pass
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()

    # Allow PluginManager.run("transform_rt")
    def run(self, *_, **__):
        self.open_dialog()

# Entry point for PluginManager
def register(window):
    return TransformRTPlugin(window)