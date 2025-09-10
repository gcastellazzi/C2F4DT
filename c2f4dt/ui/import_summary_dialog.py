# c2f4dt/ui/import_summary_dialog.py
from __future__ import annotations

from typing import Iterable, List, Dict, Any

from PySide6 import QtCore, QtWidgets

from ..utils.io.importers import ImportedObject


class ImportSummaryDialog(QtWidgets.QDialog):
    """Modal dialog to review imported objects and choose import options.

    Inspired by CloudCompare. For each object we show discovered attributes
    (read-only) and let the user:
      - Remap axes with sign (e.g., X=+Y, Y=-Z, Z=+X)
      - Apply same mapping to normals
      - Compute normals if missing
      - Choose coloring preference (RGB or colormap over fake intensity)

    This dialog DOES NOT mutate input objects. Use `operations()` to
    retrieve what the user selected and apply it in the caller.
    """

    def __init__(self, objects: Iterable[ImportedObject], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import summary")
        self.setModal(True)
        self._objects: List[ImportedObject] = list(objects)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # One collapsible-like page per object (CloudCompare-like feel)
        self.toolbox = QtWidgets.QToolBox(self)
        layout.addWidget(self.toolbox, 1)

        self._pages: List[Dict[str, Any]] = []
        for obj in self._objects:
            page = self._build_page(obj)
            self.toolbox.addItem(page["widget"], obj.name)
            self._pages.append(page)

        # Bounds/meta hint
        self.lblMeta = QtWidgets.QLabel("")
        self.lblMeta.setWordWrap(True)
        layout.addWidget(self.lblMeta)
        self._populate_bounds()

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    # ------------------------------------------------------------------
    # UI builders
    # ------------------------------------------------------------------
    def _build_page(self, obj: ImportedObject) -> Dict[str, Any]:
        """Create one page with labels and options for an object.

        Args:
            obj: ImportedObject to present.

        Returns:
            A dict with references to created widgets to read them later.
        """
        w = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(w)
        form.setLabelAlignment(QtCore.Qt.AlignRight)
        form.setFormAlignment(QtCore.Qt.AlignTop)

        # ---- Read-only attributes
        npts = obj.points.shape[0] if obj.points is not None else (
            obj.pv_mesh.n_points if obj.pv_mesh is not None else 0
        )
        nfaces = obj.faces.shape[0] if obj.faces is not None else (
            obj.pv_mesh.n_faces if obj.pv_mesh is not None else 0
        )
        has_rgb = obj.colors is not None
        has_int = obj.intensity is not None
        has_nrm = getattr(obj, "normals", None) is not None

        form.addRow("Kind", QtWidgets.QLabel(obj.kind))
        form.addRow("#Points", QtWidgets.QLabel(str(npts)))
        form.addRow("#Faces", QtWidgets.QLabel(str(nfaces)))
        form.addRow("Has RGB", QtWidgets.QLabel("Yes" if has_rgb else "No"))
        form.addRow("Has Intensity", QtWidgets.QLabel("Yes" if has_int else "No"))
        form.addRow("Has Normals", QtWidgets.QLabel("Yes" if has_nrm else "No"))

        # ---- Axis mapping (with presets)
        axis_opts = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
        cmb_preset = QtWidgets.QComboBox()
        cmb_preset.addItems([
            "Custom",
            "Z-up (identity)",      # X->+X, Y->+Y, Z->+Z
            "Y-up (swap Y/Z)",      # X->+X, Y->+Z, Z->-Y
            "X-up (swap X/Z)",      # X->+Z, Y->+Y, Z->-X
            "Flip Z",               # X->+X, Y->+Y, Z->-Z
            "Flip Y",               # X->+X, Y->-Y, Z->+Z
            "Flip X",               # X->-X, Y->+Y, Z->+Z
        ])

        cmb_x = QtWidgets.QComboBox(); cmb_x.addItems(axis_opts); cmb_x.setCurrentText("+X")
        cmb_y = QtWidgets.QComboBox(); cmb_y.addItems(axis_opts); cmb_y.setCurrentText("+Y")
        cmb_z = QtWidgets.QComboBox(); cmb_z.addItems(axis_opts); cmb_z.setCurrentText("+Z")

        def apply_preset(name: str) -> None:
            presets = {
                "Z-up (identity)": ("+X", "+Y", "+Z"),
                "Y-up (swap Y/Z)": ("+X", "+Z", "-Y"),
                "X-up (swap X/Z)": ("+Z", "+Y", "-X"),
                "Flip Z": ("+X", "+Y", "-Z"),
                "Flip Y": ("+X", "-Y", "+Z"),
                "Flip X": ("-X", "+Y", "+Z"),
            }
            if name in presets:
                x, y, z = presets[name]
                cmb_x.setCurrentText(x)
                cmb_y.setCurrentText(y)
                cmb_z.setCurrentText(z)

        cmb_preset.currentTextChanged.connect(apply_preset)

        form.addRow("Axis preset", cmb_preset)
        form.addRow("Map to X", cmb_x)
        form.addRow("Map to Y", cmb_y)
        form.addRow("Map to Z", cmb_z)

        chk_apply_normals = QtWidgets.QCheckBox("Apply same mapping to normals (if present)")
        chk_apply_normals.setChecked(True)
        form.addRow("Normals mapping", chk_apply_normals)

        chk_compute_normals = QtWidgets.QCheckBox("Compute normals if missing")
        chk_compute_normals.setChecked(not has_nrm)
        form.addRow("Normals compute", chk_compute_normals)

        # ---- Coloring preference
        grp_color = QtWidgets.QGroupBox("Coloring preference")
        v = QtWidgets.QVBoxLayout(grp_color)
        rad_rgb = QtWidgets.QRadioButton("Use RGB (if available)")
        rad_cmap = QtWidgets.QRadioButton("Use colormap (fake intensity)")
        if has_rgb:
            rad_rgb.setChecked(True)
        else:
            rad_cmap.setChecked(True)
        v.addWidget(rad_rgb)
        v.addWidget(rad_cmap)
        form.addRow(grp_color)

        return {
            "widget": w,
            "cmb_preset": cmb_preset,
            "cmb_x": cmb_x,
            "cmb_y": cmb_y,
            "cmb_z": cmb_z,
            "chk_apply_normals": chk_apply_normals,
            "chk_compute_normals": chk_compute_normals,
            "rad_rgb": rad_rgb,
            "rad_cmap": rad_cmap,
            "obj": obj,
        }

    def _populate_bounds(self) -> None:
        meta_desc = []
        for obj in self._objects:
            b = obj.bounds() if hasattr(obj, "bounds") else None
            if b:
                meta_desc.append(
                    f"{obj.name} bounds: x=[{b[0]:.3f},{b[1]:.3f}] "
                    f"y=[{b[2]:.3f},{b[3]:.3f}] z=[{b[4]:.3f},{b[5]:.3f}]"
                )
        self.lblMeta.setText("\n".join(meta_desc))

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    @property
    def objects(self) -> List[ImportedObject]:
        """Return original objects (unchanged)."""
        return self._objects

    def operations(self) -> List[Dict[str, Any]]:
        """Return per-object operations selected in the dialog.

        Each entry:
          - axis_map: {'X': str, 'Y': str, 'Z': str} in {+X,-X,+Y,-Y,+Z,-Z}
          - map_normals: bool
          - compute_normals_if_missing: bool
          - color_preference: 'rgb' | 'colormap'
        """
        ops: List[Dict[str, Any]] = []
        for page in self._pages:
            axis_map = {
                "X": page["cmb_x"].currentText(),
                "Y": page["cmb_y"].currentText(),
                "Z": page["cmb_z"].currentText(),
            }
            color_pref = "rgb" if page["rad_rgb"].isChecked() else "colormap"
            ops.append(
                {
                    "axis_map": axis_map,
                    "map_normals": page["chk_apply_normals"].isChecked(),
                    "compute_normals_if_missing": page["chk_compute_normals"].isChecked(),
                    "color_preference": color_pref,
                }
            )
        return ops