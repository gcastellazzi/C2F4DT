# -*- coding: utf-8 -*-
"""
VTK Import & Display plugin for C2F4DT.

WHAT THE PLUGIN DOES
--------------------
- Adds a menu entry: File ▸ Import VTK… (with shortcut)
- Imports a *single* VTK/VTU/VTP/VTM/VTS/VTR/VTI/OBJ/STL file
- If MultiBlock: keeps *a single actor* (as preferred)
- If the reader exposes `time_values`: adds a simple time-slider in the box
- Creates a “VTK Display” box in the scrollDISPLAY with ParaView-style controls:
    * Representation: Points / Wireframe / Surface / Surface with Edges / Volume (if applicable)
    * Color by: Solid Color / arrays (PointData/CellData), for vectors Mag/ X / Y / Z
    * LUT + invert, Rescale to Data, Scalar Bar On/Off
    * Opacity, Point Size, Line Width
    * Edge visibility + Edge color
    * Basic lighting (toggle)
- Applies changes **live** to the “current” dataset (selected in treeMCTS)
- Saves the `source_path` + essential choices in mcts (extendable towards a “style file”)

HOW TO ADAPT IT
---------------
- See the `_apply_*` methods to map the UI controls to your Viewer3D APIs
- If an API is unavailable, the `TODO` points indicate where to add the fallback
"""

from __future__ import annotations
from typing import Optional, Tuple, List, Dict
import os

from PySide6 import QtCore, QtGui, QtWidgets

# PyVista is a dependency declared in plugin.yaml
import pyvista as pv
import numpy as np


# Optionally import KDTree for mapping
try:
    from scipy.spatial import cKDTree as KDTree
except Exception:
    KDTree = None

# ---------------------------------------------------------------------
# Colormap normalization helper
# ---------------------------------------------------------------------
def _normalize_cmap(name: str, invert: bool) -> str:
    """Return a matplotlib-compatible colormap name (lowercase) and reverse if requested."""
    if not isinstance(name, str):
        return "viridis_r" if invert else "viridis"
    cm = name.strip().lower() or "viridis"
    if invert and not cm.endswith("_r"):
        cm = cm + "_r"
    return cm

# ---------------------------------------------------------------------
# Helpers UI
# ---------------------------------------------------------------------
def _add_to_display_panel(window, title: str, widget: QtWidgets.QWidget) -> None:
    """
    Adds a *group* to the right panel (scrollDISPLAY).
    Prefers an existing `add_plugin_section` API if available, otherwise
    directly inserts into the main layout of the DisplayPanel.
    """
    try:
        if hasattr(window.displayPanel, "add_plugin_section"):
            window.displayPanel.add_plugin_section(title, widget)
            return
    except Exception:
        pass

    # Fallback: wrap the widget in a QGroupBox and append it to the main vertical layout.
    box = QtWidgets.QGroupBox(title)
    box.setMaximumWidth(320)
    lay = QtWidgets.QVBoxLayout(box)
    lay.setContentsMargins(4, 4, 4, 4)
    
    widget.setMaximumWidth(320)
    widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
    lay.addWidget(widget)
    box.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
    try:
        # displayPanel is already embedded in a QScrollArea; we use its main layout
        lp = window.displayPanel.layout()
        if lp is None:
            lp = QtWidgets.QVBoxLayout(window.displayPanel)
        lp.addWidget(box)
        lp.addStretch(0)
    except Exception:
        pass


def _solid_color_button() -> QtWidgets.QPushButton:
    btn = QtWidgets.QPushButton("Solid Color…")
    btn.setObjectName("btnVTKSolidColor")
    return btn

# ---------------------------------------------------------------------
# Surface/array helpers for fallback rendering
# ---------------------------------------------------------------------
def _prefer_surface(rec: dict):
    """Return the renderable surface if present, else the best available dataset."""
    return rec.get("mesh_surface") or rec.get("pdata") or rec.get("full_pdata") or rec.get("mesh")

def _map_point_scalars_to_surface(window, rec: dict, name: str, vector_mode: str = "Magnitude"):
    """Map a point array from the original dataset to the surface, using vtkOriginalPointIds or KDTree as fallback.

    Args:
        window: Main window (unused, kept for symmetry/future logging).
        rec: Viewer dataset record.
        name: PointData array name to map.
        vector_mode: One of {'Magnitude', 'X', 'Y', 'Z'} when the array is vector-like.

    Returns:
        A numpy array with one scalar per surface point, or None if mapping fails.
    """
    try:
        import numpy as _np
        mesh = rec.get("mesh_orig") or rec.get("mesh") or rec.get("pdata") or rec.get("full_pdata")
        surf = rec.get("mesh_surface") or rec.get("pdata") or rec.get("full_pdata") or rec.get("mesh")
        if mesh is None or surf is None:
            return None

        # Find source array
        if hasattr(mesh, "point_data") and name in mesh.point_data:
            base = _np.asarray(mesh.point_data[name])
        elif hasattr(surf, "point_data") and name in surf.point_data:
            base = _np.asarray(surf.point_data[name])
        else:
            return None

        # Vector handling
        if base.ndim == 2 and base.shape[1] in (2, 3):
            vm = (vector_mode or "Magnitude").title()
            if vm == "Magnitude":
                base = _np.linalg.norm(base, axis=1)
            else:
                comp = {"X": 0, "Y": 1, "Z": 2}.get(vm, 0)
                comp = min(comp, base.shape[1] - 1)
                base = base[:, comp]

        # Try OriginalPointIds mapping
        ids = None
        if hasattr(surf, "point_data"):
            for key in ("vtkOriginalPointIds", "vtkOriginalPointID", "origids", "OriginalPointIds"):
                if key in surf.point_data:
                    ids = _np.asarray(surf.point_data[key]).astype(_np.int64)
                    break
        if ids is not None:
            ids = _np.clip(ids, 0, base.shape[0] - 1)
            return base[ids]

        # KDTree fallback
        if KDTree is not None and hasattr(mesh, "points") and hasattr(surf, "points"):
            P_src = _np.asarray(mesh.points)
            P_dst = _np.asarray(surf.points)
            if P_src.size and P_dst.size:
                tree = KDTree(P_src)
                idx = tree.query(P_dst, k=1, workers=-1)[1]
                idx = _np.clip(_np.asarray(idx, dtype=_np.int64), 0, base.shape[0] - 1)
                return base[idx]

        # Last resort: if sizes match, pass-through
        return base if getattr(surf, "n_points", -1) == base.shape[0] else None
    except Exception:
        return None

# ---------------------------------------------------------------------
# Internal fallbacks (direct PyVista control)
# ---------------------------------------------------------------------
def _get_dataset_record(window, ds: int) -> dict | None:
    """
    Return the viewer's dataset record if available.
    Tries to access `window.viewer3d._datasets[ds]` which is expected to be a dict
    containing at least a `mesh`/`pdata` (pyvista object) and an `actor` (pyvista Actor).
    """
    try:
        recs = getattr(window.viewer3d, "_datasets", [])
        if isinstance(recs, list) and 0 <= ds < len(recs):
            return recs[ds]
        if isinstance(recs, dict):
            return recs.get(ds)
    except Exception:
        pass
    return None


def _fallback_render(window, e: dict, ds: int, *,
                     representation: str | None = None,
                     color_mode: str | None = None,
                     solid_color: tuple[int, int, int] | None = None,
                     lut: str | None = None,
                     invert: bool | None = None,
                     scalar_bar: bool | None = None,
                     edge_visibility: bool | None = None,
                     edge_color: tuple[int, int, int] | None = None,
                     opacity: int | None = None,
                     point_size: int | None = None,
                     line_width: int | None = None,
                     lighting: bool | None = None,
                     manual_range: tuple[float, float] | None = None,
                     vector_mode: str | None = None) -> None:
    """
    Re-render the dataset using PyVista directly when Viewer3D API is missing.

    Strategy:
      - Remove the existing actor (if any)
      - Re-add mesh (or volume) with updated styling
      - Update the viewer record's actor reference
    """
    rec = _get_dataset_record(window, ds)
    if rec is None:
        return

    # Pull current state (with overrides)
    rep = representation if representation is not None else e.get("representation", "Surface")
    colmode = color_mode if color_mode is not None else e.get("color_mode", "Solid Color")
    lut = lut if lut is not None else e.get("colormap", "Viridis")
    invert = bool(invert if invert is not None else e.get("invert_lut", False))
    scalar_bar = bool(scalar_bar if scalar_bar is not None else e.get("scalar_bar", False))
    edge_visibility = bool(edge_visibility if edge_visibility is not None else e.get("edge_visibility", False))
    edge_color = edge_color if edge_color is not None else tuple(e.get("edge_color", (0, 0, 0)))
    opacity = int(opacity if opacity is not None else e.get("opacity", 100))
    point_size = int(point_size if point_size is not None else e.get("point_size", 3))
    line_width = int(line_width if line_width is not None else e.get("line_width", 1))
    lighting = bool(lighting if lighting is not None else e.get("lighting", True))
    solid_color = solid_color if solid_color is not None else tuple(e.get("solid_color", (255, 255, 255)))
    vector_mode = vector_mode if vector_mode is not None else e.get("vector_mode", "Magnitude")

    plotter = getattr(window.viewer3d, "plotter", None)
    if plotter is None:
        return

    pdata = _prefer_surface(rec)
    if pdata is None:
        return

    # Remove old actor
    try:
        if rec.get("actor") is not None:
            plotter.remove_actor(rec["actor"])
    except Exception:
        pass

    # Representation
    style = None
    show_edges = False
    if rep == "Points":
        style = "points"
    elif rep == "Wireframe":
        style = "wireframe"
    elif rep == "Surface with Edges":
        style = None
        show_edges = True
    elif rep == "Surface":
        style = None

    # Scalars (Solid vs Array), with mapping for PointData → surface
    scalars = None
    clim = None
    if colmode == "Solid Color":
        scalars = None
    else:
        assoc = "POINT"
        array_name = colmode
        if colmode.startswith("PointData/"):
            assoc = "POINT"
            array_name = colmode.split("/", 1)[1]
        elif colmode.startswith("CellData/"):
            assoc = "CELL"
            array_name = colmode.split("/", 1)[1]

        if assoc == "POINT":
            # Map source point array to surface points (Magnitude / component)
            scalars = _map_point_scalars_to_surface(window, rec, array_name, vector_mode)
        else:
            # TODO: add CellData mapping via vtkOriginalCellIds if needed
            try:
                arr = pdata.cell_data.get(array_name)
            except Exception:
                arr = None
            if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] in (2, 3):
                comp_map = {"X": 0, "Y": 1, "Z": 2}
                comp = comp_map.get(vector_mode or "Magnitude", None)
                if comp is None:
                    scalars = np.linalg.norm(arr, axis=1)
                else:
                    scalars = arr[:, comp if comp < arr.shape[1] else 0]
            else:
                scalars = array_name if arr is not None else None

        # Ensure numpy arrays are float64 and contiguous for VTK mapper
        if isinstance(scalars, np.ndarray):
            scalars = np.ascontiguousarray(scalars, dtype=np.float64)

        if manual_range is not None:
            clim = manual_range
        elif isinstance(scalars, np.ndarray):
            # Auto CLIM from data if we computed an array
            _min = float(np.nanmin(scalars)) if scalars.size else 0.0
            _max = float(np.nanmax(scalars)) if scalars.size else 1.0
            if _max == _min:
                _max = _min + 1e-9
            clim = (_min, _max)

    # Normalize LUT name and inversion
    cmap = _normalize_cmap(lut, bool(invert))

    # Opacity: 0–100 ➜ 0–1
    op_f = max(0.0, min(1.0, opacity / 100.0))

    # Volume path (only for image-like grids)
    actor = None
    try:
        if rep == "Volume" and hasattr(pdata, "dimensions"):
            actor = plotter.add_volume(
                pdata,
                scalars=scalars,
                cmap=cmap,
                clim=clim,
                opacity=op_f,
                name=e.get("name", "dataset"),
            )
        else:
            actor = plotter.add_mesh(
                pdata,
                scalars=scalars,
                cmap=cmap,
                clim=clim,
                style=style,
                show_edges=show_edges or edge_visibility,
                edge_color=edge_color,
                lighting=lighting,
                opacity=op_f,
                line_width=line_width,
                point_size=point_size,
                name=e.get("name", "dataset"),
                scalar_bar_args={"title": ""} if scalar_bar else None,
                copy_mesh=False,
                reset_camera=False,
            )
            if colmode == "Solid Color" and actor is not None:
                try:
                    actor.prop.color = tuple([c / 255.0 for c in solid_color])
                except Exception:
                    pass
    except Exception:
        return

    try:
        if not scalar_bar:
            plotter.remove_scalar_bar(title=None)
    except Exception:
        pass

    try:
        rec["actor"] = actor
    except Exception:
        pass

    try:
        plotter.render()
    except Exception:
        pass

# ---------------------------------------------------------------------
# Plugin class
# ---------------------------------------------------------------------
class VTKImportPlugin(QtCore.QObject):
    """
    Plugin “VTK Import & Display”.

    Structure:
      - Menu action + shortcut for import
      - Display control box, applied to the current dataset
      - Best effort detection of time-series via PyVista reader
    """
    def __init__(self, window):
        super().__init__(window)
        self.window = window

        # UI state (for current dataset - updates on each selection in the tree)
        self._current_ds: Optional[int] = None
        self._time_values: Optional[List[float]] = None
        self._time_idx: int = 0

        # ----- Menu & shortcuts --------------------------------------
        self._action_import = QtGui.QAction(QtGui.QIcon(), "Import VTK…", self)
        # Shortcut: ⌘⇧I (mac) / Ctrl+Shift+I (others)
        self._action_import.setShortcut(QtGui.QKeySequence("Ctrl+Shift+I"))
        # Note: Qt automatically adjusts shortcuts on macOS to use Cmd based on the platform shortcut context
        self._action_import.triggered.connect(self.open_dialog)

        # Aggiungi in File
        try:
            mb = window.menuBar()
            for a in mb.actions():
                if a.text().replace("&", "") == "File":
                    a.menu().addAction(self._action_import)
                    break
        except Exception:
            pass

        # ----- Box UI in DisplayPanel ---------------------------------
        self._panel = self._build_display_box()
        _add_to_display_panel(window, "VTK Display", self._panel)

        # Keep the controls synchronized with the selected dataset
        self.window.treeMCTS.itemSelectionChanged.connect(self._on_tree_selection_changed)
        try:
            window.treeMCTS.itemSelectionChanged.connect(self._on_tree_selection_changed)
        except Exception:
            pass

    # -----------------------------------------------------------------
    # DIALOG DI IMPORT
    # -----------------------------------------------------------------
    @QtCore.Slot()
    def open_dialog(self):
        """
        Dialog to select a SINGLE file and import it using PyVista.
        Handles MultiBlock → SINGLE actor.
        Detects (if present) a time-series and exposes a slider.
        """
        dlg = QtWidgets.QFileDialog(self.window, "Import VTK")
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        dlg.setNameFilters([
            "All supported (*.vtk *.vtp *.vtu *.vtr *.vts *.vtm *.vti *.obj *.stl)",
            "VTK legacy (*.vtk)",
            "VTK XML PolyData (*.vtp)",
            "VTK XML UnstructuredGrid (*.vtu)",
            "VTK XML RectilinearGrid (*.vtr)",
            "VTK XML StructuredGrid (*.vts)",
            "VTK XML MultiBlock (*.vtm)",
            "VTK ImageData (*.vti)",
            "Meshes (*.obj *.stl)",
            "All files (*)",
        ])
        if not dlg.exec():
            return
        paths = dlg.selectedFiles()
        if not paths:
            return
        path = paths[0]

        try:
            reader = pv.get_reader(path)
        except Exception as ex:
            self._msg(f"[VTK] Reader error: {ex}", error=True)
            return

        # Try to detect time values
        self._time_values = None
        try:
            tvals = getattr(reader, "time_values", None)
            if tvals is not None and len(tvals) > 0:
                self._time_values = list(tvals)
                reader.set_active_time_value(self._time_values[0])
        except Exception:
            self._time_values = None

        try:
            data = reader.read()
        except Exception as ex:
            self._msg(f"[VTK] Read error: {ex}", error=True)
            return

        # MultiBlock → unico attore
        dataset_to_add = data
        try:
            if isinstance(data, pv.MultiBlock):
                # If you really want *a single actor*, you can render the MultiBlock directly
                # (PyVista internally handles the blocks with a single composite mapper/actor).
                dataset_to_add = data
        except Exception:
            pass

        # Add to viewer
        try:
            ds_index = self._add_dataset_to_viewer(dataset_to_add, path)
        except Exception as ex:
            self._msg(f"[VTK] Viewer add failed: {ex}", error=True)
            return

        # Fit camera
        try:
            self.window.viewer3d.view_fit()
        except Exception:
            pass

        # Select the new dataset in the tree
        try:
            self._select_tree_item_for_ds(ds_index)
        except Exception:
            pass

        # Show/update time slider if needed
        self._sync_time_slider_visibility()

        # Update Inspector
        try:
            self.window._refresh_inspector_tree()
        except Exception:
            pass

        self._msg(f"[VTK] Imported: {os.path.basename(path)}")

    def _add_dataset_to_viewer(self, data, path: str) -> int:
        """
        Aggiunge il dataset PyVista al Viewer3D e registra in mcts.
        Ritorna l'indice di dataset (ds_index).
        """
        name = os.path.splitext(os.path.basename(path))[0]

        # Prefer dedicated APIs if available
        ds_index = None
        try:
            if hasattr(self.window.viewer3d, "add_pyvista_mesh"):
                ds_index = self.window.viewer3d.add_pyvista_mesh(data)
            else:
                # Generic fallback (used if add_pyvista_mesh is not available)
                actor = self.window.viewer3d.plotter.add_mesh(data, name=name)
                # Manual registration in viewer3d._datasets if necessary
                # Assuming the official API does not handle this automatically
                if not hasattr(self.window.viewer3d, "_datasets"):
                    self.window.viewer3d._datasets = []
                ds_index = len(self.window.viewer3d._datasets)
                self.window.viewer3d._datasets.append({"mesh": data, "actor": actor})
                raise RuntimeError("Viewer3D.add_pyvista_mesh not available; used fallback registration.")
        except Exception as ex:
            raise

        # Register in mcts (new instance always at the end of the list)
        entry = {
            "name": name,
            "kind": "mesh",            # PolyData / Grid / MultiBlock → keep as "mesh"
            "ds_index": ds_index,
            "source_path": path,       # automatic reopening
            # default initial style
            "representation": "Surface",
            "opacity": 100,
            "color_mode": "Solid Color",
            "solid_color": (255, 255, 255),
            "colormap": "Viridis",
            "scalar_bar": False,
            "edge_visibility": False,
            "edge_color": (0, 0, 0),
            "point_size": 3,
            "line_width": 1,
            "lighting": True,
        }
        self.window.mcts[name] = entry
        self.window.mct = entry  # becomes "current"

        # Create tree node if needed (reuse MainWindow pipeline if available)
        try:
            # If the official MainWindow import already builds the tree, you might skip this part.
            # Here we construct a minimal node as an example:
            self.window.treeMCTS.blockSignals(True)
            root = QtWidgets.QTreeWidgetItem([name])
            root.setFlags(root.flags() | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsAutoTristate)
            root.setCheckState(0, QtCore.Qt.Checked)
            # metadata sul nodo root
            root.setData(0, QtCore.Qt.UserRole, {"kind": "mesh", "ds": ds_index})
            # Aggiungi albero “Mesh”
            it_mesh = QtWidgets.QTreeWidgetItem(["Mesh"])
            it_mesh.setFlags(it_mesh.flags() | QtCore.Qt.ItemIsUserCheckable)
            it_mesh.setCheckState(0, QtCore.Qt.Checked)
            it_mesh.setData(0, QtCore.Qt.UserRole, {"kind": "mesh", "ds": ds_index})
            root.addChild(it_mesh)

            self.window.treeMCTS.addTopLevelItem(root)
            self.window.treeMCTS.blockSignals(False)
        except Exception:
            pass

        return ds_index

    # -----------------------------------------------------------------
    # BOX CONTROLS (ParaView, LIVE application)
    # -----------------------------------------------------------------
    def _build_display_box(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        w.setMaximumWidth(300)
        w.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        lay = QtWidgets.QFormLayout(w)
        lay.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        lay.setHorizontalSpacing(6)
        lay.setVerticalSpacing(6)
        lay.setContentsMargins(6, 6, 6, 6)

        # Representation
        self.cmbRep = QtWidgets.QComboBox()
        self.cmbRep.addItems(["Points", "Wireframe", "Surface", "Surface with Edges", "Volume"])
        self.cmbRep.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.cmbRep.currentTextChanged.connect(self._on_rep_changed)
        lay.addRow("Representation", self.cmbRep)

        # Color By
        self.cmbColorBy = QtWidgets.QComboBox()
        self.cmbColorBy.setMinimumWidth(120)
        self.cmbColorBy.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.cmbColorBy.currentTextChanged.connect(self._on_color_by_changed)
        lay.addRow("Color by", self.cmbColorBy)

        # Vector component
        self.cmbVectorMode = QtWidgets.QComboBox()
        self.cmbVectorMode.addItems(["Magnitude", "X", "Y", "Z"])
        self.cmbVectorMode.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.cmbVectorMode.currentTextChanged.connect(self._on_color_by_changed)
        lay.addRow("Vector component", self.cmbVectorMode)

        # LUT
        self.cmbLUT = QtWidgets.QComboBox()
        self.cmbLUT.addItems(["Viridis", "Plasma", "CoolWarm", "Gray", "Jet"])
        self.cmbLUT.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.cmbLUT.currentTextChanged.connect(self._on_color_by_changed)
        lay.addRow("LUT", self.cmbLUT)

        # Invert LUT
        self.chkInvertLUT = QtWidgets.QCheckBox("Invert")
        self.chkInvertLUT.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.chkInvertLUT.toggled.connect(self._on_color_by_changed)
        lay.addRow("", self.chkInvertLUT)

        # Solid Color button
        self.btnSolid = _solid_color_button()
        self.btnSolid.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.btnSolid.clicked.connect(self._on_pick_solid_color)
        lay.addRow(self.btnSolid)

        # Scalar range with min/max and buttons
        rngw = QtWidgets.QWidget()
        rngw.setMaximumWidth(260)
        rngw.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        rngLay = QtWidgets.QVBoxLayout(rngw)
        rngLay.setContentsMargins(0,0,0,0)
        rngLay.setSpacing(4)
        frm = QtWidgets.QFormLayout()
        frm.setContentsMargins(0,0,0,0)
        frm.setSpacing(4)
        self.editMin = QtWidgets.QLineEdit()
        self.editMin.setPlaceholderText("min")
        self.editMin.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.editMax = QtWidgets.QLineEdit()
        self.editMax.setPlaceholderText("max")
        self.editMax.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        frm.addRow("Min", self.editMin)
        frm.addRow("Max", self.editMax)
        rngLay.addLayout(frm)
        btnRow = QtWidgets.QHBoxLayout()
        btnRow.setSpacing(4)
        self.btnAuto = QtWidgets.QPushButton("Auto")
        self.btnAuto.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.btnAuto.clicked.connect(self._on_range_auto)
        self.btnRescale = QtWidgets.QPushButton("Rescale")
        self.btnRescale.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.btnRescale.clicked.connect(self._on_rescale_to_data)
        btnRow.addWidget(self.btnAuto)
        btnRow.addWidget(self.btnRescale)
        rngLay.addLayout(btnRow)
        lay.addRow("Scalar range", rngw)

        # Scalar bar
        self.chkScalarBar = QtWidgets.QCheckBox("Show scalar bar")
        self.chkScalarBar.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.chkScalarBar.toggled.connect(self._on_scalar_bar_toggle)
        lay.addRow(self.chkScalarBar)

        # Opacity slider
        self.sldOpacity = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sldOpacity.setRange(0, 100)
        self.sldOpacity.setValue(100)
        self.sldOpacity.setMaximumWidth(260)
        self.sldOpacity.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.sldOpacity.valueChanged.connect(self._on_opacity_changed)
        lay.addRow("Opacity", self.sldOpacity)

        # Point size slider
        self.sldPointSize = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sldPointSize.setRange(1, 15)
        self.sldPointSize.setValue(3)
        self.sldPointSize.setMaximumWidth(260)
        self.sldPointSize.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.sldPointSize.valueChanged.connect(self._on_point_size_changed)
        lay.addRow("Point size", self.sldPointSize)

        # Line width slider
        self.sldLineWidth = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sldLineWidth.setRange(1, 10)
        self.sldLineWidth.setValue(1)
        self.sldLineWidth.setMaximumWidth(260)
        self.sldLineWidth.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.sldLineWidth.valueChanged.connect(self._on_line_width_changed)
        lay.addRow("Line width", self.sldLineWidth)

        # Edges visible
        self.chkEdges = QtWidgets.QCheckBox("Edges visible")
        self.chkEdges.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.chkEdges.toggled.connect(self._on_edges_toggle)
        lay.addRow(self.chkEdges)

        # Edge color button
        self.btnEdgeColor = QtWidgets.QPushButton("Edge color…")
        self.btnEdgeColor.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.btnEdgeColor.clicked.connect(self._on_pick_edge_color)
        lay.addRow(self.btnEdgeColor)

        # Lighting
        self.chkLighting = QtWidgets.QCheckBox("Lighting")
        self.chkLighting.setChecked(True)
        self.chkLighting.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.chkLighting.toggled.connect(self._on_lighting_toggle)
        lay.addRow(self.chkLighting)

        # Time-series group
        self.grpTime = QtWidgets.QGroupBox("Time")
        self.grpTime.setMaximumWidth(300)
        time_lay = QtWidgets.QVBoxLayout(self.grpTime)
        self.sldTime = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sldTime.setRange(0, 0)
        self.sldTime.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.sldTime.valueChanged.connect(self._on_time_changed)
        self.lblTime = QtWidgets.QLabel("—")
        self.lblTime.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.lblTime.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        time_lay.addWidget(self.sldTime)
        time_lay.addWidget(self.lblTime)
        self.grpTime.setVisible(False)
        lay.addRow(self.grpTime)

        # Reset button
        self.btnReset = QtWidgets.QPushButton("Reset defaults")
        self.btnReset.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.btnReset.clicked.connect(self._on_reset_defaults)
        lay.addRow(self.btnReset)

        return w

    # -----------------------------------------------------------------
    # SYNC UI with current dataset / MCT entry
    # -----------------------------------------------------------------
    def _on_tree_selection_changed(self):
        ds = self._current_dataset_index()
        self._current_ds = ds
        self._rebuild_colorby_combo()
        # carica stato dal mct (se presente)
        entry = self._current_mct()
        if entry:
            self._load_ui_from_entry(entry)

    def _current_dataset_index(self) -> Optional[int]:
        try:
            return self.window._current_dataset_index()
        except Exception:
            return None

    def _current_mct(self) -> Optional[dict]:
        try:
            ds = self._current_dataset_index()
            for e in self.window.mcts.values():
                if e.get("ds_index") == ds:
                    return e
            return self.window.mct if self.window.mct.get("ds_index") == ds else None
        except Exception:
            return None

    def _select_tree_item_for_ds(self, ds_index: int) -> None:
        """Try to select the root item in the tree with ds=ds_index."""
        t = self.window.treeMCTS
        for i in range(t.topLevelItemCount()):
            root = t.topLevelItem(i)
            data = root.data(0, QtCore.Qt.UserRole)
            if isinstance(data, dict) and data.get("ds") == ds_index:
                t.setCurrentItem(root)
                break

    def _sync_time_slider_visibility(self) -> None:
        has_time = bool(self._time_values) and len(self._time_values) > 1
        self.grpTime.setVisible(has_time)
        if has_time:
            self.sldTime.blockSignals(True)
            self.sldTime.setRange(0, len(self._time_values) - 1)
            self.sldTime.setValue(0)
            self.sldTime.blockSignals(False)
            self.lblTime.setText(f"t = {self._time_values[0]:.6g}")

    # -----------------------------------------------------------------
    # UI EVENTS → APPLY
    # -----------------------------------------------------------------
    def _on_rep_changed(self, mode: str):
        ds = self._current_dataset_index()
        if ds is None: return
        # Mappatura semplice
        self._apply_representation(ds, mode)
        self._save_to_mct("representation", mode)

    def _on_color_by_changed(self):
        ds = self._current_dataset_index()
        if ds is None: return
        label = self.cmbColorBy.currentText()
        vec_mode = self.cmbVectorMode.currentText()
        lut = self.cmbLUT.currentText()
        invert = self.chkInvertLUT.isChecked()
        self._apply_coloring(ds, label, vec_mode, lut, invert)

        self._save_to_mct("color_mode", label)
        self._save_to_mct("colormap", lut)
        self._save_to_mct("invert_lut", bool(invert))
        self._save_to_mct("vector_mode", vec_mode)

    def _on_pick_solid_color(self):
        col = QtWidgets.QColorDialog.getColor(parent=self.window, title="Solid Color")
        if not col.isValid(): return
        ds = self._current_dataset_index()
        if ds is None: return
        self._apply_solid_color(ds, (col.red(), col.green(), col.blue()))
        self._save_to_mct("color_mode", "Solid Color")
        self._save_to_mct("solid_color", (col.red(), col.green(), col.blue()))
        # Force combo box to "Solid Color"
        self.cmbColorBy.blockSignals(True)
        self.cmbColorBy.setCurrentText("Solid Color")
        self.cmbColorBy.blockSignals(False)

    def _on_range_auto(self):
        self.editMin.clear(); self.editMax.clear()
        self._on_color_by_changed()  # re-apply with auto-range

    def _on_rescale_to_data(self):
        # Re-apply colormap asking the viewer to use data range
        self._on_color_by_changed()

    def _on_scalar_bar_toggle(self, on: bool):
        ds = self._current_dataset_index()
        if ds is None: return
        self._apply_scalar_bar(ds, on)
        self._save_to_mct("scalar_bar", bool(on))

    def _on_opacity_changed(self, val: int):
        ds = self._current_dataset_index()
        if ds is None: return
        self._apply_opacity(ds, val)
        self._save_to_mct("opacity", int(val))

    def _on_point_size_changed(self, val: int):
        ds = self._current_dataset_index()
        if ds is None: return
        self._apply_point_size(ds, val)
        self._save_to_mct("point_size", int(val))

    def _on_line_width_changed(self, val: int):
        ds = self._current_dataset_index()
        if ds is None: return
        self._apply_line_width(ds, val)
        self._save_to_mct("line_width", int(val))

    def _on_edges_toggle(self, on: bool):
        ds = self._current_dataset_index()
        if ds is None: return
        self._apply_edges(ds, bool(on))
        self._save_to_mct("edge_visibility", bool(on))

    def _on_pick_edge_color(self):
        col = QtWidgets.QColorDialog.getColor(parent=self.window, title="Edge Color")
        if not col.isValid(): return
        ds = self._current_dataset_index()
        if ds is None: return
        self._apply_edge_color(ds, (col.red(), col.green(), col.blue()))
        self._save_to_mct("edge_color", (col.red(), col.green(), col.blue()))

    def _on_lighting_toggle(self, on: bool):
        ds = self._current_dataset_index()
        if ds is None: return
        self._apply_lighting(ds, bool(on))
        self._save_to_mct("lighting", bool(on))

    def _on_time_changed(self, idx: int):
        if not self._time_values: return
        self._time_idx = int(idx)
        t = self._time_values[self._time_idx]
        self.lblTime.setText(f"t = {t:.6g}")
        # Rileggi il dataset al tempo selezionato
        # NOTE: servirebbe conservare il reader in self; per semplicità omesso.
        # TODO: estendere per ricaricare dal reader e aggiornare l'attore.

    def _on_reset_defaults(self):
        # Reset UI
        self.cmbRep.setCurrentText("Surface")
        self.cmbColorBy.setCurrentText("Solid Color")
        self.cmbVectorMode.setCurrentText("Magnitude")
        self.cmbLUT.setCurrentText("Viridis")
        self.chkInvertLUT.setChecked(False)
        self.chkScalarBar.setChecked(False)
        self.sldOpacity.setValue(100)
        self.sldPointSize.setValue(3)
        self.sldLineWidth.setValue(1)
        self.chkEdges.setChecked(False)
        self.chkLighting.setChecked(True)
        # Applica allo stato corrente
        self._on_rep_changed("Surface")
        self._on_color_by_changed()
        self._on_scalar_bar_toggle(False)
        self._on_opacity_changed(100)
        self._on_point_size_changed(3)
        self._on_line_width_changed(1)
        self._on_edges_toggle(False)
        self._on_lighting_toggle(True)

    # -----------------------------------------------------------------
    # APPLY (adapter using the viewer's API if available, else fallback)
    # -----------------------------------------------------------------
    def _apply_representation(self, ds: int, mode: str):
        """
        Map of representations:
        - Points, Wireframe, Surface, Surface with Edges, Volume
        """
        # Se il tuo Viewer3D espone un metodo diretto:
        fn = getattr(self.window.viewer3d, "set_mesh_representation", None)
        if callable(fn):
            fn(ds, mode)
            return
        # Fallback: re-render via PyVista
        e = self._current_mct() or {}
        _fallback_render(
            self.window, e, ds,
            representation=mode,
            manual_range=self._manual_range_or_none(),
            color_mode=e.get("color_mode", "Solid Color"),
            lut=e.get("colormap", "Viridis"),
            invert=self.chkInvertLUT.isChecked(),
            scalar_bar=self.chkScalarBar.isChecked(),
            edge_visibility=self.chkEdges.isChecked(),
            edge_color=e.get("edge_color", (0, 0, 0)),
            opacity=self.sldOpacity.value(),
            point_size=self.sldPointSize.value(),
            line_width=self.sldLineWidth.value(),
            lighting=self.chkLighting.isChecked(),
            vector_mode=self.cmbVectorMode.currentText(),
        )

    def _apply_solid_color(self, ds: int, rgb: Tuple[int, int, int]):
        fn = getattr(self.window.viewer3d, "set_dataset_color", None)
        if callable(fn):
            fn(ds, *rgb)
            return
        e = self._current_mct() or {}
        _fallback_render(
            self.window, e, ds,
            color_mode="Solid Color",
            solid_color=rgb,
            representation=e.get("representation", "Surface"),
            lut=e.get("colormap", "Viridis"),
            invert=self.chkInvertLUT.isChecked(),
            scalar_bar=self.chkScalarBar.isChecked(),
            edge_visibility=self.chkEdges.isChecked(),
            edge_color=e.get("edge_color", (0, 0, 0)),
            opacity=self.sldOpacity.value(),
            point_size=self.sldPointSize.value(),
            line_width=self.sldLineWidth.value(),
            lighting=self.chkLighting.isChecked(),
            manual_range=self._manual_range_or_none(),
            vector_mode=self.cmbVectorMode.currentText(),
        )

    def _apply_coloring(self, ds: int, label: str, vec_mode: str, lut: str, invert: bool):
        """
        label = "Solid Color" oppure "PointData/<array>" o "CellData/<array>"
        vec_mode = Magnitude / X / Y / Z
        """
        # Caso Solid Color → forza colore uniforme
        if label == "Solid Color":
            self._apply_solid_color(ds, self._current_mct().get("solid_color", (255, 255, 255)))
            # Se il viewer ha un “color mode”, impostalo
            try:
                self.window.viewer3d.set_color_mode("Solid Color", ds)
            except Exception:
                pass
            return

        # Parsing “PointData/NAME” o “CellData/NAME”
        assoc = "POINT"
        array_name = label
        if label.startswith("PointData/"):
            assoc = "POINT"
            array_name = label.split("/", 1)[1]
        elif label.startswith("CellData/"):
            assoc = "CELL"
            array_name = label.split("/", 1)[1]

        # Viewer API personalizzata (se esiste):
        # Immaginiamo una API del tipo: set_scalar_coloring(ds, array_name, assoc, component, lut, invert, range)
        fn = getattr(self.window.viewer3d, "set_scalar_coloring", None)
        rng = self._manual_range_or_none()
        component = {"Magnitude": None, "X": 0, "Y": 1, "Z": 2}.get(vec_mode, None)
        if callable(fn):
            fn(ds, array_name, assoc, component, lut, bool(invert), rng)
            return

        e = self._current_mct() or {}
        _fallback_render(
            self.window, e, ds,
            color_mode=label,
            lut=lut,
            invert=bool(invert),
            representation=e.get("representation", "Surface"),
            scalar_bar=self.chkScalarBar.isChecked(),
            edge_visibility=self.chkEdges.isChecked(),
            edge_color=e.get("edge_color", (0, 0, 0)),
            opacity=self.sldOpacity.value(),
            point_size=self.sldPointSize.value(),
            line_width=self.sldLineWidth.value(),
            lighting=self.chkLighting.isChecked(),
            manual_range=self._manual_range_or_none(),
            vector_mode=self.cmbVectorMode.currentText(),
        )

    def _manual_range_or_none(self) -> Optional[Tuple[float, float]]:
        try:
            smin = self.editMin.text().strip()
            smax = self.editMax.text().strip()
            if not smin or not smax:
                return None
            return (float(smin), float(smax))
        except Exception:
            return None

    def _apply_scalar_bar(self, ds: int, show: bool):
        fn = getattr(self.window.viewer3d, "set_scalar_bar_visible", None)
        if callable(fn):
            fn(ds, bool(show))
            return
        e = self._current_mct() or {}
        _fallback_render(
            self.window, e, ds,
            scalar_bar=bool(show),
            representation=e.get("representation", "Surface"),
            color_mode=e.get("color_mode", "Solid Color"),
            lut=e.get("colormap", "Viridis"),
            invert=self.chkInvertLUT.isChecked(),
            edge_visibility=self.chkEdges.isChecked(),
            edge_color=e.get("edge_color", (0, 0, 0)),
            opacity=self.sldOpacity.value(),
            point_size=self.sldPointSize.value(),
            line_width=self.sldLineWidth.value(),
            lighting=self.chkLighting.isChecked(),
            manual_range=self._manual_range_or_none(),
            vector_mode=self.cmbVectorMode.currentText(),
        )

    def _apply_opacity(self, ds: int, val: int):
        fn = getattr(self.window.viewer3d, "set_mesh_opacity", None)
        if callable(fn):
            fn(ds, int(val))
            return
        e = self._current_mct() or {}
        _fallback_render(
            self.window, e, ds,
            opacity=int(val),
            representation=e.get("representation", "Surface"),
            color_mode=e.get("color_mode", "Solid Color"),
            lut=e.get("colormap", "Viridis"),
            invert=self.chkInvertLUT.isChecked(),
            scalar_bar=self.chkScalarBar.isChecked(),
            edge_visibility=self.chkEdges.isChecked(),
            edge_color=e.get("edge_color", (0, 0, 0)),
            point_size=self.sldPointSize.value(),
            line_width=self.sldLineWidth.value(),
            lighting=self.chkLighting.isChecked(),
            manual_range=self._manual_range_or_none(),
            vector_mode=self.cmbVectorMode.currentText(),
        )

    def _apply_point_size(self, ds: int, val: int):
        fn = getattr(self.window.viewer3d, "set_point_size", None)
        if callable(fn):
            fn(int(val), ds)
            return
        e = self._current_mct() or {}
        _fallback_render(
            self.window, e, ds,
            point_size=int(val),
            representation=e.get("representation", "Surface"),
            color_mode=e.get("color_mode", "Solid Color"),
            lut=e.get("colormap", "Viridis"),
            invert=self.chkInvertLUT.isChecked(),
            scalar_bar=self.chkScalarBar.isChecked(),
            edge_visibility=self.chkEdges.isChecked(),
            edge_color=e.get("edge_color", (0, 0, 0)),
            opacity=self.sldOpacity.value(),
            line_width=self.sldLineWidth.value(),
            lighting=self.chkLighting.isChecked(),
            manual_range=self._manual_range_or_none(),
            vector_mode=self.cmbVectorMode.currentText(),
        )

    def _apply_line_width(self, ds: int, val: int):
        fn = getattr(self.window.viewer3d, "set_line_width", None)
        if callable(fn):
            fn(ds, int(val))
            return
        e = self._current_mct() or {}
        _fallback_render(
            self.window, e, ds,
            line_width=int(val),
            representation=e.get("representation", "Surface"),
            color_mode=e.get("color_mode", "Solid Color"),
            lut=e.get("colormap", "Viridis"),
            invert=self.chkInvertLUT.isChecked(),
            scalar_bar=self.chkScalarBar.isChecked(),
            edge_visibility=self.chkEdges.isChecked(),
            edge_color=e.get("edge_color", (0, 0, 0)),
            opacity=self.sldOpacity.value(),
            point_size=self.sldPointSize.value(),
            lighting=self.chkLighting.isChecked(),
            manual_range=self._manual_range_or_none(),
            vector_mode=self.cmbVectorMode.currentText(),
        )

    def _apply_edges(self, ds: int, on: bool):
        fn = getattr(self.window.viewer3d, "set_edge_visibility", None)
        if callable(fn):
            fn(ds, bool(on))
            return
        e = self._current_mct() or {}
        _fallback_render(
            self.window, e, ds,
            edge_visibility=bool(on),
            representation=e.get("representation", "Surface"),
            color_mode=e.get("color_mode", "Solid Color"),
            lut=e.get("colormap", "Viridis"),
            invert=self.chkInvertLUT.isChecked(),
            scalar_bar=self.chkScalarBar.isChecked(),
            edge_color=e.get("edge_color", (0, 0, 0)),
            opacity=self.sldOpacity.value(),
            point_size=self.sldPointSize.value(),
            line_width=self.sldLineWidth.value(),
            lighting=self.chkLighting.isChecked(),
            manual_range=self._manual_range_or_none(),
            vector_mode=self.cmbVectorMode.currentText(),
        )

    def _apply_edge_color(self, ds: int, rgb: Tuple[int, int, int]):
        fn = getattr(self.window.viewer3d, "set_edge_color", None)
        if callable(fn):
            fn(ds, *rgb)
            return
        e = self._current_mct() or {}
        _fallback_render(
            self.window, e, ds,
            edge_color=rgb,
            representation=e.get("representation", "Surface"),
            color_mode=e.get("color_mode", "Solid Color"),
            lut=e.get("colormap", "Viridis"),
            invert=self.chkInvertLUT.isChecked(),
            scalar_bar=self.chkScalarBar.isChecked(),
            edge_visibility=self.chkEdges.isChecked(),
            opacity=self.sldOpacity.value(),
            point_size=self.sldPointSize.value(),
            line_width=self.sldLineWidth.value(),
            lighting=self.chkLighting.isChecked(),
            manual_range=self._manual_range_or_none(),
            vector_mode=self.cmbVectorMode.currentText(),
        )

    def _apply_lighting(self, ds: int, on: bool):
        fn = getattr(self.window.viewer3d, "set_lighting_enabled", None)
        if callable(fn):
            fn(ds, bool(on))
            return
        e = self._current_mct() or {}
        _fallback_render(
            self.window, e, ds,
            lighting=bool(on),
            representation=e.get("representation", "Surface"),
            color_mode=e.get("color_mode", "Solid Color"),
            lut=e.get("colormap", "Viridis"),
            invert=self.chkInvertLUT.isChecked(),
            scalar_bar=self.chkScalarBar.isChecked(),
            edge_visibility=self.chkEdges.isChecked(),
            edge_color=e.get("edge_color", (0, 0, 0)),
            opacity=self.sldOpacity.value(),
            point_size=self.sldPointSize.value(),
            line_width=self.sldLineWidth.value(),
            manual_range=self._manual_range_or_none(),
            vector_mode=self.cmbVectorMode.currentText(),
        )

    # -----------------------------------------------------------------
    # COMBO BOX "Color by" REBUILD
    # -----------------------------------------------------------------
    def _rebuild_colorby_combo(self):
        """Reads arrays from the current PolyData/mesh and populates the “Color by” combo box."""
        self.cmbColorBy.blockSignals(True)
        self.cmbColorBy.clear()
        self.cmbColorBy.addItem("Solid Color")
        ds = self._current_dataset_index()
        if ds is None:
            self.cmbColorBy.blockSignals(False)
            return

        # if possible, retrieve arrays from the original dataset
        arrays_pt, arrays_cell = [], []
        try:
            recs = getattr(self.window.viewer3d, "_datasets", [])
            rec = recs[ds]
            # Prefer original dataset for listing arrays (more complete), fall back to surface
            pdata = rec.get("mesh_orig") or rec.get("mesh") or rec.get("pdata") or rec.get("full_pdata")
            surf = rec.get("mesh_surface") or rec.get("mesh") or rec.get("pdata") or rec.get("full_pdata")
            target = pdata or surf
            if isinstance(target, pv.MultiBlock):
                try:
                    target = target[0]
                except Exception:
                    target = None
                rec["mesh"] = target
            if target is not None:
                # PointData
                try:
                    for name in list(target.point_data.keys()):
                        if str(name).startswith("vtkOriginal"):
                            continue
                        arrays_pt.append(str(name))
                except Exception:
                    pass
                # CellData
                try:
                    for name in list(target.cell_data.keys()):
                        if str(name).startswith("vtkOriginal"):
                            continue
                        arrays_cell.append(str(name))
                except Exception:
                    pass
        except Exception:
            pass

        if arrays_pt:
            for n in arrays_pt:
                self.cmbColorBy.addItem(f"PointData/{n}")
        if arrays_cell:
            for n in arrays_cell:
                self.cmbColorBy.addItem(f"CellData/{n}")

        self.cmbColorBy.blockSignals(False)

    def _load_ui_from_entry(self, e: dict) -> None:
        """Load (best effort) controls from persisted values in the mct entry."""
        try: self.cmbRep.setCurrentText(e.get("representation", "Surface"))
        except Exception: pass
        try: self.cmbColorBy.setCurrentText(e.get("color_mode", "Solid Color"))
        except Exception: pass
        try: self.cmbLUT.setCurrentText(e.get("colormap", "Viridis"))
        except Exception: pass
        try: self.chkScalarBar.setChecked(bool(e.get("scalar_bar", False)))
        except Exception: pass
        try: self.sldOpacity.setValue(int(e.get("opacity", 100)))
        except Exception: pass
        try: self.sldPointSize.setValue(int(e.get("point_size", 3)))
        except Exception: pass
        try: self.sldLineWidth.setValue(int(e.get("line_width", 1)))
        except Exception: pass
        try: self.chkEdges.setChecked(bool(e.get("edge_visibility", False)))
        except Exception: pass
        try: self.chkLighting.setChecked(bool(e.get("lighting", True)))
        except Exception: pass

    def _save_to_mct(self, key: str, val):
        e = self._current_mct()
        if e is not None:
            e[key] = val

    # -----------------------------------------------------------------
    # LOGGING/UTILITY
    # -----------------------------------------------------------------
    def _msg(self, text: str, error: bool = False):
        # Write to status bar + message panel (if available)
        try:
            self.window.statusBar().showMessage(text, 5000)
        except Exception:
            pass
        try:
            self.window.txtMessages.appendPlainText(text)
        except Exception:
            pass
        if error:
            print(text)

# ---------------------------------------------------------------------
# ENTRY POINT per PluginManager
# ---------------------------------------------------------------------
def register(window) -> object:
    """
    Factory entry-point required by `entry_point: "plugin:register"`.
    Returns the plugin instance.
    """
    return VTKImportPlugin(window)