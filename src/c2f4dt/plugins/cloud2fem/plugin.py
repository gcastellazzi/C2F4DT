# -*- coding: utf-8 -*-
"""
Cloud2FEM – Slicing & Grid plugin for C2F4DT.

WHAT THE PLUGIN DOES
--------------------
- Adds a "Cloud2FEM" control box into the SLICING tab (right-side scroll area).
- Lets the user define slicing direction (X/Y/Z/Custom via two points), thickness,
  slicing modes (fixed Δ, fixed number, custom rule), and range (start/end).
- Optionally produces products: Centroids, Raw polylines, Refined polylines, Polygons.
- Provides actions to: Compute Slices, Generate Grid from Slices, Build FEM from Grid.
- All long-running ops report progress and honor CANCEL via the main window.

Design notes
------------
- The plugin does not modify MainWindow nor the viewer; it discovers
  the SLICING panel at runtime and attaches its own widget there.
- Heavy geometry operations are left as TODO hooks to be connected with
  Cloud2FEMi functions.

Author: C2F4DT Team
License: same as C2F4DT
"""
from __future__ import annotations
from typing import Optional


from PySide6 import QtCore, QtWidgets, QtGui

# ---------------------------------------------------------------------
# PluginManager metadata & entry point (compatible with PluginManager)
# ---------------------------------------------------------------------
class _PluginMetaShim:
    """Lightweight metadata object compatible with PluginManager.

    Only attributes accessed by the manager are defined here. We keep it
    local to avoid importing the manager types from within the plugin.
    """
    name = "cloud2fem"          # public name and folder name
    order = 60                   # load order (lower loads first)
    requires = tuple()           # optional runtime requirements
    version = "0.1.0"           # optional version label for UI

# Expose a PLUGIN object so PluginManager can read metadata without YAML
PLUGIN = _PluginMetaShim()


def load_plugin(parent):
    """Factory used by PluginManager to instantiate the plugin.

    Args:
        parent: Host main window (or app) instance passed by the manager.

    Returns:
        Cloud2FEMPlugin: The instantiated plugin.
    """
    return Cloud2FEMPlugin(parent)

# Backward compatibility (if older code calls register(window))
register = load_plugin

# ---------------------------------------------------------------------
# Helpers to attach widgets into the host SLICING tab
# ---------------------------------------------------------------------

def _add_to_slicing_panel(window, title: str, widget: QtWidgets.QWidget) -> None:
    """Attach *widget* into the SLICING tab, with a nice titled box.

    The function prefers a dedicated API if the host provides it. Otherwise
    it falls back to inserting into the scroll area container.

    Args:
        window: Main window instance.
        title: Section title.
        widget: The widget to add.
    """
    # Idempotency guard: avoid adding the Slicing panel twice
    if getattr(window, "_cloud2fem_slicing_installed", False):
        return
    # Preferred: if there's a specialized panel with API `add_plugin_section`.
    try:
        slicing_panel = getattr(window, "slicingPanel", None)
        if slicing_panel and hasattr(slicing_panel, "add_plugin_section"):
            # Try to avoid duplicates if host keeps previously added sections
            try:
                if hasattr(slicing_panel, "findChildren"):
                    for gb in slicing_panel.findChildren(QtWidgets.QGroupBox):
                        if gb.objectName() == "cloud2fem.slicing_box":
                            return
            except Exception:
                pass
            # Mark inner widget and box for future detection
            widget.setObjectName("cloud2fem.slicing_widget")
            slicing_panel.add_plugin_section(title, widget)
            try:
                window._cloud2fem_slicing_installed = True  # type: ignore[attr-defined]
            except Exception:
                pass
            return
    except Exception:
        pass

    # Fallback: use the scroll area `scrollSLICING` if available.
    try:
        scroll = getattr(window, "scrollSLICING", None)
        if isinstance(scroll, QtWidgets.QScrollArea):
            # Ensure a container widget with a VBox layout is present
            container = scroll.widget()
            if container is None:
                container = QtWidgets.QWidget()
                scroll.setWidget(container)
            if container.layout() is None:
                container.setLayout(QtWidgets.QVBoxLayout())

            # If already present, do not add again
            try:
                for gb in container.findChildren(QtWidgets.QGroupBox):
                    if gb.objectName() == "cloud2fem.slicing_box":
                        return
            except Exception:
                pass

            # Fallback: wrap the widget in a QGroupBox and append it to the main vertical layout.
            box = QtWidgets.QGroupBox(title)
            box.setObjectName("cloud2fem.slicing_box")
            box.setMaximumWidth(300)
            lay = QtWidgets.QVBoxLayout(box)
            lay.setContentsMargins(8, 8, 8, 8)
            lay.addWidget(widget)
            # Enforce the same layout/size policies as VTK box
            widget.setObjectName("cloud2fem.slicing_widget")
            widget.setMaximumWidth(300)
            widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
            box.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)

            container.layout().addWidget(box)
            container.layout().addStretch(0)
            try:
                window._cloud2fem_slicing_installed = True  # type: ignore[attr-defined]
            except Exception:
                pass
            return
    except Exception:
        pass

    # If everything else fails, try to mount as central widget (very unlikely path).
    try:
        window.setCentralWidget(widget)
    except Exception:
        pass


# ---------------------------------------------------------------------
# Plugin main class
# ---------------------------------------------------------------------
class Cloud2FEMPlugin(QtCore.QObject):
    """Plugin entry that mounts the Cloud2FEM slicing box into the SLICING tab.

    The plugin keeps a reference to the window to log and to use its
    progress/cancel mechanisms. All UI and commands are encapsulated here.
    """
    def __init__(self, window):
        super().__init__(window)
        self.window = window

        # Build and mount the panel
        self.panel = SlicingPanel(window)
        _add_to_slicing_panel(window, "Cloud2FEM", self.panel)

        # Optional: register useful shortcuts on the window
        self._install_shortcuts()

        # Install right-vertical toolbar buttons for quick toggles
        self._install_toolbar_buttons()

        self._log("INFO", "[cloud2fem] Slicing panel ready")

    # -------------------------- utilities ------------------------------
    def _install_shortcuts(self) -> None:
        """Install keyboard shortcuts bound to SlicingPanel actions."""
        try:
            mk = lambda seq, cb: QtWidgets.QShortcut(seq, self.window, activated=cb)
            mk(QtGui.QKeySequence("Ctrl+K"), self.panel._on_compute_slices)  # Cluster/Compute slices idea
            mk(QtGui.QKeySequence("Ctrl+B"), self.panel._on_grid_from_slices)
            mk(QtGui.QKeySequence("Ctrl+E"), self.panel._on_fem_from_grid)
        except Exception:
            pass

    def _install_toolbar_buttons(self) -> None:
        """Create checkable actions on the right vertical toolbar to toggle visibility.

        Buttons:
            - Slices
            - Current slice
            - Centroids
            - Polylines
            - Polygons
            - Mesh (C2F)
        """
        # Idempotency guard: avoid duplicating actions if called multiple times
        if getattr(self, "_toolbar_ready", False):
            return

        tb = None
        for attr in ("barVERTICAL_right", "right_toolbar", "toolBarRight"):
            tb = getattr(self.window, attr, None)
            if isinstance(tb, QtWidgets.QToolBar):
                break
        if tb is None:
            return  # host has no right toolbar; silently skip

        def _find_action(name: str) -> QtGui.QAction | None:
            try:
                for a in tb.actions():
                    if a.objectName() == name:
                        return a
            except Exception:
                pass
            return None

        def add_toggle(text: str, icon_name: str, slot, obj_name: str) -> QtGui.QAction:
            # Reuse existing action if already present on the toolbar
            existing = _find_action(obj_name)
            if existing is not None:
                try:
                    existing.toggled.connect(slot, QtCore.Qt.ConnectionType.UniqueConnection)
                except Exception:
                    pass
                return existing
            act = QtGui.QAction(self._qicon(icon_name), text, self.window)
            act.setObjectName(obj_name)
            act.setCheckable(True)
            act.toggled.connect(slot)
            tb.addAction(act)
            return act

        self.actSlices   = add_toggle("Slices",         "32x32_cloud_slice_3D.png",          lambda v: self.panel.set_visibility("slices", v),    "cloud2fem.actSlices")
        self.actCurrent  = add_toggle("Current slice",  "32x32_cloud_current_slice_3D.png",  lambda v: self.panel.set_visibility("current", v),   "cloud2fem.actCurrent")
        self.actCentroids= add_toggle("Centroids",      "32x32_cloud_centroids_3D.png",      lambda v: self.panel.set_visibility("centroids", v), "cloud2fem.actCentroids")
        self.actPolylines= add_toggle("Polylines",      "32x32_cloud_polylines_3D.png",      lambda v: self.panel.set_visibility("polylines", v), "cloud2fem.actPolylines")
        self.actPolygons = add_toggle("Polygons",       "32x32_cloud_polygon_3D.png",        lambda v: self.panel.set_visibility("polygons", v),  "cloud2fem.actPolygons")
        self.actMesh     = add_toggle("C2F mesh",       "32x32_cloud_mesh_3D.png",           lambda v: self.panel.set_visibility("mesh", v),      "cloud2fem.actMesh")

        # Sensible defaults
        for act in (self.actSlices, self.actCurrent, self.actCentroids, self.actPolylines, self.actPolygons):
            act.setChecked(True)
        self.actMesh.setChecked(False)

        # Mark toolbar as initialized to prevent duplicates on future calls
        self._toolbar_ready = True

    def _qicon(self, name: str) -> QtGui.QIcon:
        """Best-effort icon loader using host helper if available, else blank.

        Args:
            name: Icon filename expected in the host icon theme.
        Returns:
            QIcon instance (may be empty if not found).
        """
        try:
            from c2f4dt.utils.icons import qicon  # host helper if present
            ic = qicon(name)
            if isinstance(ic, QtGui.QIcon):
                return ic
        except Exception:
            pass
        return QtGui.QIcon()

    def _focus_slicing_tab(self) -> bool:
        """Bring the SLICING tab to front, if present.

        Returns:
            bool: True if a tab switch was performed, False otherwise.
        """
        try:
            # First try: climb from scrollSLICING to its parent QTabWidget
            scroll = getattr(self.window, "scrollSLICING", None)
            if isinstance(scroll, QtWidgets.QWidget):
                w = scroll
                while w is not None:
                    if isinstance(w, QtWidgets.QTabWidget):
                        w.setCurrentWidget(scroll.parentWidget() or scroll)
                        return True
                    w = w.parentWidget()
        except Exception:
            pass

        try:
            # Fallback: search any QTabWidget whose tab text matches 'slicing'
            tabs = self.window.findChildren(QtWidgets.QTabWidget)
            for tw in tabs:
                for i in range(tw.count()):
                    try:
                        if str(tw.tabText(i)).strip().lower() == "slicing":
                            tw.setCurrentIndex(i)
                            return True
                    except Exception:
                        pass
        except Exception:
            pass
        return False
    
    def _log(self, level: str, text: str) -> None:
        """Log into the host message panel and stdout.

        Args:
            level: Log level string, e.g., "INFO".
            text: Message text.
        """
        try:
            if hasattr(self.window, "txtMessages"):
                self.window.txtMessages.appendPlainText(text)
        except Exception:
            pass
        print(f"[{level}] {text}")

    # -------------------------- actions for PluginManager --------------
    def run(self, **ctx) -> bool:  # preferred
        ok = self._focus_slicing_tab()
        if not ok:
            self._log("INFO", "[cloud2fem] SLICING tab not found")
        return ok

    def exec(self, **ctx) -> bool:  # alias
        return self.run(**ctx)

    def execute(self, **ctx) -> bool:  # alias
        return self.run(**ctx)

    def show(self, **ctx) -> bool:  # alias
        return self.run(**ctx)

    def __call__(self, *args, **kwargs) -> bool:  # alias
        return self.run(**kwargs)
    
# ---------------------------------------------------------------------
# Slicing panel widget (all UI + logic stays inside the plugin)
# ---------------------------------------------------------------------
class SlicingPanel(QtWidgets.QWidget):
    """Cloud2FEM Slicing Panel widget.

    The panel defines the slicing parameters and exposes actions to compute slices,
    synchronize the grid planes, and build a FEM grid by extrusion.
    """
    def __init__(self, window, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.window = window
        # Load default options from module (kept local to plugin to avoid global state)
        try:
            from .slicing_options import default_slice_options
            self._opts = default_slice_options()
        except Exception:
            self._opts = {}
        self._slices: list[dict] = []
        self._vis = {"slices": True, "current": True, "centroids": True, "polylines": True, "polygons": True, "mesh": False}
        self._build_ui()
        self._init_defaults()

    # ------------------------------ UI ---------------------------------
    def _build_ui(self) -> None:
        """Build the full vertical layout for slicing controls."""
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        # Direction group -------------------------------------------------
        grp_dir = QtWidgets.QGroupBox("Direction")
        gdl = QtWidgets.QGridLayout(grp_dir)
        self.cboAxis = QtWidgets.QComboBox(); self.cboAxis.addItems(["X", "Y", "Z", "Custom"]) 
        self.cboAxis.currentIndexChanged.connect(self._on_dir_changed)
        gdl.addWidget(QtWidgets.QLabel("Axis:"), 0, 0); gdl.addWidget(self.cboAxis, 0, 1)
        # Custom P1/P2
        self.edP1X = QtWidgets.QLineEdit(); self.edP1X.setPlaceholderText("x1")
        self.edP1Y = QtWidgets.QLineEdit(); self.edP1Y.setPlaceholderText("y1")
        self.edP1Z = QtWidgets.QLineEdit(); self.edP1Z.setPlaceholderText("z1")
        self.edP2X = QtWidgets.QLineEdit(); self.edP2X.setPlaceholderText("x2")
        self.edP2Y = QtWidgets.QLineEdit(); self.edP2Y.setPlaceholderText("y2")
        self.edP2Z = QtWidgets.QLineEdit(); self.edP2Z.setPlaceholderText("z2")
        for ed in (self.edP1X, self.edP1Y, self.edP1Z, self.edP2X, self.edP2Y, self.edP2Z):
            ed.setMaximumWidth(88)
        rowP1 = QtWidgets.QHBoxLayout(); [rowP1.addWidget(w) for w in (self.edP1X, self.edP1Y, self.edP1Z)]
        rowP2 = QtWidgets.QHBoxLayout(); [rowP2.addWidget(w) for w in (self.edP2X, self.edP2Y, self.edP2Z)]
        self.btnPickP1 = QtWidgets.QPushButton("Pick P1"); self.btnPickP2 = QtWidgets.QPushButton("Pick P2")
        self.btnPickP1.setEnabled(False); self.btnPickP2.setEnabled(False)  # wire when host exposes picker
        pickRow = QtWidgets.QHBoxLayout(); pickRow.addWidget(self.btnPickP1); pickRow.addWidget(self.btnPickP2); pickRow.addStretch(1)
        gdl.addWidget(QtWidgets.QLabel("Custom P1:"), 1, 0); wP1 = QtWidgets.QWidget(); wP1.setLayout(rowP1); gdl.addWidget(wP1, 1, 1)
        gdl.addWidget(QtWidgets.QLabel("Custom P2:"), 2, 0); wP2 = QtWidgets.QWidget(); wP2.setLayout(rowP2); gdl.addWidget(wP2, 2, 1)
        gdl.addLayout(pickRow, 3, 1)
        lay.addWidget(grp_dir)

        # Thickness & Mode ------------------------------------------------
        grp_mode = QtWidgets.QGroupBox("Thickness & Mode")
        fl = QtWidgets.QFormLayout(grp_mode)
        self.spnThickness = QtWidgets.QDoubleSpinBox(); self.spnThickness.setRange(0.0, 1e6); self.spnThickness.setDecimals(5); self.spnThickness.setSingleStep(0.01); self.spnThickness.setValue(0.0)
        fl.addRow("Thickness:", self.spnThickness)
        self.rdoDelta = QtWidgets.QRadioButton("Fixed Δ (step)"); self.rdoCount = QtWidgets.QRadioButton("Fixed number of slices"); self.rdoCustom = QtWidgets.QRadioButton("Custom rule"); self.rdoDelta.setChecked(True)
        rowModes = QtWidgets.QHBoxLayout(); [rowModes.addWidget(w) for w in (self.rdoDelta, self.rdoCount, self.rdoCustom)]
        fl.addRow("Mode:", self._wrap(rowModes))
        self.spnDelta = QtWidgets.QDoubleSpinBox(); self.spnDelta.setRange(0.0, 1e6); self.spnDelta.setDecimals(5); self.spnDelta.setSingleStep(0.01); self.spnDelta.setValue(0.10)
        self.spnCount = QtWidgets.QSpinBox(); self.spnCount.setRange(1, 100000); self.spnCount.setValue(20)
        self.edCustomRule = QtWidgets.QLineEdit(); self.edCustomRule.setPlaceholderText("e.g. z<10 then Δ=0.05 else Δ=0.10")
        fl.addRow("Δ (fixed step):", self.spnDelta)
        fl.addRow("N slices:", self.spnCount)
        fl.addRow("Custom rule:", self.edCustomRule)
        lay.addWidget(grp_mode)

        # Range -----------------------------------------------------------
        grp_rng = QtWidgets.QGroupBox("Range")
        fr = QtWidgets.QFormLayout(grp_rng)
        self.spnStart = QtWidgets.QDoubleSpinBox(); self.spnStart.setRange(-1e9, 1e9); self.spnStart.setDecimals(6)
        self.spnEnd = QtWidgets.QDoubleSpinBox(); self.spnEnd.setRange(-1e9, 1e9); self.spnEnd.setDecimals(6)
        fr.addRow("Start coord:", self.spnStart)
        fr.addRow("End coord:", self.spnEnd)
        lay.addWidget(grp_rng)

        # Products --------------------------------------------------------
        grp_prod = QtWidgets.QGroupBox("Products")
        hl = QtWidgets.QHBoxLayout(grp_prod)
        self.chkCentroids = QtWidgets.QCheckBox("Centroids")
        self.chkRawPolylines = QtWidgets.QCheckBox("Raw polylines")
        self.chkRefinedPolylines = QtWidgets.QCheckBox("Refined polylines")
        self.chkPolygons = QtWidgets.QCheckBox("Polygons")
        for c in (self.chkCentroids, self.chkRawPolylines, self.chkRefinedPolylines, self.chkPolygons):
            c.setChecked(True); hl.addWidget(c)
        lay.addWidget(grp_prod)

        # Actions ---------------------------------------------------------
        rowAct = QtWidgets.QHBoxLayout()
        self.btnCompute = QtWidgets.QPushButton("Compute Slices")
        self.btnGrid = QtWidgets.QPushButton("Generate Grid from Slices")
        self.btnFEM = QtWidgets.QPushButton("Build FEM from Grid")
        [rowAct.addWidget(b) for b in (self.btnCompute, self.btnGrid, self.btnFEM)]
        lay.addWidget(self._wrap(rowAct))
        lay.addStretch(1)

        # Wire
        self.btnCompute.clicked.connect(self._on_compute_slices)
        self.btnGrid.clicked.connect(self._on_grid_from_slices)
        self.btnFEM.clicked.connect(self._on_fem_from_grid)

    def _init_defaults(self) -> None:
        """Initialize default range from active dataset bounds (best effort)."""
        try:
            # 1) Apply in-repo defaults from slicing_options.py
            try:
                slices = dict(self._opts.get("slices", {})) if isinstance(self._opts, dict) else {}
                if slices:
                    # Axis
                    ax = str(slices.get("axis", "Z")).upper()
                    idx = {"X": 0, "Y": 1, "Z": 2, "CUSTOM": 3}.get(ax, 2)
                    self.cboAxis.setCurrentIndex(idx)
                    # Thickness
                    if "thickness" in slices:
                        self.spnThickness.setValue(float(slices["thickness"]))
                    # Mode mapping
                    mode = str(slices.get("spacing_mode", "fixed_count")).lower()
                    self.rdoDelta.setChecked(mode == "fixed_step")
                    self.rdoCount.setChecked(mode == "fixed_count")
                    self.rdoCustom.setChecked(mode == "custom")
                    # Mode params
                    if "fixed_step" in slices:
                        self.spnDelta.setValue(float(slices["fixed_step"]))
                    if "fixed_count" in slices:
                        self.spnCount.setValue(int(slices["fixed_count"]))
            except Exception:
                pass
            # Optional: apply defaults from slice_options.py if available
            try:
                import importlib.util, os
                for cand in ("slice_options.py", os.path.join(os.getcwd(), "slice_options.py")):
                    if os.path.exists(cand):
                        spec = importlib.util.spec_from_file_location("slice_options", cand)
                        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
                        defaults = getattr(mod, "DEFAULTS", getattr(mod, "defaults", {}))
                        # Apply known keys if present
                        if isinstance(defaults, dict):
                            if "axis" in defaults:
                                ax = str(defaults["axis"]).upper();
                                i = max(0, ["X","Y","Z","CUSTOM"].index(ax) if ax in ("X","Y","Z","CUSTOM") else 0)
                                self.cboAxis.setCurrentIndex(i)
                            if "thickness" in defaults: self.spnThickness.setValue(float(defaults["thickness"]))
                            if "mode" in defaults:
                                m = str(defaults["mode"]).lower()
                                self.rdoDelta.setChecked(m == "delta"); self.rdoCount.setChecked(m == "count"); self.rdoCustom.setChecked(m == "custom")
                            if "delta" in defaults: self.spnDelta.setValue(float(defaults["delta"]))
                            if "count" in defaults: self.spnCount.setValue(int(defaults["count"]))
                            if "custom_rule" in defaults: self.edCustomRule.setText(str(defaults["custom_rule"]))
                            if "start" in defaults: self.spnStart.setValue(float(defaults["start"]))
                            if "end" in defaults: self.spnEnd.setValue(float(defaults["end"]))
                        break
            except Exception:
                pass
            ds = self._current_dataset_index()
            b = self._active_dataset_bounds(ds)
            if b is not None:
                self.spnStart.setValue(min(b)); self.spnEnd.setValue(max(b))
        except Exception:
            pass
        self._on_dir_changed(self.cboAxis.currentIndex())

    @staticmethod
    def _wrap(layout: QtWidgets.QLayout) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget(); w.setLayout(layout); return w

    def _on_dir_changed(self, idx: int) -> None:
        """Enable/disable custom direction controls based on selection."""
        is_custom = (self.cboAxis.currentText().upper() == "CUSTOM")
        for w in (self.edP1X, self.edP1Y, self.edP1Z, self.edP2X, self.edP2Y, self.edP2Z, self.btnPickP1, self.btnPickP2):
            w.setEnabled(is_custom)

    # --------------------------- Actions --------------------------------
    def _on_compute_slices(self) -> None:
        """Compute slices and derived products, honoring CANCEL.

        This method reports progress via the host window and polls a local cancel flag.
        Heavy-lifting ops are left as TODO markers to be replaced with Cloud2FEMi calls.
        """
        p = self._params()
        self._progress_begin("Computing slices…")
        win = self.window
        canceled = {"flag": False}

        def _cancel():
            canceled["flag"] = True

        # Temporarily hook CANCEL if present
        try:
            btn = getattr(win, "btnCancel", None)
            if isinstance(btn, QtWidgets.QPushButton):
                btn.setEnabled(True)
                btn.clicked.connect(_cancel, QtCore.Qt.ConnectionType.UniqueConnection)
        except Exception:
            pass

        try:
            # Resolve axis and range
            axis = p["axis"]; start = p["start"]; end = p["end"]
            ds = self._current_dataset_index()
            bounds_full = self._active_dataset_bounds(ds)
            if bounds_full is None:
                self._progress_update(0, "No active dataset bounds"); return
            if start is None or end is None:
                lo, hi = min(bounds_full), max(bounds_full)
                start = lo if start is None else start
                end = hi if end is None else end
            if end < start:
                start, end = end, start
            L = max(0.0, end - start)

            # Steps computation
            steps = []
            if p["mode"] == "delta":
                d = max(1e-9, p["delta"]); n = int(max(1, round(L / d)))
                steps = [start + i * d for i in range(n + 1) if start + i * d <= end + 1e-12]
            elif p["mode"] == "count":
                n = int(max(1, p["count"]))
                if n == 1: steps = [0.5 * (start + end)]
                else:
                    h = L / float(n - 1); steps = [start + i * h for i in range(n)]
            else:
                rule = (p["custom_rule"] or "").strip(); d = None
                for tok in ("Δ=", "delta=", "step="):
                    if tok in rule:
                        try: d = float(rule.split(tok, 1)[1].strip())
                        except Exception: d = None
                        break
                if d is None: d = max(1e-9, (L / 20.0))
                n = int(max(1, round(L / d)))
                steps = [start + i * d for i in range(n + 1) if start + i * d <= end + 1e-12]

            # Thickness default suggestion: 3× average spacing
            thick = float(p["thickness"])
            if thick <= 0.0:
                spacing = self._estimate_point_spacing(ds)
                thick = 3.0 * spacing if spacing > 0 else max(1e-3, L / 200.0)
                self.spnThickness.setValue(thick)

            # Produce slices (placeholders)
            out = []
            total = max(1, len(steps))
            for i, coord in enumerate(steps):
                if canceled["flag"]:
                    self._progress_update(int(100 * i / total), "Canceled"); break
                rec = {
                    "coord": float(coord),
                    "axis": axis,
                    "thickness": float(thick),
                    "centroids": [] if p["centroids"] else None,
                    "polyline_raw": [] if p["raw_polylines"] else None,
                    "polyline_refined": [] if p["refined_polylines"] else None,
                    "polygons": [] if p["polygons"] else None,
                }
                out.append(rec)
                self._progress_update(int(5 + 90 * (i + 1) / total), f"Slice {i+1}/{total}")

            self._slices = out
            self._progress_end()
            self._log("INFO", f"Slices ready: {len(out)}")
            self._ensure_tree()
        finally:
            # Unhook CANCEL
            try:
                btn = getattr(win, "btnCancel", None)
                if isinstance(btn, QtWidgets.QPushButton):
                    btn.setEnabled(False)
                    btn.clicked.disconnect(_cancel)
            except Exception:
                pass

    def _on_grid_from_slices(self) -> None:
        """Generate grid planes from current slices (placeholder)."""
        try:
            if not self._slices:
                self._log("INFO", "No slices to sync grid from"); return
            coords = [s.get("coord") for s in self._slices if isinstance(s, dict)]
            if not coords:
                self._log("INFO", "No valid slice coordinates"); return
            # TODO: call actual grid sync using host/grid APIs
            self._log("INFO", f"Grid planes synced from slices: {len(coords)}")
        except Exception as ex:
            self._log("WARN", f"Grid sync failed: {ex}")

    def _on_fem_from_grid(self) -> None:
        """Build FEM from grid by extruding quads across slices (placeholder)."""
        try:
            self._log("INFO", f"FEM-from-grid requested (slices={len(self._slices)})")
            # TODO: implement extrusion to Hex8 voxels
        except Exception as ex:
            self._log("WARN", f"FEM build failed: {ex}")

    # --------------------------- Helpers -------------------------------
    def _params(self) -> dict:
        axis = self.cboAxis.currentText().upper()
        def _f(ed: QtWidgets.QLineEdit):
            try: return float(ed.text())
            except Exception: return None
        p1 = (_f(self.edP1X), _f(self.edP1Y), _f(self.edP1Z))
        p2 = (_f(self.edP2X), _f(self.edP2Y), _f(self.edP2Z))
        if self.rdoDelta.isChecked(): mode = "delta"
        elif self.rdoCount.isChecked(): mode = "count"
        else: mode = "custom"
        return {
            "axis": axis, "p1": p1, "p2": p2,
            "thickness": float(self.spnThickness.value()),
            "mode": mode, "delta": float(self.spnDelta.value()), "count": int(self.spnCount.value()), "custom_rule": self.edCustomRule.text(),
            "start": float(self.spnStart.value()), "end": float(self.spnEnd.value()),
            "centroids": self.chkCentroids.isChecked(), "raw_polylines": self.chkRawPolylines.isChecked(),
            "refined_polylines": self.chkRefinedPolylines.isChecked(), "polygons": self.chkPolygons.isChecked(),
        }

    def set_visibility(self, kind: str, visible: bool) -> None:
        """Toggle visibility of a product overlay in the viewer.

        Args:
            kind: One of {"slices", "current", "centroids", "polylines", "polygons", "mesh"}.
            visible: True to show, False to hide.
        """
        self._vis[kind] = bool(visible)
        # TODO: wire to actual overlay updates; for now, just log
        self._log("INFO", f"Visibility → {kind}={'ON' if visible else 'OFF'}")
        
    def _ensure_tree(self) -> None:
        """Ensure the dataset node contains the Cloud2FEM sections.

        Structure:
            <dataset>
              - Point cloud
                  - Normals
              - Slices
                  - Slices
                  - Centroids
                  - Polylines
                  - Polygons
                  - Materials
              - FEM
                  - C2F mesh
        """
        try:
            tree = getattr(self.window, "treeDatasets", None) or getattr(self.window, "tree", None)
            if tree is None or not hasattr(tree, "invisibleRootItem"):
                return
            # Heuristic: find current dataset item by selection or by index name
            cur_name = None
            try:
                idx = self._current_dataset_index()
                v = getattr(self.window, "viewer3d", None)
                recs = getattr(v, "_datasets", []) if v is not None else []
                if isinstance(idx, int) and 0 <= idx < len(recs):
                    cur_name = recs[idx].get("name") or recs[idx].get("filename")
            except Exception:
                pass

            root = tree.invisibleRootItem()
            # Try to find node by text; else use selected item
            ds_item = None
            if cur_name:
                for i in range(root.childCount()):
                    it = root.child(i)
                    if cur_name and it.text(0) == cur_name:
                        ds_item = it
                        break
            if ds_item is None:
                ds_item = tree.currentItem() or root

            def ensure_child(parent, label: str):
                # Find by exact text in column 0; create if missing
                for i in range(parent.childCount()):
                    it = parent.child(i)
                    if it.text(0) == label:
                        return it
                it = QtWidgets.QTreeWidgetItem([label])
                parent.addChild(it)
                return it

            # Build structure
            pc = ensure_child(ds_item, "Point cloud")
            ensure_child(pc, "Normals")

            sl = ensure_child(ds_item, "Slices")
            ensure_child(sl, "Slices")
            ensure_child(sl, "Centroids")
            ensure_child(sl, "Polylines")
            ensure_child(sl, "Polygons")
            ensure_child(sl, "Materials")

            fem = ensure_child(ds_item, "FEM")
            ensure_child(fem, "C2F mesh")

            tree.expandItem(ds_item)
            tree.expandItem(sl)
            tree.expandItem(fem)
        except Exception:
            pass

    def _estimate_point_spacing(self, ds: Optional[int]) -> float:
        try:
            v = getattr(self.window, "viewer3d", None)
            recs = getattr(v, "_datasets", []) if v is not None else []
            if not (isinstance(ds, int) and 0 <= ds < len(recs)): return 0.0
            pdata = recs[ds].get("pdata")
            if pdata is None: return 0.0
            n = int(getattr(pdata, "n_points", 0) or 0)
            b = pdata.bounds if hasattr(pdata, "bounds") else None
            if n <= 0 or b is None: return 0.0
            dx = float(b[1]-b[0]); dy = float(b[3]-b[2]); dz = float(b[5]-b[4])
            vol = max(0.0, dx*dy*dz)
            if vol <= 0 or n <= 0: return 0.0
            return (vol ** (1.0/3.0)) / (max(1, n) ** (1.0/3.0))
        except Exception:
            return 0.0

    def _active_dataset_bounds(self, ds: Optional[int]):
        v = getattr(self.window, "viewer3d", None)
        if v is None: return None
        try:
            recs = getattr(v, "_datasets", [])
            if not (isinstance(ds, int) and 0 <= ds < len(recs)): return None
            pdata = recs[ds].get("pdata")
            if pdata is None or not hasattr(pdata, "bounds"): return None
            b = pdata.bounds
            axis = self.cboAxis.currentText().upper()
            if axis == "X": return [float(b[0]), float(b[1])]
            if axis == "Y": return [float(b[2]), float(b[3])]
            if axis == "Z": return [float(b[4]), float(b[5])]
            return [float(min(b[0], b[2], b[4])), float(max(b[1], b[3], b[5]))]
        except Exception:
            return None

    def _current_dataset_index(self) -> Optional[int]:
        try:
            return int(self.window._current_dataset_index())
        except Exception:
            return None

    # ---- host logging & progress helpers ------------------------------
    def _log(self, level: str, text: str) -> None:
        try:
            if hasattr(self.window, "txtMessages"):
                self.window.txtMessages.appendPlainText(text)
        except Exception:
            pass
        print(f"[{level}] {text}")

    def _progress_begin(self, title: str) -> None:
        try:
            if hasattr(self.window, "_progress_start"):
                self.window._progress_start(title)
            elif hasattr(self.window, "_import_progress_begin"):
                self.window._import_progress_begin(title)
        except Exception:
            pass

    def _progress_update(self, percent: Optional[int], message: Optional[str]) -> None:
        try:
            if hasattr(self.window, "_import_progress_update"):
                self.window._import_progress_update(percent=percent, message=message)
        except Exception:
            pass

    def _progress_end(self) -> None:
        try:
            if hasattr(self.window, "_progress_finish"):
                self.window._progress_finish()
            elif hasattr(self.window, "_import_progress_end"):
                self.window._import_progress_end()
        except Exception:
            pass

