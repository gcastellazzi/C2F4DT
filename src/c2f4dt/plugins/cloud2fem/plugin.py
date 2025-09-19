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

            # If already present, do not add again (look for our inner widget)
            try:
                for w in container.findChildren(QtWidgets.QWidget):
                    if w.objectName() == "cloud2fem.slicing_widget":
                        return
            except Exception:
                pass

            # Boxless fallback: append the widget directly to the container's main layout
            widget.setObjectName("cloud2fem.slicing_widget")
            widget.setMaximumWidth(320)
            widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
            container.layout().addWidget(widget)
            # keep a little stretch to push content to top (but avoid duplicates)
            # ensure only one final stretch exists
            stretches = [container.layout().itemAt(i) for i in range(container.layout().count())]
            if not stretches or stretches[-1] is None or stretches[-1].spacerItem() is None:
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
            tb.addSeparator() if tb is not None else None
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
        self.actPlanes   = add_toggle("Section planes", "32x32_3D_section_planes.png",       lambda v: self.panel.set_visibility("planes", v),    "cloud2fem.actPlanes" )

        prop = _find_action("cloud2fem.actProps")
        if prop is None:
            prop = QtGui.QAction(self._qicon("32x32_applications-system.png"),
                                "Cloud2FEM display properties", self.window)
            prop.setObjectName("cloud2fem.actProps")
            prop.triggered.connect(self.panel._on_open_display_props)
            tb.addAction(prop)
        # Sensible defaults
        for act in (self.actPlanes, self.actSlices, self.actCurrent, self.actCentroids, self.actPolylines, self.actPolygons):
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
        # Runtime overlays storage (actors/ids handled by the viewer/plotter)
        self._actors = {
            "slices": [],
            "current": [],
            "centroids": [],
            "polylines": [],
            "polygons": [],
            "mesh": [],
        }
        # Visibility channels (planes, per-slice products)
        self._vis = {
            "planes": True,      # NEW: section planes only
            "slices": True,      # slice points
            "current": True,
            "centroids": True,
            "polylines": True,
            "polygons": True,
            "mesh": False,
        }
        # Basic style defaults (editable via properties dialog)
        self._style = {
            "point_size_factor": 2.0,      # multiplier over viewer's default
            "centroid_size": 12.0,
            "planes_opacity": 0.18,
            "current_plane_opacity": 0.35,
            "line_width": 2.0,             # polylines width
            "poly_opacity": 0.25,          # polygons fill opacity
            "poly_edge_width": 1.5,        # polygons outline width
            # axis colors (R,G,B)
            "color_x": (1.0, 0.25, 0.25),
            "color_y": (0.25, 1.0, 0.25),
            "color_z": (0.25, 0.5, 1.0),
        }
        # Track if user explicitly set thickness (avoid auto-reset on compute)
        self._setting_thickness = False   # internal guard for programmatic sets
        self._thickness_user_set = False  # becomes True when user edits the spinbox
        self._build_ui()
        self._init_defaults()

    # ------------------------------ UI ---------------------------------
    def _build_ui(self) -> None:
        """Build the full vertical layout for slicing controls."""
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Quick view filter box (axis visibility for products) -------------
        grp_view = QtWidgets.QGroupBox("View Slices")
        grp_view.setMaximumWidth(276)
        hv = QtWidgets.QHBoxLayout(grp_view)
        self.chkViewX = QtWidgets.QCheckBox("X"); self.chkViewX.setChecked(True)
        self.chkViewY = QtWidgets.QCheckBox("Y"); self.chkViewY.setChecked(True)
        self.chkViewZ = QtWidgets.QCheckBox("Z"); self.chkViewZ.setChecked(True)
        self.chkViewAll = QtWidgets.QCheckBox("All"); self.chkViewAll.setChecked(True)
        for w in (self.chkViewX, self.chkViewY, self.chkViewZ, self.chkViewAll):
            hv.addWidget(w)
        lay.addWidget(grp_view)
        # Wire filters
        self.chkViewAll.toggled.connect(self._on_view_all)
        self.chkViewX.toggled.connect(lambda _v: self._update_overlays())
        self.chkViewY.toggled.connect(lambda _v: self._update_overlays())
        self.chkViewZ.toggled.connect(lambda _v: self._update_overlays())
        # Direction group -------------------------------------------------
        grp_dir = QtWidgets.QGroupBox("Slicing Direction")
        grp_dir.setMaximumWidth(276)
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
        grp_mode = QtWidgets.QGroupBox("Slicing Mode")
        grp_mode.setMaximumWidth(276)
        fl = QtWidgets.QFormLayout(grp_mode)
        self.spnThickness = QtWidgets.QDoubleSpinBox(); self.spnThickness.setRange(0.0, 1e6); self.spnThickness.setDecimals(5); self.spnThickness.setSingleStep(0.01); self.spnThickness.setValue(0.0)
        fl.addRow("Thickness:", self.spnThickness)
        self.spnThickness.valueChanged.connect(self._on_thickness_changed)
        # Mode: use a combo for clarity and simpler state management
        self.cboMode = QtWidgets.QComboBox()
        self.cboMode.addItems(["Fixed number", "Fixed step", "Custom rule"])  # 0,1,2
        self.cboMode.currentIndexChanged.connect(self._on_mode_changed)
        fl.addRow("Mode:", self.cboMode)

        # Keep params (already defined below), just ensure they are present
        # self.spnDelta, self.spnCount, self.edCustomRule are already created as in your file
        # Initialize enable state once UI is built (do a first call at end of _build_ui)
        self.spnDelta = QtWidgets.QDoubleSpinBox(); self.spnDelta.setRange(0.0, 1e6); self.spnDelta.setDecimals(5); self.spnDelta.setSingleStep(0.01); self.spnDelta.setValue(0.10)
        self.spnCount = QtWidgets.QSpinBox(); self.spnCount.setRange(1, 100000); self.spnCount.setValue(20)
        self.edCustomRule = QtWidgets.QLineEdit(); self.edCustomRule.setPlaceholderText("e.g. z<10 then Δ=0.05 else Δ=0.10")
        fl.addRow("Δ (fixed step):", self.spnDelta)
        fl.addRow("N slices:", self.spnCount)
        fl.addRow("Custom rule:", self.edCustomRule)
        lay.addWidget(grp_mode)

        # Range -----------------------------------------------------------
        grp_rng = QtWidgets.QGroupBox("Slicing Range")
        grp_rng.setMaximumWidth(276)
        fr = QtWidgets.QFormLayout(grp_rng)
        self.spnStart = QtWidgets.QDoubleSpinBox(); self.spnStart.setRange(-1e9, 1e9); self.spnStart.setDecimals(6)
        self.spnEnd = QtWidgets.QDoubleSpinBox(); self.spnEnd.setRange(-1e9, 1e9); self.spnEnd.setDecimals(6)
        fr.addRow("Start coord:", self.spnStart)
        fr.addRow("End coord:", self.spnEnd)
        lay.addWidget(grp_rng)

        # Products --------------------------------------------------------
        grp_prod = QtWidgets.QGroupBox("Slicing Products")
        grp_prod.setMaximumWidth(276)
        hl = QtWidgets.QVBoxLayout(grp_prod)
        self.chkCentroids = QtWidgets.QCheckBox("Centroids")
        self.chkRawPolylines = QtWidgets.QCheckBox("Raw polylines")
        self.chkRefinedPolylines = QtWidgets.QCheckBox("Refined polylines")
        self.chkPolygons = QtWidgets.QCheckBox("Polygons")
        for c in (self.chkCentroids, self.chkRawPolylines, self.chkRefinedPolylines, self.chkPolygons):
            c.setChecked(True); hl.addWidget(c)
        lay.addWidget(grp_prod)

        # Actions ---------------------------------------------------------
        rowAct = QtWidgets.QVBoxLayout()
        self.btnCompute = QtWidgets.QPushButton("Generate Slices")
        self.btnGrid = QtWidgets.QPushButton("Generate Grid")
        self.btnFEM = QtWidgets.QPushButton("Build FEM")
        [rowAct.addWidget(b) for b in (self.btnCompute, self.btnGrid, self.btnFEM)]
        lay.addWidget(self._wrap(rowAct))
        lay.addStretch(1)
        
        # Initialize mode-dependent widgets
        self._on_mode_changed(self.cboMode.currentIndex())

        # Wire
        self.btnCompute.clicked.connect(self._on_compute_slices)
        self.btnGrid.clicked.connect(self._on_grid_from_slices)
        self.btnFEM.clicked.connect(self._on_fem_from_grid)


    def _on_view_all(self, v: bool) -> None:
        try:
            self.chkViewX.blockSignals(True); self.chkViewY.blockSignals(True); self.chkViewZ.blockSignals(True)
            self.chkViewX.setChecked(v); self.chkViewY.setChecked(v); self.chkViewZ.setChecked(v)
        finally:
            self.chkViewX.blockSignals(False); self.chkViewY.blockSignals(False); self.chkViewZ.blockSignals(False)
        self._update_overlays()

    def _axis_color(self, ax: str) -> tuple[float, float, float]:
        ax = (ax or "Z").upper()
        if ax == "X": return self._style.get("color_x", (1.0, 0.25, 0.25))
        if ax == "Y": return self._style.get("color_y", (0.25, 1.0, 0.25))
        return self._style.get("color_z", (0.25, 0.5, 1.0))

    def _axis_visible(self, ax: str) -> bool:
        ax = (ax or "Z").upper()
        if getattr(self, "chkViewAll", None) and self.chkViewAll.isChecked():
            return True
        if ax == "X": return getattr(self, "chkViewX", None) and self.chkViewX.isChecked()
        if ax == "Y": return getattr(self, "chkViewY", None) and self.chkViewY.isChecked()
        return getattr(self, "chkViewZ", None) and self.chkViewZ.isChecked()

    def _on_mode_changed(self, idx: int) -> None:
        """Enable/disable fields according to mode combo.

        0 = Fixed number, 1 = Fixed step, 2 = Custom rule
        """
        is_count = (idx == 0)
        is_step = (idx == 1)
        is_custom = (idx == 2)
        self.spnCount.setEnabled(is_count)
        self.spnDelta.setEnabled(is_step)
        self.edCustomRule.setEnabled(is_custom)
        
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
                        self._setting_thickness = True
                        try:
                            self.spnThickness.setValue(float(slices["thickness"]))
                        finally:
                            self._setting_thickness = False
                        # programmatic suggestion: do not mark as user-set
                        self._thickness_user_set = False
                    # Mode mapping -> combo index
                    mode = str(slices.get("spacing_mode", "fixed_count")).lower()
                    idxm = {"fixed_count": 0, "fixed_step": 1, "custom": 2}.get(mode, 0)
                    self.cboMode.setCurrentIndex(idxm)
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
                            if "thickness" in defaults:
                                self._setting_thickness = True
                                try:
                                    self.spnThickness.setValue(float(defaults["thickness"]))
                                finally:
                                    self._setting_thickness = False
                                self._thickness_user_set = False
                            if "mode" in defaults:
                                m = str(defaults["mode"]).lower()
                                self.cboMode.setCurrentIndex({"count":0, "delta":1, "fixed":1, "custom":2}.get(m, 0))
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
        try:
            L = abs(float(self.spnEnd.value()) - float(self.spnStart.value()))
            n = max(1, int(self.spnCount.value()))
            self.spnDelta.setValue(L if n <= 1 else (L / float(n - 1)))
        except Exception:
            pass

    @staticmethod
    def _wrap(layout: QtWidgets.QLayout) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget(); w.setLayout(layout); return w

    def _on_dir_changed(self, idx: int) -> None:
        """Enable/disable custom direction controls based on selection."""
        is_custom = (self.cboAxis.currentText().upper() == "CUSTOM")
        for w in (self.edP1X, self.edP1Y, self.edP1Z, self.edP2X, self.edP2Y, self.edP2Z, self.btnPickP1, self.btnPickP2):
            w.setEnabled(is_custom)

        # On direction change, allow auto-suggestion again unless user edits afterwards
        self._thickness_user_set = False

        # Auto-update range from active dataset bounds for the chosen axis
        ds = self._current_dataset_index()
        b = self._active_dataset_bounds(ds)
        if isinstance(b, (list, tuple)) and len(b) == 2:
            try:
                self.spnStart.blockSignals(True); self.spnEnd.blockSignals(True)
                self.spnStart.setValue(float(min(b))); self.spnEnd.setValue(float(max(b)))
            finally:
                self.spnStart.blockSignals(False); self.spnEnd.blockSignals(False)
            # Recompute fixed step from range / current count
            try:
                L = abs(float(self.spnEnd.value()) - float(self.spnStart.value()))
                n = max(1, int(self.spnCount.value()))
                self.spnDelta.setValue(L if n <= 1 else (L / float(n - 1)))
            except Exception:
                pass

        # Enforce thickness suggestion: at least 3× mean spacing (unless user-set)
        try:
            thr = max(self._suggest_thickness(ds), 0.0)
            if thr > 0.0 and not self._thickness_user_set:
                self._setting_thickness = True
                try:
                    self.spnThickness.setValue(thr)
                finally:
                    self._setting_thickness = False
                self._thickness_user_set = False
        except Exception:
            pass
    
    def _suggest_thickness(self, ds: Optional[int]) -> float:
        """Return a suggested thickness (>= 3× mean point spacing).

        Tries Cloud Inspection metrics if available, falls back to a quick
        spacing estimate from bounds and point count.
        """
        mnn = self._mean_nn_distance(ds)
        if mnn and mnn > 0:
            return 3.0 * float(mnn)
        sp = self._estimate_point_spacing(ds)
        return 3.0 * float(sp) if sp > 0 else 0.0

    def _mean_nn_distance(self, ds: Optional[int]) -> float:
        """Fetch mean nearest-neighbor distance from Cloud Inspection results, if present."""
        try:
            v = getattr(self.window, "viewer3d", None)
            recs = getattr(v, "_datasets", []) if v is not None else []
            if not (isinstance(ds, int) and 0 <= ds < len(recs)):
                return 0.0
            glob = recs[ds].get("inspection_global", {}) or {}
            # Try common keys (supporting your Cloud Inspection plugin)
            for k in ("mean_nn_distance", "nn_mean", "mean_nn", "avg_nn_dist", "mean_neighbor_distance"):
                val = glob.get(k)
                if val is not None:
                    try:
                        return float(val)
                    except Exception:
                        pass
        except Exception:
            return 0.0
        return 0.0

    # --------------------------- Actions --------------------------------
    def _on_compute_slices(self) -> None:
        """Compute slices and derived products, honoring CANCEL.

        This method reports progress via the host window and polls a local cancel flag.
        Heavy-lifting ops are left as TODO markers to be replaced with Cloud2FEMi calls.
        """
        p = self._params()
        self._log("INFO", f"[c2f] compute: axis={p['axis']} start={p['start']:.4g} end={p['end']:.4g} thick={p['thickness']:.4g} mode={p['mode']}")
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
                # Avoid UniqueConnection here because _cancel is a local function (non QObject slot)
                btn.clicked.connect(_cancel)
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
            min_thick = self._suggest_thickness(ds)
            if thick <= 0.0:
                # Last resort: no user value, choose something reasonable
                thick = max(1e-3, min_thick if (min_thick and min_thick > 0.0) else (L / 200.0))
                self._setting_thickness = True
                try:
                    self.spnThickness.setValue(thick)
                finally:
                    self._setting_thickness = False
                # do not mark user-set here
                self._thickness_user_set = False
            elif (min_thick and min_thick > 0.0) and (thick < min_thick) and (not self._thickness_user_set):
                # Only auto-bump to suggested minimum if the user did not explicitly override
                thick = float(min_thick)
                self._setting_thickness = True
                try:
                    self.spnThickness.setValue(thick)
                finally:
                    self._setting_thickness = False
                self._thickness_user_set = False
            # else: respect user's explicit value even if below suggestion; optionally warn
            elif (min_thick and min_thick > 0.0) and (thick < min_thick) and self._thickness_user_set:
                self._log("INFO", f"Using user thickness {thick:.4g} below suggested min {min_thick:.4g}")

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
            # Sync to store as soon as we have the slices
            self._sync_mct_store()
            self._progress_end()
            self._log("INFO", f"Slices ready: {len(out)}")
            self._ensure_tree()
            self._sync_mct_store()
            # Refresh overlays to show new slices immediately if toggles are ON
            try:
                for k in ("planes", "slices", "centroids", "polylines", "polygons", "current"):
                    if self._vis.get(k, False):
                        self._update_overlays(k)
            except Exception:
                pass
        finally:
            # Unhook CANCEL
            try:
                btn = getattr(win, "btnCancel", None)
                if isinstance(btn, QtWidgets.QPushButton):
                    btn.setEnabled(False)
                    try:
                        btn.clicked.disconnect(_cancel)
                    except Exception:
                        pass
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
        mode_map = {0: "count", 1: "delta", 2: "custom"}
        mode = mode_map.get(self.cboMode.currentIndex(), "count")
        return {
            "axis": axis, "p1": p1, "p2": p2,
            "thickness": float(self.spnThickness.value()),
            "mode": mode, "delta": float(self.spnDelta.value()), "count": int(self.spnCount.value()), "custom_rule": self.edCustomRule.text(),
            "start": float(self.spnStart.value()), "end": float(self.spnEnd.value()),
            "centroids": self.chkCentroids.isChecked(), "raw_polylines": self.chkRawPolylines.isChecked(),
            "refined_polylines": self.chkRefinedPolylines.isChecked(), "polygons": self.chkPolygons.isChecked(),
        }

    def set_visibility(self, kind: str, visible: bool) -> None:
        """Toggle visibility of a product overlay in the viewer."""
        self._vis[kind] = bool(visible)
        self._log("INFO", f"Visibility → {kind}={'ON' if visible else 'OFF'}")
        # Drive overlays
        try:
            self._update_overlays(kind)
        except Exception as ex:
            self._log("WARN", f"Overlay update failed for {kind}: {ex}")
    
    # ------------------------- Overlay helpers -------------------------
    def _plotter(self):
        """Return PyVista plotter from host viewer if available."""
        v = getattr(self.window, "viewer3d", None)
        return getattr(v, "plotter", None)

    def _clear_actors(self, kind: str) -> None:
        plt = self._plotter()
        if plt is None:
            self._actors[kind] = []
            return
        try:
            for a in self._actors.get(kind, []) or []:
                try:
                    # PyVista supports remove_actor(actor) or remove_actor(name); try both
                    try:
                        plt.remove_actor(a)
                    except Exception:
                        if isinstance(a, str):
                            plt.remove_actor(a)
                except Exception:
                    pass
        finally:
            self._actors[kind] = []

    def _update_overlays(self, kind: Optional[str] = None) -> None:
        """Update one or all overlay kinds based on current visibility and data."""
        kinds = [kind] if kind else list(self._vis.keys())
        for k in kinds:
            if not self._vis.get(k, False):
                self._clear_actors(k)
                continue
            if k == "planes":
                self._render_planes()
            elif k == "slices":
                self._render_slice_points()
            elif k == "current":
                self._render_current_slice()
            elif k == "centroids":
                self._render_centroids()
            elif k == "polylines":
                self._render_polylines()
            elif k == "polygons":
                self._render_polygons()

    def _dataset_extents(self, ds: Optional[int]):
        """Return (center_xyz, size_xyz) from active dataset bounds (safe)."""
        v = getattr(self.window, "viewer3d", None)
        recs = getattr(v, "_datasets", []) if v is not None else []
        if not (isinstance(ds, int) and 0 <= ds < len(recs)):
            return None, None
        pdata = recs[ds].get("pdata")
        if pdata is None or not hasattr(pdata, "bounds"):
            return None, None
        b = pdata.bounds
        cx = 0.5 * (b[0] + b[1]); cy = 0.5 * (b[2] + b[3]); cz = 0.5 * (b[4] + b[5])
        sx = abs(b[1] - b[0]); sy = abs(b[3] - b[2]); sz = abs(b[5] - b[4])
        return (cx, cy, cz), (sx, sy, sz)

    def _render_planes(self) -> None:
        plt = self._plotter()
        if plt is None:
            return
        self._clear_actors("planes")
        if not self._slices:
            return
        try:
            import pyvista as pv
        except Exception:
            return
        ds = self._current_dataset_index()
        center, size = self._dataset_extents(ds)
        if center is None or size is None:
            return
        sx, sy, sz = size
        # Slightly larger than bbox for visibility
        i_size_xy = max(1e-9, 1.05 * sx)
        j_size_xy = max(1e-9, 1.05 * sy)
        i_size_xz = max(1e-9, 1.05 * sx)
        j_size_xz = max(1e-9, 1.05 * sz)
        i_size_yz = max(1e-9, 1.05 * sy)
        j_size_yz = max(1e-9, 1.05 * sz)

        # Default axis from combo; each slice may override with its own axis
        default_axis = (self.cboAxis.currentText() or "Z").upper()
        actors = []
        for i, s in enumerate(self._slices):
            c = float(s.get("coord", 0.0))
            ax_s = (s.get("axis") or default_axis).upper()
            if not self._axis_visible(ax_s):
                continue
            color = self._axis_color(ax_s)
            opacity = float(self._style.get("planes_opacity", 0.18))
            if ax_s == "X":
                plane = pv.Plane(center=(c, center[1], center[2]), direction=(1, 0, 0),
                                 i_size=j_size_yz, j_size=i_size_yz)
            elif ax_s == "Y":
                plane = pv.Plane(center=(center[0], c, center[2]), direction=(0, 1, 0),
                                 i_size=i_size_xz, j_size=j_size_xz)
            else:  # Z
                plane = pv.Plane(center=(center[0], center[1], c), direction=(0, 0, 1),
                                 i_size=i_size_xy, j_size=j_size_xy)
            try:
                a = plt.add_mesh(plane, name=f"c2f_plane_{i}", color=color, opacity=opacity, pickable=False)
            except TypeError:
                a = plt.add_mesh(plane, color=color, opacity=opacity, pickable=False)
            actors.append(a if a is not None else f"c2f_plane_{i}")
        self._actors["planes"] = actors
        try:
            plt.render()
        except Exception:
            pass

    def _slice_indices_and_points(self, rec, pdata):
        """Return (indices, Nx3 points) inside the slice window for *rec*.

        Args:
            rec: dict with keys {axis, coord, thickness}
            pdata: active pyvista.PolyData
        Returns:
            (idx, pts) or (None, None) on failure/empty.
        """
        try:
            import numpy as np
        except Exception:
            return None, None
        if pdata is None or not hasattr(pdata, "points"):
            return None, None
        pts = pdata.points
        if pts is None:
            return None, None
        axis = (rec.get("axis") or self.cboAxis.currentText() or "Z").upper()
        coord = float(rec.get("coord", 0.0))
        thick = float(rec.get("thickness", 0.0))
        dim = {"X": 0, "Y": 1, "Z": 2}.get(axis, 2)
        lo = coord - 0.5 * thick
        hi = coord + 0.5 * thick
        try:
            mask = (pts[:, dim] >= lo) & (pts[:, dim] <= hi)
            idx = np.nonzero(mask)[0]
            if idx.size == 0:
                return None, None
            return idx, pts[idx]
        except Exception:
            return None, None

    def _render_slice_points(self) -> None:
        plt = self._plotter()
        if plt is None:
            return
        self._clear_actors("slices")
        if not self._slices:
            return
        try:
            import pyvista as pv
        except Exception:
            return
        ds = self._current_dataset_index()
        v = getattr(self.window, "viewer3d", None)
        recs = getattr(v, "_datasets", []) if v is not None else []
        if not (isinstance(ds, int) and 0 <= ds < len(recs)):
            return
        pdata = recs[ds].get("pdata")
        if pdata is None:
            return
        actors = []
        for i, s in enumerate(self._slices):
            ax_s = (s.get("axis") or self.cboAxis.currentText() or "Z").upper()
            if not self._axis_visible(ax_s):
                continue
            idx, pts = self._slice_indices_and_points(s, pdata)
            if pts is None:
                continue
            color = self._axis_color(ax_s)
            try:
                cloud = pv.PolyData(pts)
                a = plt.add_points(cloud, color=color,
                                   point_size=max(1.0, float(self._style.get("point_size_factor", 2.0)) * 2.0),
                                   render_points_as_spheres=False)
            except TypeError:
                a = plt.add_points(pts, color=color,
                                   point_size=max(1.0, float(self._style.get("point_size_factor", 2.0)) * 2.0),
                                   render_points_as_spheres=False)
            actors.append(a if a is not None else f"c2f_slice_pts_{i}")
        self._actors["slices"] = actors
        try:
            plt.render()
        except Exception:
            pass

    def _render_centroids(self) -> None:
        plt = self._plotter()
        if plt is None:
            return
        self._clear_actors("centroids")
        if not self._slices:
            return
        try:
            import pyvista as pv
            import numpy as np
        except Exception:
            return
        ds = self._current_dataset_index()
        v = getattr(self.window, "viewer3d", None)
        recs = getattr(v, "_datasets", []) if v is not None else []
        if not (isinstance(ds, int) and 0 <= ds < len(recs)):
            return
        pdata = recs[ds].get("pdata")
        if pdata is None:
            return
        actors = []
        for i, s in enumerate(self._slices):
            ax_s = (s.get("axis") or self.cboAxis.currentText() or "Z").upper()
            if not self._axis_visible(ax_s):
                continue
            idx, pts = self._slice_indices_and_points(s, pdata)
            if pts is None:
                continue
            try:
                c = pts.mean(axis=0)
            except Exception:
                continue
            color = self._axis_color(ax_s)
            try:
                a = plt.add_points(pv.PolyData(c.reshape(1, 3)), color=color,
                                   point_size=float(self._style.get("centroid_size", 12.0)),
                                   render_points_as_spheres=True)
            except TypeError:
                a = plt.add_points(c.reshape(1, 3), color=color,
                                   point_size=float(self._style.get("centroid_size", 12.0)),
                                   render_points_as_spheres=True)
            actors.append(a if a is not None else f"c2f_centroid_{i}")
        self._actors["centroids"] = actors
        try:
            plt.render()
        except Exception:
            pass

    def _render_current_slice(self) -> None:
        plt = self._plotter()
        if plt is None:
            return
        self._clear_actors("current")
        if not self._slices:
            return
        try:
            import pyvista as pv
        except Exception:
            return
        ds = self._current_dataset_index()
        center, size = self._dataset_extents(ds)
        if center is None or size is None:
            return
        sx, sy, sz = size
        # Pick the middle slice by index
        idx = len(self._slices) // 2
        c = float(self._slices[idx].get("coord", 0.0))
        ax_s = (self._slices[idx].get("axis") or (self.cboAxis.currentText() or "Z")).upper()
        if not self._axis_visible(ax_s):
            return
        color = self._axis_color(ax_s)
        opacity = float(self._style.get("current_plane_opacity", 0.35))
        if ax_s == "X":
            plane = pv.Plane(center=(c, center[1], center[2]), direction=(1, 0, 0),
                             i_size=1.08 * max(1e-9, sy), j_size=1.08 * max(1e-9, sz))
        elif ax_s == "Y":
            plane = pv.Plane(center=(center[0], c, center[2]), direction=(0, 1, 0),
                             i_size=1.08 * max(1e-9, sx), j_size=1.08 * max(1e-9, sz))
        else:
            plane = pv.Plane(center=(center[0], center[1], c), direction=(0, 0, 1),
                             i_size=1.08 * max(1e-9, sx), j_size=1.08 * max(1e-9, sy))
        try:
            a = plt.add_mesh(plane, name="c2f_slice_current", color=color, opacity=opacity, pickable=False)
        except TypeError:
            a = plt.add_mesh(plane, color=color, opacity=opacity, pickable=False)
        self._actors["current"] = [a if a is not None else "c2f_slice_current"]
        try:
            plt.render()
        except Exception:
            pass
    
    def _render_polylines(self) -> None:
        plt = self._plotter()
        if plt is None: return
        self._clear_actors("polylines")
        if not self._slices: return
        try:
            import pyvista as pv, numpy as np
        except Exception:
            return
        actors = []
        for i, s in enumerate(self._slices):
            ax_s = (s.get("axis") or self.cboAxis.currentText() or "Z").upper()
            if not self._axis_visible(ax_s):
                continue
            color = self._axis_color(ax_s)
            width = float(self._style.get("line_width", 2.0))
            polys = s.get("polyline_refined") or s.get("polyline_raw")
            if not polys:
                continue
            seq = polys if isinstance(polys, (list, tuple)) else [polys]
            for j, arr in enumerate(seq):
                try:
                    pts = np.asarray(arr, dtype=float).reshape(-1, 3)
                    if pts.shape[0] < 2: continue
                except Exception:
                    continue
                # PolyData line
                n = pts.shape[0]
                cells = np.hstack([[n], np.arange(n, dtype=np.int32)])
                pd = pv.PolyData(pts); pd.lines = cells
                try:
                    a = plt.add_mesh(pd, color=color, line_width=width, render_lines_as_tubes=True)
                except TypeError:
                    a = plt.add_mesh(pd, color=color)
                actors.append(a if a is not None else f"c2f_polyline_{i}_{j}")
                # Vertex glyphs (small cones as “spigoli” markers)
                try:
                    centers = pv.PolyData(pts)
                    glyph = pv.Cone(direction=(0,0,1), height=0.02, radius=0.008, resolution=8)
                    g = centers.glyph(geom=glyph, scale=False)
                    ga = plt.add_mesh(g, color=color, opacity=0.9)
                    actors.append(ga if ga is not None else f"c2f_poly_glyph_{i}_{j}")
                except Exception:
                    pass
        self._actors["polylines"] = actors
        try: plt.render()
        except Exception: pass
    
    def _render_polygons(self) -> None:
        plt = self._plotter()
        if plt is None: return
        self._clear_actors("polygons")
        if not self._slices: return
        try:
            import pyvista as pv, numpy as np
        except Exception:
            return
        actors = []
        for i, s in enumerate(self._slices):
            ax_s = (s.get("axis") or self.cboAxis.currentText() or "Z").upper()
            if not self._axis_visible(ax_s):
                continue
            color = self._axis_color(ax_s)
            edge_w = float(self._style.get("poly_edge_width", 1.5))
            opacity = float(self._style.get("poly_opacity", 0.25))
            polys = s.get("polygons")
            if not polys:
                continue
            seq = polys if isinstance(polys, (list, tuple)) else [polys]
            for j, poly in enumerate(seq):
                try:
                    pts = np.asarray(poly, dtype=float).reshape(-1, 3)
                    if pts.shape[0] < 3: continue
                    n = pts.shape[0]
                    faces = np.hstack([[n], np.arange(n, dtype=np.int32)])
                    surf = pv.PolyData(pts); surf.faces = faces
                except Exception:
                    continue
                try:
                    a = plt.add_mesh(surf, color=color, opacity=opacity,
                                    show_edges=True, edge_color=color, line_width=edge_w)
                except TypeError:
                    a = plt.add_mesh(surf, color=color, opacity=opacity)
                actors.append(a if a is not None else f"c2f_polygon_{i}_{j}")
        self._actors["polygons"] = actors
        try: plt.render()
        except Exception: pass
    # ----------------------- Host integration --------------------------    
    def _ensure_tree(self) -> None:
        """Ensure the dataset node contains the *Cloud2FEM* sections.

        Structure:
            <dataset(top-level)>
              - Cloud2FEM
                  - slices
                  - centroids
                  - polylines
                  - polygons
        Notes:
            - We no longer create a duplicate "Point Cloud/Normals" branch.
            - Items are selectable but not checkable to avoid host visibility side-effects.
            - Each child stores a UserRole dict {plugin:"cloud2fem", kind:"..."} for host-side routing.
        """
        try:
            # Prefer treeDatasets, else generic 'tree' (AVOID treeMCTS to prevent recursion)
            tree = (getattr(self.window, "treeDatasets", None) or
                    getattr(self.window, "tree", None))
            if tree is None or not hasattr(tree, "invisibleRootItem"):
                return

            # Find active dataset name from viewer
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
            ds_item = None
            # 1) exact match among top-level items
            for i in range(root.childCount()):
                it = root.child(i)
                if cur_name and it.text(0) == cur_name:
                    ds_item = it
                    break
            # 2) if no match, fall back to the top-level ancestor of the current item (if any)
            if ds_item is None:
                cur = tree.currentItem()
                if cur is not None:
                    # climb to top-level
                    while cur.parent() is not None:
                        cur = cur.parent()
                    ds_item = cur
            # 3) as a last resort, do nothing
            if ds_item is None:
                return

            def ensure_child(parent: QtWidgets.QTreeWidgetItem, label: str, kind: str) -> QtWidgets.QTreeWidgetItem:
                # find existing
                for i in range(parent.childCount()):
                    it = parent.child(i)
                    if it.text(0) == label:
                        # ensure flags and role are set
                        it.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                        it.setData(0, QtCore.Qt.UserRole, {"plugin": "cloud2fem", "kind": kind})
                        return it
                # create new
                it = QtWidgets.QTreeWidgetItem([label])
                it.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                it.setData(0, QtCore.Qt.UserRole, {"plugin": "cloud2fem", "kind": kind})
                parent.addChild(it)
                return it

            # Build Cloud2FEM branch
            # Avoid duplicate by searching for an existing child named "Cloud2FEM"
            c2f = None
            for i in range(ds_item.childCount()):
                cand = ds_item.child(i)
                if cand.text(0) == "Cloud2FEM":
                    c2f = cand
                    break
            if c2f is None:
                c2f = QtWidgets.QTreeWidgetItem(["Cloud2FEM"])
                c2f.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                c2f.setData(0, QtCore.Qt.UserRole, {"plugin": "cloud2fem", "kind": "group"})
                ds_item.addChild(c2f)

            ensure_child(c2f, "slices",     "slices")
            ensure_child(c2f, "centroids",  "centroids")
            ensure_child(c2f, "polylines",  "polylines")
            ensure_child(c2f, "polygons",   "polygons")

            tree.expandItem(ds_item)
            tree.expandItem(c2f)
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

    def _on_open_display_props(self) -> None:
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Cloud2FEM Display Properties")
        form = QtWidgets.QFormLayout(dlg)
        spnPoint = QtWidgets.QDoubleSpinBox(); spnPoint.setRange(0.1, 50.0); spnPoint.setValue(float(self._style.get("point_size_factor", 2.0)))
        spnCent  = QtWidgets.QDoubleSpinBox(); spnCent.setRange(1.0, 64.0);  spnCent.setValue(float(self._style.get("centroid_size", 12.0)))
        spnPOp   = QtWidgets.QDoubleSpinBox(); spnPOp.setRange(0.0, 1.0); spnPOp.setSingleStep(0.05); spnPOp.setValue(float(self._style.get("planes_opacity", 0.18)))
        spnCOp   = QtWidgets.QDoubleSpinBox(); spnCOp.setRange(0.0, 1.0); spnCOp.setSingleStep(0.05); spnCOp.setValue(float(self._style.get("current_plane_opacity", 0.35)))
        spnLW    = QtWidgets.QDoubleSpinBox(); spnLW.setRange(0.1, 20.0); spnLW.setValue(float(self._style.get("line_width", 2.0)))
        spnPolyO = QtWidgets.QDoubleSpinBox(); spnPolyO.setRange(0.0, 1.0); spnPolyO.setSingleStep(0.05); spnPolyO.setValue(float(self._style.get("poly_opacity", 0.25)))
        spnPEW   = QtWidgets.QDoubleSpinBox(); spnPEW.setRange(0.1, 20.0); spnPEW.setValue(float(self._style.get("poly_edge_width", 1.5)))
        form.addRow("Point size ×:", spnPoint)
        form.addRow("Centroid size:", spnCent)
        form.addRow("Planes opacity:", spnPOp)
        form.addRow("Current plane opacity:", spnCOp)
        form.addRow("Polyline width:", spnLW)
        form.addRow("Polygon opacity:", spnPolyO)
        form.addRow("Polygon edge width:", spnPEW)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        form.addRow(btns)
        def _apply():
            self._style["point_size_factor"] = float(spnPoint.value())
            self._style["centroid_size"] = float(spnCent.value())
            self._style["planes_opacity"] = float(spnPOp.value())
            self._style["current_plane_opacity"] = float(spnCOp.value())
            self._style["line_width"] = float(spnLW.value())
            self._style["poly_opacity"] = float(spnPolyO.value())
            self._style["poly_edge_width"] = float(spnPEW.value())
            self._update_overlays()
            dlg.accept()
        btns.accepted.connect(_apply)
        btns.rejected.connect(dlg.reject)
        dlg.exec()
    
    def _sync_mct_store(self) -> None:
        """Mirror current slices/products into the active dataset record.

        Only the axis involved in *this* compute pass is replaced; other axes are preserved.
        Store schema (per-axis arrays): slices/centroids/polylines/polygons with keys X,Y,Z,C.
        """
        try:
            if not self._slices:
                self._log("INFO", "[c2f] _sync_mct_store: no slices in RAM; skip")
                return
            # Detect current axis from the first slice (fallback to combo)
            cur_axis = (self._slices[0].get("axis") or (self.cboAxis.currentText() or "Z")).upper()
            if cur_axis not in ("X","Y","Z"): cur_axis = "C"

            v = getattr(self.window, "viewer3d", None)
            if v is None:
                self._log("WARN", "[c2f] _sync_mct_store: viewer3d not found")
                return
            recs = getattr(v, "_datasets", None)
            if not isinstance(recs, list) or not recs:
                self._log("WARN", "[c2f] _sync_mct_store: no datasets in viewer3d")
                return
            ds = self._current_dataset_index()
            if not (isinstance(ds, int) and 0 <= ds < len(recs)):
                self._log("WARN", f"[c2f] _sync_mct_store: invalid dataset index {ds}")
                return
            rec = recs[ds]
            if not isinstance(rec, dict):
                self._log("WARN", "[c2f] _sync_mct_store: dataset record is not a dict; cannot attach store")
                return

            store = rec.setdefault("cloud2fem", {})

            def _axis_map():
                return {"X": [], "Y": [], "Z": [], "C": []}

            slices_map    = store.setdefault("slices",    _axis_map())
            centroids_map = store.setdefault("centroids", _axis_map())
            polylines_map = store.setdefault("polylines", _axis_map())
            polygons_map  = store.setdefault("polygons",  _axis_map())

            # Rebuild ONLY the current axis, preserve others
            slices_map[cur_axis]    = []
            centroids_map[cur_axis] = []
            polylines_map[cur_axis] = []
            polygons_map[cur_axis]  = []

            for s in (self._slices or []):
                ax = (s.get("axis") or cur_axis).upper()
                if ax not in ("X","Y","Z"): ax = "C"
                ax = cur_axis  # normalize all slices of this run under the current axis bucket
                slices_map[ax].append({
                    "coord": float(s.get("coord", 0.0)),
                    "thickness": float(s.get("thickness", 0.0)),
                })
                C = s.get("centroids");     centroids_map[ax].append([] if C in (None, False) else C)
                P = s.get("polyline_refined") or s.get("polyline_raw"); polylines_map[ax].append([] if P in (None, False) else P)
                G = s.get("polygons");      polygons_map[ax].append([] if G in (None, False) else G)

            self._log("INFO", f"[c2f] store sync: axis={cur_axis} slices={len(self._slices)} | totals: X={len(slices_map['X'])} Y={len(slices_map['Y'])} Z={len(slices_map['Z'])} C={len(slices_map['C'])}")
        except Exception as ex:
            self._log("WARN", f"[c2f] _sync_mct_store failed: {ex}")
    def _on_thickness_changed(self, _v: float) -> None:
        """Mark that the thickness was set by the user when the widget has focus.

        We use a small guard to distinguish programmatic changes from manual edits.
        """
        if self._setting_thickness:
            return
        # Heuristic: consider it a user change if the spinbox currently has focus
        try:
            if self.spnThickness.hasFocus():
                self._thickness_user_set = True
        except Exception:
            # Fallback: assume user intent
            self._thickness_user_set = True