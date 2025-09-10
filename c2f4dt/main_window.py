from __future__ import annotations
import os
from typing import Optional
from PySide6 import QtCore, QtGui, QtWidgets

from .utils.theme import apply_user_theme
from .utils.systeminfo import disk_usage_percent
from .utils.icons import qicon
from .plugins.manager import PluginManager
from .ui.console import ConsoleWidget
from .ui.viewer3d import Viewer3DPlaceholder
from .ui.display_panel import DisplayPanel

try:
    from .ui.viewer3d import Viewer3D as _Viewer3D
    _HAS_PYVISTA = True
except Exception:
    _HAS_PYVISTA = False
    _Viewer3D = Viewer3DPlaceholder

class MainWindow(QtWidgets.QMainWindow):
    """Main GUI window for C2F4DT."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("C2F4DT – Release 1.0")
        apply_user_theme(self)
        self._apply_styles()

        self._compute_initial_geometry()

        self.mcts = {}
        self.mct = {}

        self.undo_stack = QtGui.QUndoStack(self)
        self._build_actions()
        self._build_menus()
        self._build_toolbars()
        self._build_central_area()
        self._build_statusbar()

        self.plugin_manager = PluginManager(self, extensions_dir=self._default_extensions_dir())
        self._populate_plugins_ui()

        self._start_disk_timer()

        self.undo_stack.cleanChanged.connect(self._on_undo_changed)
        self.undo_stack.indexChanged.connect(self._on_undo_changed)
        self._on_undo_changed()

        # Preferences
        self.downsample_method = "random"  # or "voxel"

        # Tree update guard to avoid cascading on auto-updates/partial states
        self._tree_updating = False

    def _apply_styles(self) -> None:
        css = r'''
QPushButton#buttonCANCEL {
  background-color: #c62828;
  color: white;
  border: 1px solid #8e0000;
  padding: 2px 8px;
  border-radius: 3px;
}
QPushButton#buttonCANCEL:disabled {
  background-color: #ef9a9a;
  color: #333;
  border-color: #e57373;
}

QProgressBar#barPROGRESS {
  border: 1px solid #3c3f41;
  border-radius: 3px;
  text-align: center;
  background: #2b2b2b;
  color: #e6e6e6;
}
QProgressBar#barPROGRESS::chunk {
  background-color: #43a047;
}

QProgressBar#diskUsageBar {
  border: 1px solid #3c3f41;
  border-radius: 3px;
  background: #2b2b2b;
  text-align: center;
  color: #e6e6e6;
}
QProgressBar#diskUsageBar::chunk {
  background-color: #5dade2;
}
'''
        self.setStyleSheet(css)

    def _compute_initial_geometry(self) -> None:
        scr = QtGui.QGuiApplication.screenAt(QtGui.QCursor.pos())
        if scr is None:
            scr = QtGui.QGuiApplication.primaryScreen()
        avail = scr.availableGeometry()
        w = int(avail.width() * 0.78)
        h = int(avail.height() * 0.78)
        x = avail.x() + (avail.width() - w) // 2
        y = avail.y() + (avail.height() - h) // 2
        self.setGeometry(x, y, w, h)

    def _default_extensions_dir(self) -> str:
        base = os.path.dirname(os.path.dirname(__file__))
        return os.path.join(base, "..", "extensions")

    def _build_actions(self) -> None:
        self.act_new = QtGui.QAction(qicon("32x32_document-new.png"), "New", self)
        self.act_new.setShortcut(QtGui.QKeySequence.New)
        self.act_open = QtGui.QAction(qicon("32x32_document-open.png"), "Open…", self)
        self.act_open.setShortcut(QtGui.QKeySequence.Open)
        self.act_save = QtGui.QAction(qicon("32x32_document-save.png"), "Save", self)
        self.act_save.setShortcut(QtGui.QKeySequence.Save)
        self.act_save_as = QtGui.QAction(qicon("32x32_document-save-as.png"), "Save As…", self)
        self.act_save_as.setShortcut(QtGui.QKeySequence.SaveAs)
        self.act_import_cloud = QtGui.QAction(qicon("32x32_import_cloud.png"), "Import Cloud", self)
        self.act_import_cloud.triggered.connect(self._on_import_cloud)

        self.act_undo = self.undo_stack.createUndoAction(self, "Undo"); self.act_undo.setIcon(qicon("32x32_edit-undo.png"))
        self.act_undo.setShortcut(QtGui.QKeySequence.Undo)
        self.act_redo = self.undo_stack.createRedoAction(self, "Redo"); self.act_redo.setIcon(qicon("32x32_edit-redo.png"))
        self.act_redo.setShortcut(QtGui.QKeySequence.Redo)
        self.act_clear = QtGui.QAction(qicon("32x32_edit-clear.png"), "Clear", self)

        self.act_tab_interaction = QtGui.QAction(qicon("32x32_3D_inspector.png"), "Interaction", self)
        self.act_tab_slices = QtGui.QAction(qicon("32x32_slice.png"), "Slices", self)
        self.act_tab_fem = QtGui.QAction(qicon("32x32_mesh_generation.png"), "FEM/Mesh", self)
        self.act_tab_inspector = QtGui.QAction(qicon("32x32_model_info.png"), "Inspector", self)
        self.act_open_2d = QtGui.QAction(qicon("32x32_2D_window.png"), "Open 2D Viewer", self)

        self.act_create_grid = QtGui.QAction(qicon("32x32_grid_generation.png"), "Create Grid", self)
        self.act_toggle_grid = QtGui.QAction(qicon("32x32_3D_grid.png"), "Toggle Grid", self); self.act_toggle_grid.setCheckable(True)
        self.act_toggle_normals = QtGui.QAction(qicon("32x32_normals.png"), "Toggle Normals", self); self.act_toggle_normals.setCheckable(True)

        self.act_fit = QtGui.QAction(qicon("32x32_view-fullscreen.png"), "Fit view", self)
        self.act_xp = QtGui.QAction(qicon("32x32_view_Xp.png"), "View +X", self)
        self.act_xm = QtGui.QAction(qicon("32x32_view_Xm.png"), "View −X", self)
        self.act_yp = QtGui.QAction(qicon("32x32_view_Yp.png"), "View +Y", self)
        self.act_ym = QtGui.QAction(qicon("32x32_view_Ym.png"), "View −Y", self)
        self.act_zp = QtGui.QAction(qicon("32x32_view_Zp.png"), "View +Z", self)
        self.act_zm = QtGui.QAction(qicon("32x32_view_Zm.png"), "View −Z", self)
        self.act_iso_p = QtGui.QAction(qicon("32x32_view_Isometric_p.png"), "Isometric +", self)
        self.act_iso_m = QtGui.QAction(qicon("32x32_view_Isometric_m.png"), "Isometric −", self)
        self.act_invert = QtGui.QAction(qicon("32x32_view_inverted.png"), "Invert view", self)
        self.act_refresh = QtGui.QAction(qicon("32x32_view-refresh.png"), "Refresh", self)

    def _build_menus(self) -> None:
        menubar = self.menuBar()
        m_file = menubar.addMenu("&File")
        for a in [self.act_new, self.act_open, self.act_save, self.act_save_as, self.act_import_cloud]: m_file.addAction(a)
        m_edit = menubar.addMenu("&Edit")
        for a in [self.act_undo, self.act_redo, self.act_clear]: m_edit.addAction(a)
        m_view = menubar.addMenu("&View")
        for a in [self.act_tab_interaction, self.act_tab_slices, self.act_tab_fem, self.act_tab_inspector, self.act_open_2d]: m_view.addAction(a)
        m_tools = menubar.addMenu("&Tools")
        for a in [self.act_create_grid, self.act_toggle_grid, self.act_toggle_normals]:
            m_tools.addAction(a)

        # Rendering submenu
        self.act_safe_render = QtGui.QAction("Safe Rendering (macOS)", self)
        self.act_safe_render.setCheckable(True)
        self.act_safe_render.toggled.connect(lambda on: getattr(self.viewer3d, "enable_safe_rendering", lambda *_: None)(on))

        self.act_points_as_spheres = QtGui.QAction("Points as spheres", self)
        self.act_points_as_spheres.setCheckable(True)
        self.act_points_as_spheres.setChecked(True)
        self.act_points_as_spheres.toggled.connect(lambda on: getattr(self.viewer3d, "set_points_as_spheres", lambda *_: None)(on))

        m_render = m_tools.addMenu("Rendering")
        m_render.addAction(self.act_safe_render)
        m_render.addAction(self.act_points_as_spheres)

        # Downsampling submenu (import-time behavior)
        m_ds = m_tools.addMenu("Downsampling")
        group_ds = QtGui.QActionGroup(self)
        group_ds.setExclusive(True)
        self.act_ds_random = QtGui.QAction("Random (accurate %)", self, checkable=True)
        self.act_ds_voxel = QtGui.QAction("Voxel (spatial)", self, checkable=True)
        self.act_ds_random.setChecked(True)
        for a in (self.act_ds_random, self.act_ds_voxel):
            group_ds.addAction(a)
            m_ds.addAction(a)
        self.act_ds_random.triggered.connect(lambda: setattr(self, "downsample_method", "random"))
        self.act_ds_voxel.triggered.connect(lambda: setattr(self, "downsample_method", "voxel"))

        # Testing submenu: run external Python scripts in the console context
        m_test = m_tools.addMenu("Testing")
        self.act_run_script = QtGui.QAction("Run Script…", self)
        self.act_run_script.triggered.connect(self._on_run_script)
        m_test.addAction(self.act_run_script)

        # Example convenience action (optional) to run project triplet tests
        self.act_run_triplet = QtGui.QAction("Run Triplet Import (tests/*.ply)", self)
        self.act_run_triplet.triggered.connect(self._on_run_tests_triplet)
        m_test.addAction(self.act_run_triplet)

        menubar.addMenu("&Help")

    def _build_toolbars(self) -> None:
        self.top_toolbar = QtWidgets.QToolBar("barTOPCOMMAND", self)
        self.top_toolbar.setIconSize(QtCore.QSize(32, 32))
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.top_toolbar)

        for a in [self.act_new, self.act_open, self.act_save, self.act_save_as, self.act_import_cloud]:
            self.top_toolbar.addAction(a)
        self.top_toolbar.addSeparator()
        for a in [self.act_undo, self.act_redo, self.act_clear]:
            self.top_toolbar.addAction(a)
        self.top_toolbar.addSeparator()
        for a in [self.act_tab_interaction, self.act_tab_slices, self.act_tab_fem, self.act_tab_inspector, self.act_open_2d]:
            self.top_toolbar.addAction(a)
        self.top_toolbar.addSeparator()
        for a in [self.act_create_grid, self.act_toggle_grid, self.act_toggle_normals]:
            self.top_toolbar.addAction(a)

        self.left_toolbar = QtWidgets.QToolBar("barVERTICALCOMMAND_left", self)
        self.left_toolbar.setIconSize(QtCore.QSize(32, 32))
        self.left_toolbar.setOrientation(QtCore.Qt.Vertical)
        self.addToolBar(QtCore.Qt.LeftToolBarArea, self.left_toolbar)
        for a in [self.act_fit, self.act_xp, self.act_xm, self.act_yp, self.act_ym, self.act_zp, self.act_zm, self.act_iso_p, self.act_iso_m, self.act_invert, self.act_refresh]:
            self.left_toolbar.addAction(a)

        self.right_toolbar = QtWidgets.QToolBar("barVERTICALCOMMAND_right", self)
        self.right_toolbar.setIconSize(QtCore.QSize(32, 32))
        self.right_toolbar.setOrientation(QtCore.Qt.Vertical)
        self.addToolBar(QtCore.Qt.RightToolBarArea, self.right_toolbar)

        # Connect view actions
        self.act_fit.triggered.connect(lambda: self.viewer3d.view_fit())
        self.act_xp.triggered.connect(lambda: self.viewer3d.view_axis("+X"))
        self.act_xm.triggered.connect(lambda: self.viewer3d.view_axis("-X"))
        self.act_yp.triggered.connect(lambda: self.viewer3d.view_axis("+Y"))
        self.act_ym.triggered.connect(lambda: self.viewer3d.view_axis("-Y"))
        self.act_zp.triggered.connect(lambda: self.viewer3d.view_axis("+Z"))
        self.act_zm.triggered.connect(lambda: self.viewer3d.view_axis("-Z"))
        self.act_iso_p.triggered.connect(lambda: self.viewer3d.view_iso(True))
        self.act_iso_m.triggered.connect(lambda: self.viewer3d.view_iso(False))
        self.act_invert.triggered.connect(lambda: self.viewer3d.invert_view())
        self.act_refresh.triggered.connect(lambda: self.viewer3d.refresh())

    def _build_central_area(self) -> None:
        central = QtWidgets.QWidget(self)
        central_layout = QtWidgets.QVBoxLayout(central)
        central_layout.setContentsMargins(4, 4, 4, 4); central_layout.setSpacing(6)

        mid_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal, central)

        self.tabINTERACTION = QtWidgets.QTabWidget(mid_split)
        self.tabINTERACTION.setObjectName("tabINTERACTION")

        self.tabDISPLAY = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(self.tabDISPLAY); v.setContentsMargins(4, 4, 4, 4)
        split = QtWidgets.QSplitter(QtCore.Qt.Vertical, self.tabDISPLAY)

        self.treeMCTS = QtWidgets.QTreeWidget()
        self.treeMCTS.setHeaderLabels(["Object"]); self.treeMCTS.setColumnCount(1)
        self.treeMCTS.itemChanged.connect(self._on_tree_item_changed)
        self.treeMCTS.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.treeMCTS.customContextMenuRequested.connect(self._on_tree_context_menu)
        self.treeMCTS.itemSelectionChanged.connect(self._on_tree_selection_changed)
        split.addWidget(self.treeMCTS)
        

        self.scrollDISPLAY = QtWidgets.QScrollArea(); self.scrollDISPLAY.setWidgetResizable(True)
        self.displayPanel = DisplayPanel(); self.scrollDISPLAY.setWidget(self.displayPanel)
        split.addWidget(self.scrollDISPLAY)

        v.addWidget(split)
        self.tabINTERACTION.addTab(self.tabDISPLAY, "DISPLAY")

        self.tabSLICING = QtWidgets.QWidget()
        v2 = QtWidgets.QVBoxLayout(self.tabSLICING)
        self.scrollSLICING = QtWidgets.QScrollArea(); self.scrollSLICING.setWidgetResizable(True)
        v2.addWidget(self.scrollSLICING)
        self.tabINTERACTION.addTab(self.tabSLICING, "SLICING")

        self.tabFEM = QtWidgets.QWidget()
        v3 = QtWidgets.QVBoxLayout(self.tabFEM)
        self.scrollFEM = QtWidgets.QScrollArea(); self.scrollFEM.setWidgetResizable(True)
        v3.addWidget(self.scrollFEM)
        self.tabINTERACTION.addTab(self.tabFEM, "FEM")

        self.tabINSPECTOR = QtWidgets.QWidget()
        v4 = QtWidgets.QVBoxLayout(self.tabINSPECTOR)
        self.treeMCT = QtWidgets.QTreeWidget(); self.treeMCT.setHeaderLabels(["Current MCT"])
        v4.addWidget(self.treeMCT)
        self.tabINTERACTION.addTab(self.tabINSPECTOR, "INSPECTOR")

        viewer_container = QtWidgets.QWidget()
        viewer_layout = QtWidgets.QVBoxLayout(viewer_container)
        viewer_layout.setContentsMargins(4, 4, 4, 4); viewer_layout.setSpacing(4)

        bar_plugin = QtWidgets.QHBoxLayout()
        self.comboPlugins = QtWidgets.QComboBox()
        self.comboPlugins.addItem("— No plugins installed —"); self.comboPlugins.setEnabled(False)
        bar_plugin.addWidget(QtWidgets.QLabel("Plugin scope:"))
        bar_plugin.addWidget(self.comboPlugins, 1)
        viewer_layout.addLayout(bar_plugin)

        self.viewer3d = _Viewer3D()
        viewer_layout.addWidget(self.viewer3d, 1)

        mid_split.addWidget(self.tabINTERACTION)
        mid_split.addWidget(viewer_container)
        mid_split.setStretchFactor(0, 0); mid_split.setStretchFactor(1, 1)

        central_layout.addWidget(mid_split, 1)

        self.tabCONSOLE_AND_MESSAGES = QtWidgets.QTabWidget()
        self.tabMESSAGES = QtWidgets.QWidget()
        vm = QtWidgets.QVBoxLayout(self.tabMESSAGES)
        self.txtMessages = QtWidgets.QPlainTextEdit(); self.txtMessages.setReadOnly(True)
        vm.addWidget(self.txtMessages)
        self.tabCONSOLE_AND_MESSAGES.addTab(self.tabMESSAGES, "MESSAGES")

        self.console = ConsoleWidget(context_provider=self._console_context)
        self.console.sigExecuted.connect(self._on_console_executed)
        self.tabCONSOLE_AND_MESSAGES.addTab(self.console, "CONSOLE")

        central_layout.addWidget(self.tabCONSOLE_AND_MESSAGES, 0)

        self.setCentralWidget(central)

        # Wire DisplayPanel to viewer
        self.displayPanel.sigPointSizeChanged.connect(self.viewer3d.set_point_size)
        self.displayPanel.sigPointBudgetChanged.connect(self.viewer3d.set_point_budget)
        self.displayPanel.sigColorModeChanged.connect(self.viewer3d.set_color_mode)
        self.displayPanel.sigSolidColorChanged.connect(lambda c: self.viewer3d.set_solid_color(c.red(), c.green(), c.blue()))
        self.displayPanel.sigColormapChanged.connect(self.viewer3d.set_colormap)

    def _on_console_executed(self, cmd: str) -> None:
        """Append executed console command to the MESSAGES panel."""
        try:
            self.txtMessages.appendPlainText(cmd)
        except Exception:
            pass

    def _on_tree_selection_changed(self) -> None:
        """Keep `mct` synced with the selected dataset (if any)."""
        item = self.treeMCTS.currentItem()
        if item is None:
            return
        # Walk up to the root item (dataset node)
        root = item
        while root.parent() is not None:
            root = root.parent()
        name = root.text(0)
        entry = self.mcts.get(name)
        if entry:
            self.mct = entry

    def _build_statusbar(self) -> None:
        sb = QtWidgets.QStatusBar(self)
        self.setStatusBar(sb)

        self.btnCancel = QtWidgets.QPushButton("CANCEL")
        self.btnCancel.setObjectName("buttonCANCEL")
        self.btnCancel.setEnabled(False)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setObjectName("barPROGRESS")
        self.progress.setRange(0, 100); self.progress.setValue(0)
        self.progress.setFormat("Idle")

        self.disk = QtWidgets.QProgressBar()
        self.disk.setObjectName("diskUsageBar")
        self.disk.setRange(0, 100); self.disk.setValue(0)
        self.disk.setTextVisible(True)

        sb.addWidget(self.btnCancel, 0)
        sb.addWidget(self.progress, 1)
        sb.addPermanentWidget(self.disk, 0)

    def _start_disk_timer(self) -> None:
        self._disk_timer = QtCore.QTimer(self)
        self._disk_timer.timeout.connect(self._update_disk)
        self._disk_timer.start(2000)
        self._update_disk()

    def _update_disk(self) -> None:
        used, free, percent = disk_usage_percent()
        self.disk.setValue(int(percent))
        self.disk.setFormat(f"Disk used: {percent:.0f}%")

    def _populate_plugins_ui(self) -> None:
        names = self.plugin_manager.available_plugins()
        self.comboPlugins.clear()
        if not names:
            self.comboPlugins.addItem("— No plugins installed —")
            self.comboPlugins.setEnabled(False)
        else:
            self.comboPlugins.addItems(names)
            self.comboPlugins.setEnabled(True)

    def _on_undo_changed(self) -> None:
        self.act_undo.setEnabled(self.undo_stack.canUndo())
        self.act_redo.setEnabled(self.undo_stack.canRedo())

    def _console_context(self) -> dict:
        return {"mcts": self.mcts, "mct": self.mct, "window": self, "undo_stack": self.undo_stack}
    
    def _on_import_cloud(self) -> None:
        """Handle Import Cloud: open dialog, parse file, show summary, then add to scene & tree."""
        dlg = QtWidgets.QFileDialog(self, "Import point cloud / mesh")
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        dlg.setNameFilters([
            "All supported (*.ply *.obj *.vtp *.stl *.vtk *.gltf *.glb *.las *.laz *.e57)",
            "Point clouds (*.ply *.las *.laz *.e57)",
            "Meshes (*.ply *.obj *.vtp *.stl *.vtk *.gltf *.glb)",
            "All files (*)",
        ])
        if not dlg.exec():
            return
        paths = dlg.selectedFiles()
        if not paths:
            return
        path = paths[0]

        # Import
        from .utils.io.importers import import_file
        from .ui.import_summary_dialog import ImportSummaryDialog

        try:
            objects = import_file(path)
        except Exception as ex:
            QtWidgets.QMessageBox.critical(self, "Import error", str(ex))
            return

        # Summary dialog
        summary = ImportSummaryDialog(objects, self)
        if summary.exec() != QtWidgets.QDialog.Accepted:
            return

        # Retrieve per-object operations from the summary (axis mapping, normals, color pref)
        ops = summary.operations()

        # --- helpers -----------------------------------------------------
        def _apply_axis_map(arr, axis_map):
            """Apply an axis remapping with sign to an (N,3) array.

            Args:
                arr: Points or normals array with shape (N, 3) or None.
                axis_map: Dict like {'X': '+Y', 'Y': '-Z', 'Z': '+X'}.
            Returns:
                New array with same shape, or the original if None/error.
            """
            try:
                import numpy as _np
                if arr is None:
                    return None
                src = _np.asarray(arr, dtype=float)
                if src.ndim != 2 or src.shape[1] != 3:
                    return arr
                out = _np.empty_like(src)
                axes = {"X": 0, "Y": 1, "Z": 2}
                for tgt_key, expr in axis_map.items():
                    sign = -1.0 if expr.startswith("-") else 1.0
                    src_axis = axes[expr[-1]]  # last char is X/Y/Z
                    out[:, axes[tgt_key]] = sign * src[:, src_axis]
                return out
            except Exception:
                return arr

        def _compute_normals(obj):
            """Compute rough normals if missing using PyVista (best-effort)."""
            try:
                import numpy as _np
                import pyvista as _pv  # type: ignore
                if obj.points is None or obj.points.shape[0] == 0:
                    return None
                pdata = _pv.PolyData(_np.asarray(obj.points))
                pdata = pdata.compute_normals(
                    consistent=False,
                    auto_orient_normals=False,
                    feature_angle=180.0,
                )
                n = getattr(pdata, 'point_normals', None)
                if n is not None:
                    return _np.asarray(n, dtype=_np.float32)
            except Exception:
                return None
            return None
        # -----------------------------------------------------------------

        # 1) Apply axis mapping / normals ops per object (before downsampling)
        for obj, spec in zip(objects, ops):
            # Points mapping
            obj.points = _apply_axis_map(obj.points, spec['axis_map'])
            # Normals mapping or compute if missing
            if spec.get('map_normals', True):
                if getattr(obj, 'normals', None) is not None:
                    obj.normals = _apply_axis_map(obj.normals, spec['axis_map'])
                elif spec.get('compute_normals_if_missing', False):
                    n = _compute_normals(obj)
                    if n is not None:
                        obj.normals = n
            # Store color preference for later use
            obj.meta['color_preference'] = spec.get('color_preference', 'rgb')

        # SUGGESTED VIEW BUDGET (cap as hint): compute a percent based on total visible points
        try:
            import sys
            # Count current visible points in the viewer
            total_current = 0
            try:
                for _rec in getattr(self.viewer3d, "_datasets", []):
                    if not _rec.get("visible", True):
                        continue
                    fp = _rec.get("full_pdata", _rec.get("pdata"))
                    if hasattr(fp, "n_points"):
                        total_current += int(fp.n_points)
            except Exception:
                total_current = 0

            # Count incoming points (post axis-map but pre downsampling)
            total_incoming = 0
            try:
                for _o in objects:
                    if _o.kind == "points" and _o.points is not None:
                        total_incoming += int(_o.points.shape[0])
            except Exception:
                pass

            total_after = max(1, total_current + total_incoming)

            # Heuristic caps similar to viewer3d (_target_visible_points)
            points_as_spheres = bool(getattr(self.viewer3d, "_points_as_spheres", False))
            if sys.platform == "darwin":
                cap = 600_000 if points_as_spheres else 2_000_000
            else:
                cap = 1_200_000 if points_as_spheres else 4_000_000

            suggested = min(100, max(1, int(cap * 100 / total_after)))

            # Update UI slider and viewer with suggested percent (user can override later)
            try:
                if hasattr(self.displayPanel, "spinBudget") and self.displayPanel.spinBudget is not None:
                    self.displayPanel.spinBudget.blockSignals(True)
                    self.displayPanel.spinBudget.setValue(suggested)
                    self.displayPanel.spinBudget.blockSignals(False)
            except Exception:
                pass
            try:
                getattr(self.viewer3d, "set_point_budget", lambda *_: None)(suggested)
            except Exception:
                pass
        except Exception:
            pass

        # 2) Apply point budget on import (from Display panel)
        try:
            budget = int(getattr(self.displayPanel, "spinBudget", None).value())
        except Exception:
            budget = 100

        if budget < 100:
            from .utils.io.importers import downsample_random, downsample_voxel_auto
            for obj in objects:
                if obj.kind == "points" and obj.points is not None and obj.points.shape[0] > 0:
                    n0 = obj.points.shape[0]
                    if self.downsample_method == "voxel":
                        idx = downsample_voxel_auto(obj.points, budget)
                    else:
                        idx = downsample_random(obj.points, budget)
                    obj.points = obj.points[idx]
                    if obj.colors is not None and obj.colors.shape[0] == n0:
                        obj.colors = obj.colors[idx]
                    if obj.intensity is not None and obj.intensity.shape[0] == n0:
                        obj.intensity = obj.intensity[idx]
                    if getattr(obj, 'normals', None) is not None and obj.normals.shape[0] == n0:
                        obj.normals = obj.normals[idx]
                    obj.meta["downsample"] = {"method": self.downsample_method, "percent": budget, "kept": int(obj.points.shape[0])}

        # 3) Add transformed objects to the viewer honoring color preference
        for obj in objects:
            # Temporarily adjust viewer color mode according to preference
            try:
                prev_mode = getattr(self.viewer3d, '_color_mode', None)
                pref = obj.meta.get('color_preference', 'rgb')
                if pref == 'colormap':
                    self.viewer3d.set_color_mode('Normal Colormap')
                else:
                    self.viewer3d.set_color_mode('Normal RGB')
            except Exception:
                prev_mode = None

            if obj.kind == "points" and obj.points is not None:
                ds_index = self.viewer3d.add_points(obj.points, obj.colors, getattr(obj, "normals", None))
            elif obj.kind == "mesh" and obj.pv_mesh is not None:
                self.viewer3d.add_pyvista_mesh(obj.pv_mesh)

            # Restore previous color mode
            try:
                if prev_mode is not None:
                    self.viewer3d.set_color_mode(prev_mode)
            except Exception:
                pass

            #
            # Tree: gerarchico, selezionabile, con metadati.
            # Tree: hierarchical, checkable, with metadata.
            self.treeMCTS.blockSignals(True)
            root = QtWidgets.QTreeWidgetItem([obj.name])
            root.setFlags(
                root.flags()
                | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                | QtCore.Qt.ItemFlag.ItemIsAutoTristate
            )
            root.setCheckState(0, QtCore.Qt.Checked)
            self.treeMCTS.addTopLevelItem(root)

            if obj.kind == "points":
                # Figlio nuvola di punti
                # Point cloud child
                it_points = QtWidgets.QTreeWidgetItem(["Point cloud"])
                it_points.setFlags(
                    it_points.flags()
                    | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                    | QtCore.Qt.ItemFlag.ItemIsAutoTristate
                )
                it_points.setCheckState(0, QtCore.Qt.Checked)
                it_points.setData(0, QtCore.Qt.UserRole, {"kind": "points", "ds": ds_index})
                root.addChild(it_points)

                # Figlio Normali (se presente)
                # Normals child (if available)
                if getattr(obj, "normals", None) is not None:
                    it_normals = QtWidgets.QTreeWidgetItem(["Normals"])
                    it_normals.setFlags(
                        it_normals.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                    )
                    it_normals.setCheckState(0, QtCore.Qt.Unchecked)
                    it_normals.setData(0, QtCore.Qt.UserRole, {"kind": "normals", "ds": ds_index})
                    it_points.addChild(it_normals)
            else:
                # Segnaposto Mesh (overlay futuri)
                # Mesh placeholder child (future overlays)
                it_mesh = QtWidgets.QTreeWidgetItem(["Mesh"])
                it_mesh.setFlags(it_mesh.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                it_mesh.setCheckState(0, QtCore.Qt.Checked)
                root.addChild(it_mesh)

            # Sblocca i segnali dopo la creazione del sottoalbero
            # Unblock signals after building the subtree
            self.treeMCTS.blockSignals(False)

            # Register into MCTS and set current MCT so it's visible in console
            try:
                entry = {
                    "name": obj.name,
                    "kind": obj.kind,
                    "has_rgb": obj.colors is not None,
                    "has_intensity": getattr(obj, 'intensity', None) is not None,
                    "has_normals": getattr(obj, 'normals', None) is not None,
                    "ds_index": ds_index if (obj.kind == 'points' and 'ds_index' in locals()) else None,
                }
                self.mcts[obj.name] = entry
                self.mct = entry
                # Select the newly added root item for clarity
                try:
                    self.treeMCTS.setCurrentItem(root)
                except Exception:
                    pass
            except Exception:
                pass

            self.statusBar().showMessage(f"Imported {len(objects)} object(s) from {path}", 5000)
            # Ricalcola la visibilità iniziale dopo l'importazione.
            # Recompute initial visibility after import.
            self._refresh_tree_visibility()
    def _update_parent_checkstate(self, child: QtWidgets.QTreeWidgetItem) -> None:
        """Aggiorna gli antenati in base ai figli (tri-stato manuale).
        Update ancestors according to children (manual tri-state)."""
        if child is None:
            return
        self._tree_updating = True
        try:
            parent = child.parent()
            while parent is not None:
                total = parent.childCount()
                if total == 0:
                    break
                checked = 0
                unchecked = 0
                for i in range(total):
                    st = parent.child(i).checkState(0)
                    if st == QtCore.Qt.Checked:
                        checked += 1
                    elif st == QtCore.Qt.Unchecked:
                        unchecked += 1
                if checked == total:
                    parent.setCheckState(0, QtCore.Qt.Checked)
                elif unchecked == total:
                    parent.setCheckState(0, QtCore.Qt.Unchecked)
                else:
                    parent.setCheckState(0, QtCore.Qt.PartiallyChecked)
                parent = parent.parent()
        finally:
            self._tree_updating = False

    def _set_descendant_checkstate(
        self, item: QtWidgets.QTreeWidgetItem, state: QtCore.Qt.CheckState
    ) -> None:
        """Imposta lo stato di tutti i discendenti.
        Set the check state of all descendants."""
        for i in range(item.childCount()):
            child = item.child(i)
            child.setCheckState(0, state)
            self._set_descendant_checkstate(child, state)

    def _is_effectively_checked(self, item: QtWidgets.QTreeWidgetItem) -> bool:
        """Verifica se l'elemento e i suoi antenati sono selezionati.
        Return True if the item and all its ancestors are checked."""
        while item is not None:
            if item.checkState(0) != QtCore.Qt.Checked:
                return False
            item = item.parent()
        return True

    def _refresh_tree_visibility(self) -> None:
        """Ricalcola la visibilità delle geometrie in base all'albero.
        Recompute geometry visibility based on the tree."""

        def recurse(node: QtWidgets.QTreeWidgetItem) -> None:
            data = node.data(0, QtCore.Qt.UserRole)
            if isinstance(data, dict):
                kind = data.get("kind")
                ds = data.get("ds")
                if ds is not None:
                    visible = self._is_effectively_checked(node)
                    if kind == "points":
                        getattr(self.viewer3d, "set_points_visibility", lambda *_: None)(ds, visible)
                    elif kind == "normals":
                        getattr(self.viewer3d, "set_normals_visibility", lambda *_: None)(ds, visible)
            for i in range(node.childCount()):
                recurse(node.child(i))

        for i in range(self.treeMCTS.topLevelItemCount()):
            recurse(self.treeMCTS.topLevelItem(i))

    def _on_tree_item_changed(self, item: QtWidgets.QTreeWidgetItem, col: int) -> None:
        # Evita la rientranza durante gli aggiornamenti programmati.
        # Avoid re-entrancy during programmatic updates.
        if getattr(self, "_tree_updating", False):
            return

        state = item.checkState(0)
        self._tree_updating = True
        try:
            self._set_descendant_checkstate(item, state)
            self._update_parent_checkstate(item)
        finally:
            self._tree_updating = False
        self._refresh_tree_visibility()

    def _on_tree_context_menu(self, pos: QtCore.QPoint) -> None:
        item = self.treeMCTS.itemAt(pos)
        if item is None:
            return
        data = item.data(0, QtCore.Qt.UserRole)
        if not isinstance(data, dict):
            return
        # Consentire la modifica del colore solo per i nodi della nuvola di punti.
        # Only allow color edit for point cloud nodes.
        if data.get("kind") != "points":
            return
        ds = data.get("ds")
        if ds is None:
            return
        menu = QtWidgets.QMenu(self)
        act_color = menu.addAction("Set Color…")
        act_random = menu.addAction("Random Color")
        chosen = menu.exec(self.treeMCTS.viewport().mapToGlobal(pos))
        if chosen is None:
            return
        if chosen is act_color:
            col = QtWidgets.QColorDialog.getColor(parent=self, title="Choose color for point cloud")
            if col.isValid():
                getattr(self.viewer3d, "set_dataset_color", lambda *_: None)(ds, col.red(), col.green(), col.blue())
        elif chosen is act_random:
            import random
            r, g, b = [random.randint(32, 224) for _ in range(3)]
            getattr(self.viewer3d, "set_dataset_color", lambda *_: None)(ds, r, g, b)

    # ------------------------------------------------------------------
    # Test / Esecutore di script
    # Testing / Script runner
    # ------------------------------------------------------------------
    def _on_run_script(self) -> None:
        """Open a .py file and execute it inside the console context.

        The script has access to: mcts, mct, window, undo_stack and a helper
        function `import_cloud(path, **kwargs)` (see `import_cloud_programmatic`).
        """
        dlg = QtWidgets.QFileDialog(self, "Select Python script to run")
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        dlg.setNameFilters(["Python scripts (*.py)", "All files (*)"])
        # Default to tests/ directory if it exists
        project_root = os.path.dirname(os.path.dirname(__file__))
        tests_dir = os.path.abspath(os.path.join(project_root, "tests"))
        if os.path.isdir(tests_dir):
            dlg.setDirectory(tests_dir)
        if not dlg.exec():
            return
        sel = dlg.selectedFiles()
        if not sel:
            return
        self._exec_script_file(sel[0])

    def _exec_script_file(self, path: str) -> None:
        """Execute a Python file in the same context used by the console.

        Args:
            path: path to a .py file
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                code = f.read()
        except Exception as ex:
            QtWidgets.QMessageBox.critical(self, "Script error", f"Cannot read script: {ex}")
            return

        # Build execution context
        ctx = dict(self._console_context())
        ctx.setdefault("window", self)
        ctx.setdefault("import_cloud", self.import_cloud_programmatic)
        try:
            import numpy as _np  # noqa: F401
            ctx.setdefault("np", _np)
        except Exception:
            pass
        try:
            import pyvista as _pv  # noqa: F401
            ctx.setdefault("pv", _pv)
        except Exception:
            pass
        ctx.setdefault("QtWidgets", QtWidgets)
        ctx.setdefault("QtCore", QtCore)
        ctx.setdefault("QtGui", QtGui)

        # Provide dunder variables for compatibility
        ctx["__file__"] = path
        ctx["__name__"] = "__main__"
        ctx["__package__"] = None

        # Allow relative imports and relative paths like in a normal script
        import sys
        script_dir = os.path.dirname(os.path.abspath(path))
        project_root = os.path.dirname(os.path.dirname(__file__))
        tests_dir = os.path.abspath(os.path.join(project_root, "tests"))
        old_cwd = os.getcwd()
        try:
            # Update sys.path
            for pth in (script_dir, tests_dir, project_root):
                if pth and pth not in sys.path:
                    sys.path.insert(0, pth)
            # Run with the script's folder as CWD
            os.chdir(script_dir)

            compiled = compile(code, path, "exec")
            exec(compiled, ctx, ctx)
            self.statusBar().showMessage(f"Executed script: {os.path.basename(path)}", 4000)
        except Exception as ex:
            QtWidgets.QMessageBox.critical(self, "Script execution error", str(ex))
        finally:
            # Restore working directory
            try:
                os.chdir(old_cwd)
            except Exception:
                pass

    def _on_run_tests_triplet(self) -> None:
        """Convenience: import the three example PLYs from tests/ directory."""
        project_root = os.path.dirname(os.path.dirname(__file__))
        tests_dir = os.path.abspath(os.path.join(project_root, "tests"))
        files = [
            "test_1_Corinthian_Column_Capital_RGB_no_normals.ply",
            "test_2_Rocca_North_tower_no_RGB.ply",
            "test_3_Turkish_pillar_RGB_normals.ply",
        ]
        missing = []
        for name in files:
            p = os.path.join(tests_dir, name)
            if os.path.isfile(p):
                try:
                    self.import_cloud_programmatic(p)
                except Exception as ex:
                    QtWidgets.QMessageBox.critical(self, "Import error", f"{name}: {ex}")
                    return
            else:
                missing.append(name)
        if missing:
            QtWidgets.QMessageBox.warning(self, "Missing files", "\n".join(["Not found in tests/:", *missing]))
        else:
            self.statusBar().showMessage("Triplet import completed", 4000)

    # ------------------------------------------------------------------
    # Programmatic import (no dialog)
    # ------------------------------------------------------------------
    def import_cloud_programmatic(
        self,
        path: str,
        *,
        axis_preset: str = "Z-up (identity)",
        color_preference: str = "auto",
        compute_normals_if_missing: bool = True,
        map_normals: bool = True,
    ) -> None:
        """Import a geometry file without showing the summary dialog.

        Mirrors `_on_import_cloud` pipeline with sensible defaults:
        - axis_preset: one of the presets in the summary dialog
        - color_preference: 'auto' | 'rgb' | 'colormap'
        - compute_normals_if_missing: compute rough normals if absent
        - map_normals: if True, apply axis preset also to normals
        """
        from .utils.io.importers import import_file

        # Presets consistent with ImportSummaryDialog
        presets = {
            "Z-up (identity)":  {"X": "+X", "Y": "+Y", "Z": "+Z"},
            "Y-up (swap Y/Z)":  {"X": "+X", "Y": "+Z", "Z": "-Y"},
            "X-up (swap X/Z)":  {"X": "+Z", "Y": "+Y", "Z": "-X"},
            "Flip Z":           {"X": "+X", "Y": "+Y", "Z": "-Z"},
            "Flip Y":           {"X": "+X", "Y": "-Y", "Z": "+Z"},
            "Flip X":           {"X": "-X", "Y": "+Y", "Z": "+Z"},
        }
        axis_map = presets.get(axis_preset, presets["Z-up (identity)"])

        try:
            objects = import_file(path)
        except Exception as ex:
            raise RuntimeError(f"Import error: {ex}")

        # Helpers (reuse lambdas from _on_import_cloud but kept local here)
        def _apply_axis_map(arr, axis_map):
            try:
                import numpy as _np
                if arr is None:
                    return None
                src = _np.asarray(arr, dtype=float)
                if src.ndim != 2 or src.shape[1] != 3:
                    return arr
                out = _np.empty_like(src)
                axes = {"X": 0, "Y": 1, "Z": 2}
                for tgt_key, expr in axis_map.items():
                    sign = -1.0 if expr.startswith("-") else 1.0
                    src_axis = axes[expr[-1]]
                    out[:, axes[tgt_key]] = sign * src[:, src_axis]
                return out
            except Exception:
                return arr

        def _compute_normals(obj):
            try:
                import numpy as _np
                import pyvista as _pv  # type: ignore
                if obj.points is None or obj.points.shape[0] == 0:
                    return None
                pdata = _pv.PolyData(_np.asarray(obj.points))
                pdata = pdata.compute_normals(
                    consistent=False,
                    auto_orient_normals=False,
                    feature_angle=180.0,
                )
                n = getattr(pdata, 'point_normals', None)
                if n is not None:
                    return _np.asarray(n, dtype=_np.float32)
            except Exception:
                return None
            return None

        # 1) Apply axis mapping / normals ops per object
        for obj in objects:
            obj.points = _apply_axis_map(obj.points, axis_map)
            if map_normals and getattr(obj, 'normals', None) is not None:
                obj.normals = _apply_axis_map(obj.normals, axis_map)
            elif compute_normals_if_missing and getattr(obj, 'normals', None) is None:
                n = _compute_normals(obj)
                if n is not None:
                    obj.normals = n
            # Color preference per object (auto: prefer RGB if available)
            if color_preference == "auto":
                obj.meta['color_preference'] = 'rgb' if obj.colors is not None else 'colormap'
            else:
                obj.meta['color_preference'] = color_preference

        # 2) Reuse the final part of the GUI pipeline to add to viewer & tree
        # Temporarily set viewer color-mode per object, as in _on_import_cloud
        for obj in objects:
            prev_mode = getattr(self.viewer3d, '_color_mode', None)
            try:
                pref = obj.meta.get('color_preference', 'rgb')
                if pref == 'colormap':
                    self.viewer3d.set_color_mode('Normal Colormap')
                else:
                    self.viewer3d.set_color_mode('Normal RGB')
            except Exception:
                prev_mode = None

            if obj.kind == "points" and obj.points is not None:
                ds_index = self.viewer3d.add_points(obj.points, obj.colors, getattr(obj, "normals", None))
            elif obj.kind == "mesh" and obj.pv_mesh is not None:
                ds_index = None
                self.viewer3d.add_pyvista_mesh(obj.pv_mesh)
            else:
                ds_index = None

            try:
                if prev_mode is not None:
                    self.viewer3d.set_color_mode(prev_mode)
            except Exception:
                pass

            # Costruisci le voci dell'albero (come nell'import interattivo)
            # Build the tree entries (same as interactive import)
            self.treeMCTS.blockSignals(True)
            root = QtWidgets.QTreeWidgetItem([obj.name])
            root.setFlags(
                root.flags()
                | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                | QtCore.Qt.ItemFlag.ItemIsAutoTristate
            )
            root.setCheckState(0, QtCore.Qt.Checked)
            self.treeMCTS.addTopLevelItem(root)

            if obj.kind == "points":
                it_points = QtWidgets.QTreeWidgetItem(["Point cloud"])
                it_points.setFlags(
                    it_points.flags()
                    | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                    | QtCore.Qt.ItemFlag.ItemIsAutoTristate
                )
                it_points.setCheckState(0, QtCore.Qt.Checked)
                it_points.setData(0, QtCore.Qt.UserRole, {"kind": "points", "ds": ds_index})
                root.addChild(it_points)

                if getattr(obj, "normals", None) is not None:
                    it_normals = QtWidgets.QTreeWidgetItem(["Normals"])
                    it_normals.setFlags(
                        it_normals.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                    )
                    it_normals.setCheckState(0, QtCore.Qt.Unchecked)
                    it_normals.setData(0, QtCore.Qt.UserRole, {"kind": "normals", "ds": ds_index})
                    it_points.addChild(it_normals)
            else:
                it_mesh = QtWidgets.QTreeWidgetItem(["Mesh"])
                it_mesh.setFlags(it_mesh.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                it_mesh.setCheckState(0, QtCore.Qt.Checked)
                root.addChild(it_mesh)

            self.treeMCTS.blockSignals(False)
            # Aggiorna la visibilità dopo l'aggiunta.
            # Update visibility after adding.
            self._refresh_tree_visibility()

            # Register into MCTS and set current MCT so it's visible in console
            try:
                entry = {
                    "name": obj.name,
                    "kind": obj.kind,
                    "has_rgb": obj.colors is not None,
                    "has_intensity": getattr(obj, 'intensity', None) is not None,
                    "has_normals": getattr(obj, 'normals', None) is not None,
                    "ds_index": ds_index if (obj.kind == 'points' and ds_index is not None) else None,
                }
                self.mcts[obj.name] = entry
                self.mct = entry
                try:
                    self.treeMCTS.setCurrentItem(root)
                except Exception:
                    pass
            except Exception:
                pass

        self.statusBar().showMessage(f"Imported from {os.path.basename(path)}", 3000)