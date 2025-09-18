from __future__ import annotations
import os
from typing import Optional
from PySide6 import QtCore, QtGui, QtWidgets

import json

from c2f4dt.utils.theme import apply_user_theme
from c2f4dt.utils.systeminfo import disk_usage_percent
from c2f4dt.utils.icons import qicon
from c2f4dt.plugins.manager import PluginManager
from c2f4dt.plugins.cloud2fem import Cloud2FEMPlugin

from c2f4dt.ui.console import ConsoleWidget
from c2f4dt.ui.viewer3d import Viewer3DPlaceholder
from c2f4dt.ui.display_panel import DisplayPanel

try:
    from .ui.viewer3d import Viewer3D as _Viewer3D
    _HAS_PYVISTA = True
except Exception:
    _HAS_PYVISTA = False
    _Viewer3D = Viewer3DPlaceholder

class _NormalsWorker(QtCore.QObject):
    """Background worker to compute point-cloud normals.

    Signals:
        progress(int): Progress percentage [0..100].
        message(str): Human-friendly status.
        finished(object): Returns Nx3 normals (float64) or None on failure/cancel.

    The algorithm uses a PCA on k-NN neighborhoods. If ``fast`` is True and the
    point count is larger than ``fast_max_points``, it computes normals on a
    random subset and propagates them to the full set via nearest neighbors if
    SciPy is available; otherwise it falls back to full PCA.
    """

    progress = QtCore.Signal(int)
    message = QtCore.Signal(str)
    finished = QtCore.Signal(object)

    def __init__(self, points, k=16, subset_size=80000, fast=True, fast_max_points=250000):
        super().__init__()
        self._P = points
        self._k = int(max(3, k))
        self._subset = int(max(1000, subset_size))
        self._fast = bool(fast)
        self._fast_max = int(max(10000, fast_max_points))
        self._cancel = False

    def request_cancel(self):
        self._cancel = True

    def _pca_normals(self, P, k):
        import numpy as np
        from numpy.linalg import eigh
        # Prefer SciPy KDTree for stable kNN on macOS
        idx = None
        try:
            from scipy.spatial import cKDTree  # type: ignore
            tree = cKDTree(P)
            # query k neighbors for each point (include point itself)
            _, idx = tree.query(P, k=min(k, P.shape[0]))
            if idx.ndim == 1:  # when k==1 returns shape (N,)
                idx = idx[:, None]
        except Exception:
            # Fallback: use a partial argpartition per-point (O(N log N) approx)
            n = P.shape[0]
            idx = np.empty((n, k), dtype=int)
            # Precompute squared norms for speed
            for i in range(n):
                # distances to all points
                d2 = np.sum((P - P[i])**2, axis=1)
                # get k smallest indices (including i)
                sel = np.argpartition(d2, kth=min(k, n-1))[:k]
                idx[i, :sel.shape[0]] = sel
                if sel.shape[0] < k:
                    # wrap-fill if needed
                    wrap = np.resize(sel, k)
                    idx[i] = wrap
        N = np.empty_like(P)
        # compute normals via smallest eigenvector of covariance
        for i in range(P.shape[0]):
            if self._cancel:
                return None
            nbrs = P[idx[i]]
            C = np.cov(nbrs.T)
            w, v = eigh(C)
            n = v[:, 0]  # eigenvector of smallest eigenvalue
            # orient consistently (optional): point roughly outward from centroid
            if np.dot(n, P[i] - nbrs.mean(axis=0)) < 0:
                n = -n
            N[i] = n
            if i % 1000 == 0:
                self.progress.emit(int(5 + 90 * i / max(1, P.shape[0])))
        return N

    @QtCore.Slot()
    def run(self):
        import numpy as np
        try:
            P = np.asarray(self._P, dtype=float)
            n = int(P.shape[0])
            if n == 0 or P.shape[1] != 3:
                self.message.emit("Invalid point cloud array")
                self.finished.emit(None)
                return
            self.message.emit(f"Computing normals (N={n}, k={self._k})…")
            self.progress.emit(2)

            if self._fast and n > self._fast_max:
                self.message.emit("FAST mode: subset + propagate…")
                # subset
                rng = np.random.default_rng(42)
                pick = rng.choice(n, size=min(self._subset, n), replace=False)
                Psub = P[pick]
                # compute on subset
                self.progress.emit(4)
                Nsub = self._pca_normals(Psub, self._k)
                if Nsub is None:
                    self.finished.emit(None)
                    return
                # propagate using nearest neighbors if SciPy available
                try:
                    from scipy.spatial import cKDTree  # type: ignore
                    tree = cKDTree(Psub)
                    _, j = tree.query(P, k=1)
                    N = Nsub[j]
                except Exception:
                    # As a last resort, avoid quadratic cost; compute full normals instead
                    self.message.emit("Propagation unavailable; switching to full PCA…")
                    N = self._pca_normals(P, self._k)
                    if N is None:
                        self.finished.emit(None)
                        return
            else:
                N = self._pca_normals(P, self._k)
                if N is None:
                    self.finished.emit(None)
                    return

            self.progress.emit(100)
            self.message.emit("Normals ready")
            self.finished.emit(N)
        except Exception as ex:
            self.message.emit(f"Normals failed: {ex}")
            self.finished.emit(None)


class MainWindow(QtWidgets.QMainWindow):
    """
    Main GUI window for C2F4DT.

    This class manages the main application window, including the menu, toolbars, status bar,
    central widgets, plugin management, and the 3D/2D viewers. It also handles the import and
    visualization of point clouds and meshes, as well as user interactions with the dataset tree.
    """

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

        self.plugin_manager = PluginManager(self, plugins_dir=self._default_plugins_dir())
        
        self._cloud2fem = Cloud2FEMPlugin(self)

        self._populate_plugins_ui()

        self._start_disk_timer()

        # Debouncer to rebuild the 3D scene once after bursts of changes
        self._rebuild_timer = QtCore.QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.timeout.connect(self._reset_viewer3d_from_tree)

        self.undo_stack.cleanChanged.connect(self._on_undo_changed)
        self.undo_stack.indexChanged.connect(self._on_undo_changed)
        self._on_undo_changed()

        # Preferences
        self.downsample_method = "random"  # or "voxel"
        self._session_path: Optional[str] = None

        # Tree update guard to avoid cascading on auto-updates/partial states
        # Explicitly ensure the flag exists and starts false
        self._tree_updating = False

        self._mount_cloud2fem_panel()
        
    def _iter_children(self, item: QtWidgets.QTreeWidgetItem):
        """Yield direct children of *item* safely."""
        for i in range(item.childCount()):
            yield item.child(i)

    def _set_item_checked(self, item: QtWidgets.QTreeWidgetItem, on: bool) -> None:
        """Set the checkbox state of *item* without recursive signal storms."""
        try:
            self._tree_updating = True
            item.setCheckState(0, QtCore.Qt.CheckState.Checked if on else QtCore.Qt.CheckState.Unchecked)
        finally:
            self._tree_updating = False

    @QtCore.Slot(QtWidgets.QTreeWidgetItem, int)
    # def _on_tree_item_changed(self, item: QtWidgets.QTreeWidgetItem, column: int) -> None:
    #     """
    #     Handle checkbox toggles in treeMCTS.

    #     Behavior:
    #     - If 'Tree ➜ Parent check toggles children' is ON, propagate the state to children.
    #     - Toggle dataset visibility for nodes that carry {'kind': 'points'|'mesh'|'normals', 'ds': int}.
    #     - Ensure actors are (re)created when turning visibility ON.
    #     - Keep self.mcts and viewer._datasets['visible'] in sync.
    #     """
    #     # Guard against programmatic changes
    #     if getattr(self, "_tree_updating", False):
    #         return
    #     try:
    #         role = QtCore.Qt.ItemDataRole.UserRole
    #         data = item.data(0, role)
    #         checked = item.checkState(0) == QtCore.Qt.CheckState.Checked
    #         propagate = False
    #         try:
    #             propagate = bool(self.act_tree_propagate.isChecked())
    #         except Exception:
    #             propagate = False

    #         # 1) Propagate to children if requested
    #         if propagate:
    #             try:
    #                 self._tree_updating = True
    #                 for ch in self._iter_children(item):
    #                     ch.setCheckState(0, QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked)
    #             finally:
    #                 self._tree_updating = False

    #         # 2) If this node maps to a dataset, toggle visibility accordingly
    #         if isinstance(data, dict):
    #             kind = data.get("kind")
    #             ds = data.get("ds")
    #             if isinstance(ds, int) and kind in ("points", "mesh", "normals"):
    #                 if kind in ("points", "mesh"):
    #                     # Main dataset visibility
    #                     self._viewer_set_visibility(kind, ds, bool(checked))
    #                     # Persist visible flag into cache + session dicts
    #                     try:
    #                         self._persist_dataset_prop(ds, "visible", bool(checked))
    #                     except Exception:
    #                         pass
    #                     # If we just turned OFF points, force normals OFF for that ds
    #                     if kind == "points" and not checked:
    #                         try:
    #                             self._viewer_set_visibility("normals", ds, False)
    #                         except Exception:
    #                             pass
    #                 elif kind == "normals":
    #                     # Normals visibility only if dataset exists
    #                     try:
    #                         getattr(self.viewer3d, "set_normals_visibility", lambda *_: None)(ds, bool(checked))
    #                     except Exception:
    #                         pass
    #                     try:
    #                         self._persist_dataset_prop(ds, "normals_visible", bool(checked))
    #                     except Exception:
    #                         pass

    #         # 3) If a parent WITHOUT explicit kind was toggled, and propagate=False,
    #         #    try to reflect the parent checkbox on immediate children that have ds/kind.
    #         if (not propagate) and (not isinstance(data, dict)):
    #             for ch in self._iter_children(item):
    #                 dch = ch.data(0, role)
    #                 if isinstance(dch, dict):
    #                     kind = dch.get("kind")
    #                     ds = dch.get("ds")
    #                     if isinstance(ds, int) and kind in ("points", "mesh", "normals"):
    #                         # Do not change checkbox UI; only re-sync visibility with current child state
    #                         on = ch.checkState(0) == QtCore.Qt.CheckState.Checked
    #                         if kind in ("points", "mesh"):
    #                             self._viewer_set_visibility(kind, ds, bool(on))
    #                         elif kind == "normals":
    #                             try:
    #                                 getattr(self.viewer3d, "set_normals_visibility", lambda *_: None)(ds, bool(on))
    #                             except Exception:
    #                                 pass
    #                         try:
    #                             if kind == "normals":
    #                                 self._persist_dataset_prop(ds, "normals_visible", bool(on))
    #                             else:
    #                                 self._persist_dataset_prop(ds, "visible", bool(on))
    #                         except Exception:
    #                             pass

    #         # 4) Best-effort overlays refresh and viewer refresh
    #         try:
    #             self._reapply_overlays_safe()
    #         except Exception:
    #             pass
    #         try:
    #             self.viewer3d.refresh()
    #         except Exception:
    #             pass
    #     except Exception:
    #         # Avoid breaking the UI due to unexpected data
    #         pass

    #     # 3D view preferences (dataclass-like dict, for settings dialog)
    #     self._view_prefs = {
    #         "bg": (82, 87, 110),
    #         "grid": True,
    #         "points_as_spheres": False,
    #         "colorbar_mode": "vertical-tr",
    #         "colorbar_title": "",
    #     }

    def _schedule_scene_rebuild(self, delay_ms: int = 60) -> None:
        """Schedule a single full-scene rebuild after a short delay (debounced)."""
        try:
            self._rebuild_timer.start(int(max(1, delay_ms)))
        except Exception:
            # Fallback: rebuild immediately
            self._reset_viewer3d_from_tree()
            
    def _apply_styles(self) -> None:
            """
            Apply custom CSS styles to the main window widgets for a consistent look and feel.
            """
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

    def _default_plugins_dir(self) -> str:
        base = os.path.dirname(os.path.dirname(__file__))  # .../C2F4DT/c2f4dt
        return os.path.join(base, "c2f4dt/plugins")

    def _build_actions(self) -> None:
        self.act_new = QtGui.QAction(qicon("32x32_document-new.png"), "New", self)
        self.act_new.setShortcut(QtGui.QKeySequence.New)
        self.act_open = QtGui.QAction(qicon("32x32_document-open.png"), "Open…", self)
        self.act_open.setShortcut(QtGui.QKeySequence.Open)
        self.act_save = QtGui.QAction(qicon("32x32_document-save.png"), "Save", self)
        self.act_save.setShortcut(QtGui.QKeySequence.Save)
        self.act_save_as = QtGui.QAction(qicon("32x32_document-save-as.png"), "Save As…", self)
        self.act_save_as.setShortcut(QtGui.QKeySequence.SaveAs)
        self.act_new.triggered.connect(self._on_new_session)
        self.act_open.triggered.connect(self._on_open_session)
        self.act_save.triggered.connect(self._on_save_session)
        self.act_save_as.triggered.connect(self._on_save_session_as)
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
        # Add a separator and a new action for 3D View Settings dialog
        self.act_view_settings = QtGui.QAction("3D View Settings…", self)
        self.act_view_settings.setIcon(qicon("32x32_settings.png")) if qicon else None
        self.act_view_settings.triggered.connect(self._on_open_view_settings)
        m_view.addSeparator()
        m_view.addAction(self.act_view_settings)
        m_tools = menubar.addMenu("&Tools")
        for a in [self.act_create_grid, self.act_toggle_grid, self.act_toggle_normals]:
            m_tools.addAction(a)
        
        # --- Tree behaviour submenu -----------------------------------
        m_tree = m_tools.addMenu("Tree")
        self.act_tree_propagate = QtGui.QAction("Parent check toggles children", self)
        self.act_tree_propagate.setCheckable(True)
        self.act_tree_propagate.setChecked(False)  # default: come ora (propaga ai figli)
        m_tree.addAction(self.act_tree_propagate)


        # Activate Plugins menu
        self.m_plugins = menubar.addMenu("&Plugins")
        self.m_plugins.setToolTipsVisible(True)
        self.m_plugins_about_to_show = False
        

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
        for a in [self.act_create_grid,  self.act_toggle_normals]:
            self.top_toolbar.addAction(a)

        self.left_toolbar = QtWidgets.QToolBar("barVERTICALCOMMAND_left", self)
        self.left_toolbar.setIconSize(QtCore.QSize(32, 32))
        self.left_toolbar.setOrientation(QtCore.Qt.Vertical)
        self.addToolBar(QtCore.Qt.LeftToolBarArea, self.left_toolbar)
        # for a in [self.act_fit, self.act_refresh, self.act_xp, self.act_xm, self.act_yp, self.act_ym, self.act_zp, self.act_zm, self.act_iso_p, self.act_iso_m, self.act_invert, self.act_toggle_grid,]:
        #     self.left_toolbar.addAction(a)
        for a in [self.act_fit, self.act_refresh, self.act_xp, self.act_yp, self.act_zp,  self.act_iso_p, self.act_iso_m, self.act_invert, self.act_toggle_grid,]:
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
        #
        self.act_toggle_normals.toggled.connect(self._on_toggle_normals_clicked)

        self.act_toggle_grid.toggled.connect(
            lambda on: (
                getattr(self.viewer3d, "set_grid_enabled",
                        getattr(self.viewer3d, "set_grid_visible",
                                getattr(self.viewer3d, "toggle_grid", lambda *_: None)))(on),
                getattr(self.viewer3d, "reapply_overlays",
                        getattr(self.viewer3d, "_apply_overlays", lambda: None))()
            )
        )

    def _mount_cloud2fem_panel(self):
        """Mount Cloud2FEM panel into right toolbar."""
        try:
            hooks = HostHooks(
                window=self,
                viewer3d=self.viewer3d,
                log=lambda lvl, msg: self.txtMessages.appendPlainText(f"[{lvl}] {msg}"),
                progress_begin=lambda title: self._progress_start(title),
                progress_update=lambda p, m: self._import_progress_update(percent=p, message=m),
                progress_end=lambda: self._progress_finish(),
                add_badge=lambda name, txt: self.statusBar().showMessage(f"{name}: {txt}", 2000),
            )
            self._cloud2fem.mount(hooks)
        except Exception:
            pass

    

    def _build_central_area(self) -> None:
        central = QtWidgets.QWidget(self)
        central_layout = QtWidgets.QVBoxLayout(central)
        central_layout.setContentsMargins(4, 4, 4, 4); central_layout.setSpacing(6)

        mid_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal, central)

        self.tabINTERACTION = QtWidgets.QTabWidget(mid_split)
        self.tabINTERACTION.setObjectName("tabINTERACTION")
        self.tabINTERACTION.setMinimumWidth(334)  # Set minimum width to 320 for better alignment with scrollDISPLAY

        self.tabDISPLAY = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(self.tabDISPLAY); v.setContentsMargins(4, 4, 4, 4)
        split = QtWidgets.QSplitter(QtCore.Qt.Vertical, self.tabDISPLAY)

        self.treeMCTS = QtWidgets.QTreeWidget()
        self.treeMCTS.setHeaderLabels(["Object"]); self.treeMCTS.setColumnCount(1)
        self.treeMCTS.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.treeMCTS.customContextMenuRequested.connect(self._on_tree_context_menu)
        self.treeMCTS.itemSelectionChanged.connect(self._on_tree_selection_changed)
        split.addWidget(self.treeMCTS)


        self.scrollDISPLAY = QtWidgets.QScrollArea(); self.scrollDISPLAY.setWidgetResizable(True)
        self.displayPanel = DisplayPanel(); self.scrollDISPLAY.setWidget(self.displayPanel)
        # --- Wire normals UI from DisplayPanel ---
        dp = self.displayPanel
        dp.sigNormalsStyleChanged.connect(self._on_normals_style_changed)
        dp.sigNormalsColorChanged.connect(self._on_normals_color_changed)
        dp.sigNormalsPercentChanged.connect(self._on_normals_percent_changed)
        dp.sigNormalsScaleChanged.connect(self._on_normals_scale_changed)
        dp.sigComputeNormals.connect(self._on_compute_normals)       # già esistente, riusa il tuo handler
        dp.sigFastNormalsChanged.connect(self._on_fast_normals_toggled)  # opzionale: salva preferenza FAST
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

        # Top bar with a Refresh button for the Inspector
        bar_ins = QtWidgets.QHBoxLayout()
        bar_ins.addStretch(1)
        self.btnRefreshInspector = QtWidgets.QPushButton("Refresh")
        self.btnRefreshInspector.setObjectName("btnRefreshInspector")
        bar_ins.addWidget(self.btnRefreshInspector)
        v4.addLayout(bar_ins)

        # Tree that shows the current MCT content
        self.treeMCT = QtWidgets.QTreeWidget()
        self.treeMCT.setObjectName("treeMCT")
        self.treeMCT.setHeaderLabels(["Key", "Value"])
        self.treeMCT.setColumnCount(2)
        self.treeMCT.header().setStretchLastSection(True)
        v4.addWidget(self.treeMCT, 1)

        # Hook up refresh
        self.btnRefreshInspector.clicked.connect(self._refresh_inspector_tree)

        self.tabINTERACTION.addTab(self.tabINSPECTOR, "INSPECTOR")

        viewer_container = QtWidgets.QWidget()
        viewer_layout = QtWidgets.QVBoxLayout(viewer_container)
        viewer_layout.setContentsMargins(4, 4, 4, 4); viewer_layout.setSpacing(4)

        bar_plugin = QtWidgets.QHBoxLayout()
        self.comboPlugins = QtWidgets.QComboBox()
        self.comboPlugins.addItem("— No plugins installed —"); self.comboPlugins.setEnabled(False)
        bar_plugin.addWidget(QtWidgets.QLabel("Plugin scope:"))
        bar_plugin.addWidget(self.comboPlugins, 1)
        self.comboPlugins.activated.connect(self._on_plugin_combo_activated)
        
        viewer_layout.addLayout(bar_plugin)

        self.viewer3d = _Viewer3D()
        viewer_layout.addWidget(self.viewer3d, 1)
        # React to check/uncheck from the MCTS tree (avoid duplicates)
        self.treeMCTS.itemChanged.connect(
            self._on_tree_item_changed,
            QtCore.Qt.ConnectionType.UniqueConnection
        )
        # Ora che viewer3d esiste, aggiorna la visibilità
        self._refresh_tree_visibility()

        mid_split.addWidget(self.tabINTERACTION)
        mid_split.addWidget(viewer_container)
        mid_split.setStretchFactor(0, 0); mid_split.setStretchFactor(1, 1)

        # central_layout.addWidget(mid_split, 1)   # REMOVE this line, replaced by splitter below

        self.tabCONSOLE_AND_MESSAGES = QtWidgets.QTabWidget()
        self.tabMESSAGES = QtWidgets.QWidget()
        vm = QtWidgets.QVBoxLayout(self.tabMESSAGES)
        self.txtMessages = QtWidgets.QPlainTextEdit(); self.txtMessages.setReadOnly(True)
        vm.addWidget(self.txtMessages)
        self.tabCONSOLE_AND_MESSAGES.addTab(self.tabMESSAGES, "Messages")

        self.console = ConsoleWidget(context_provider=self._console_context)
        self.console.sigExecuted.connect(self._on_console_executed)
        self.tabCONSOLE_AND_MESSAGES.addTab(self.console, "Console")

        # central_layout.addWidget(self.tabCONSOLE_AND_MESSAGES, 0)   # REMOVE this line, replaced by splitter below

        # --- Make the upper (interaction+viewer) and lower (messages/console) panes vertically resizable ---
        main_split = QtWidgets.QSplitter(QtCore.Qt.Vertical, central)
        main_split.setObjectName("splitMAIN_VERTICAL")
        main_split.setChildrenCollapsible(False)
        # put the big middle UI (tabs + viewer) on top, and the messages/console tabs below
        main_split.addWidget(mid_split)
        main_split.addWidget(self.tabCONSOLE_AND_MESSAGES)
        # sizing hints: top takes most of the space, bottom a fixed minimum height
        self.tabCONSOLE_AND_MESSAGES.setMinimumHeight(140)  # tweak if you want a different minimum
        main_split.setStretchFactor(0, 1)
        main_split.setStretchFactor(1, 0)
        try:
            # give an initial 80/20 split (best-effort; works after first show)
            main_split.setSizes([self.height() * 4 // 5, self.height() * 1 // 5])
        except Exception:
            pass

        # Add the vertical splitter to the central layout (instead of adding the two widgets separately)
        central_layout.addWidget(main_split, 1)

        self.setCentralWidget(central)

        # Collega il DisplayPanel agli handler specifici.
        # Wire the DisplayPanel to specific handlers.
        self.displayPanel.sigPointSizeChanged.connect(self._on_point_size_changed)
        self.displayPanel.sigPointBudgetChanged.connect(self._on_point_budget_changed)
        self.displayPanel.sigColorModeChanged.connect(self._on_color_mode_changed)
        self.displayPanel.sigSolidColorChanged.connect(self._on_solid_color_changed)
        self.displayPanel.sigColormapChanged.connect(self._on_colormap_changed)
        self.displayPanel.sigMeshRepresentationChanged.connect(self._on_mesh_rep_changed)
        self.displayPanel.sigMeshOpacityChanged.connect(self._on_mesh_opacity_changed)
        # --- Normals visualization ---
        self.displayPanel.sigNormalsStyleChanged.connect(self._on_normals_style_changed)
        self.displayPanel.sigNormalsColorChanged.connect(self._on_normals_color_changed)
        self.displayPanel.sigNormalsPercentChanged.connect(self._on_normals_percent_changed)
        self.displayPanel.sigNormalsScaleChanged.connect(self._on_normals_scale_changed)
        #

    def _on_console_executed(self, cmd: str) -> None:
        """Append executed console command to the MESSAGES panel."""
        try:
            self.txtMessages.appendPlainText(cmd)
        except Exception:
            pass

    def _on_tree_selection_changed(self) -> None:
        """
        Synchronize the `mct` dictionary with the currently selected item in the treeMCTS widget.
        Updates the display panel with the parameters of the selected dataset (point size, budget, color mode, etc).

        This ensures that when a user selects a node in the tree, the display panel reflects the properties
        of the corresponding dataset, allowing for correct editing and visualization.
        """
        item = self.treeMCTS.currentItem()
        if item is None:
            return
        # Find the root (file node) and its name
        root = item
        while root.parent() is not None:
            root = root.parent()
        name = root.text(0)
        entry = self.mcts.get(name)
        if not entry:
            return
        # Find dataset info and ds_index
        info = self._dataset_info_from_item(item)
        ds_index = info.get("ds") if info else None
        #
        # ... dopo aver determinato ds_index / entry_to_use ...
        try:
            ds = self._current_dataset_index()
            if ds is not None:
                recs = getattr(self.viewer3d, "_datasets", [])
                if 0 <= ds < len(recs):
                    nvis = bool(recs[ds].get("normals_visible", False))
                    self.act_toggle_normals.blockSignals(True)
                    self.act_toggle_normals.setChecked(nvis)
                    self.act_toggle_normals.blockSignals(False)
        except Exception:
            pass
        # Select the correct entry if it exists for ds_index
        entry_to_use = entry
        if ds_index is not None:
            for e in self.mcts.values():
                if e.get("ds_index") == ds_index:
                    entry_to_use = e
                    break
        self.mct = entry_to_use
        # Update the display panel with the selected dataset's parameters
        if info:
            self.displayPanel.set_mode(info.get("kind", "points"))
            self.displayPanel.apply_properties(entry_to_use)
        # Keep the INSPECTOR tab in sync with the current MCT
        try:
            self._refresh_inspector_tree()
        except Exception:
            pass


    def _refresh_inspector_tree(self) -> None:
        """Rebuild the Inspector tree from a synthesized snapshot of the current state.
        Includes: current mct entry, viewer settings, per-dataset details and app options.
        """
        try:
            data = self._inspector_current_payload()
            self._populate_inspector_tree(data)
        except Exception:
            # Best effort: clear on failure
            try:
                self.treeMCT.clear()
            except Exception:
                pass

    def _inspector_current_payload(self) -> dict:
        """Collect a rich snapshot of the current session for the INSPECTOR tab.

        Structure:
            {
                'mct': ...,                    # currently selected entry (as-is)
                'options': { ... },            # app/UI options affecting behavior
                'plugins': [ ... ],            # plugins summary from PluginManager
                'viewer': { ... },             # global viewer settings
                'dataset': { ... },            # details for the currently selected dataset
            }
        """
        payload: dict = {}

        # --- mct (as-is) -----------------------------------------------------
        try:
            payload['mct'] = self.mct
        except Exception:
            payload['mct'] = None

        # --- options (app-wide knobs) ----------------------------------------
        opts = {}
        try:
            opts['downsample_method'] = getattr(self, 'downsample_method', None)
        except Exception:
            pass
        # Normals controls (from DisplayPanel if available)
        try:
            fast = None
            if hasattr(self, 'displayPanel') and self.displayPanel is not None:
                # `fast_normals_enabled()` is our helper; fallback to attr
                try:
                    fast = bool(self.displayPanel.fast_normals_enabled())
                except Exception:
                    fast = None
            if fast is None:
                fast = bool(getattr(self, 'normals_fast_enabled', False))
            opts['normals_fast_enabled'] = fast
        except Exception:
            pass
        for k, default in (('normals_k', 16), ('normals_fast_max_points', 250_000)):
            try:
                opts[k] = getattr(self, k)
            except Exception:
                opts[k] = default
        payload['options'] = opts

        # --- plugins summary --------------------------------------------------
        plugs = []
        try:
            items = self.plugin_manager.ui_combo_items()
            for it in items:
                plugs.append({
                    'key': it.get('key'),
                    'label': it.get('label'),
                    'enabled': it.get('enabled', True),
                    'color': it.get('color'),
                    'tooltip': it.get('tooltip', ''),
                    'order': it.get('order'),
                })
        except Exception:
            pass
        payload['plugins'] = plugs

        # --- viewer global settings ------------------------------------------
        viewer = {}
        try:
            v = self.viewer3d
            viewer['color_mode'] = getattr(v, '_color_mode', None)
            viewer['colormap'] = getattr(v, '_cmap', None)
            viewer['point_size'] = getattr(v, '_point_size', None)
            viewer['view_budget_percent'] = getattr(v, '_view_budget_percent', None)
            viewer['points_as_spheres'] = getattr(v, '_points_as_spheres', None)
            # Safe rendering toggle (if exposed via menu action)
            try:
                viewer['safe_rendering'] = bool(self.act_safe_render.isChecked())
            except Exception:
                viewer['safe_rendering'] = None
            # Counts
            try:
                recs = getattr(v, '_datasets', [])
                viewer['datasets_count'] = len(recs)
            except Exception:
                viewer['datasets_count'] = None
        except Exception:
            pass
        payload['viewer'] = viewer

        # --- current dataset details -----------------------------------------
        ds_info = {}
        try:
            ds = self._current_dataset_index()
            ds_info['index'] = ds
            v = self.viewer3d
            recs = getattr(v, '_datasets', [])
            if isinstance(ds, int) and 0 <= ds < len(recs):
                rec = recs[ds]
                # Basic flags
                ds_info['visible'] = bool(rec.get('visible', True))
                ds_info['kind'] = rec.get('kind', 'points')
                ds_info['solid_color'] = tuple(rec.get('solid_color', (1.0, 1.0, 1.0)))
                # PolyData summary
                try:
                    pdata = rec.get('pdata')
                    ds_info['n_points'] = int(getattr(pdata, 'n_points', 0)) if pdata is not None else None
                    ds_info['n_cells'] = int(getattr(pdata, 'n_cells', 0)) if pdata is not None else None
                    # Available arrays
                    pt_names = []
                    try:
                        if hasattr(pdata, 'point_data'):
                            pt_names = list(pdata.point_data.keys())
                    except Exception:
                        pass
                    ds_info['point_arrays'] = pt_names
                except Exception:
                    pass
                # Normals section (per-dataset state kept by viewer)
                norms = {
                    'has_normals_array': False,
                    'normals_visible': bool(rec.get('normals_visible', False)),
                    'normals_style': rec.get('normals_style'),
                    'normals_color': tuple(rec.get('normals_color', (1.0, 0.8, 0.2))),
                    'normals_percent': int(rec.get('normals_percent', getattr(v, '_normals_percent', 1))),
                    'normals_scale': int(rec.get('normals_scale', getattr(v, '_normals_scale', 20))),
                    'actor_exists': rec.get('actor_normals') is not None,
                }
                try:
                    pdata = rec.get('pdata')
                    norms['has_normals_array'] = bool(pdata is not None and ('Normals' in getattr(pdata, 'point_data', {})))
                except Exception:
                    pass
                ds_info['normals'] = norms
            else:
                ds_info['note'] = 'No valid dataset selected.'
        except Exception:
            pass
        payload['dataset'] = ds_info

        return payload

    def _populate_inspector_tree(self, data) -> None:
        """Populate the Inspector QTreeWidget with a nested view of *data*."""
        try:
            self.treeMCT.clear()
        except Exception:
            return

        root = QtWidgets.QTreeWidgetItem(["session", self._format_inspector_value(data)])
        self.treeMCT.addTopLevelItem(root)
        self._inspector_add_children(root, data)
        try:
            self.treeMCT.expandAll()
        except Exception:
            pass

    def _inspector_add_children(self, parent: QtWidgets.QTreeWidgetItem, obj) -> None:
        """Recursive expansion of mappings, sequences, numpy arrays, and PyVista datasets."""
        # Avoid deep expansion of basic/leaf values
        try:
            import numpy as _np
        except Exception:
            _np = None
        try:
            import pyvista as _pv  # type: ignore
        except Exception:
            _pv = None

        # Dict-like
        try:
            from collections.abc import Mapping, Sequence
        except Exception:
            Mapping, Sequence = dict, (list, tuple)  # fallbacks

        if isinstance(obj, Mapping):
            for k, v in obj.items():
                key = str(k)
                val = self._format_inspector_value(v)
                child = QtWidgets.QTreeWidgetItem([key, val])
                parent.addChild(child)
                self._inspector_add_children(child, v)
            return

        # List/tuple-like (but not str/bytes)
        if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
            for i, v in enumerate(obj):
                key = f"[{i}]"
                val = self._format_inspector_value(v)
                child = QtWidgets.QTreeWidgetItem([key, val])
                parent.addChild(child)
                self._inspector_add_children(child, v)
            return

        # numpy arrays: show shape/dtype
        if _np is not None and isinstance(obj, _np.ndarray):
            # Already summarized in the value; also expose shape/dtype explicitly
            sh = tuple(obj.shape)
            dt = str(obj.dtype)
            parent.addChild(QtWidgets.QTreeWidgetItem(["shape", str(sh)]))
            parent.addChild(QtWidgets.QTreeWidgetItem(["dtype", dt]))
            return

        # PyVista datasets: summarize counts and arrays
        if _pv is not None and isinstance(obj, _pv.DataSet):
            try:
                parent.addChild(QtWidgets.QTreeWidgetItem(["type", type(obj).__name__]))
            except Exception:
                pass
            try:
                parent.addChild(QtWidgets.QTreeWidgetItem(["n_points", str(getattr(obj, "n_points", "?"))]))
            except Exception:
                pass
            try:
                parent.addChild(QtWidgets.QTreeWidgetItem(["n_cells", str(getattr(obj, "n_cells", "?"))]))
            except Exception:
                pass
            # Point data arrays
            try:
                if hasattr(obj, "point_data") and len(obj.point_data) > 0:
                    pd = QtWidgets.QTreeWidgetItem(["point_data", f"{len(obj.point_data)} arrays"])
                    parent.addChild(pd)
                    for name in obj.point_data.keys():
                        arr = obj.point_data[name]
                        label = f"{name}  shape={getattr(arr, 'shape', '?')} dtype={getattr(arr, 'dtype', '?')}"
                        pd.addChild(QtWidgets.QTreeWidgetItem([name, label]))
            except Exception:
                pass
            # Cell data arrays
            try:
                if hasattr(obj, "cell_data") and len(obj.cell_data) > 0:
                    cd = QtWidgets.QTreeWidgetItem(["cell_data", f"{len(obj.cell_data)} arrays"])
                    parent.addChild(cd)
                    for name in obj.cell_data.keys():
                        arr = obj.cell_data[name]
                        label = f"{name}  shape={getattr(arr, 'shape', '?')} dtype={getattr(arr, 'dtype', '?')}"
                        cd.addChild(QtWidgets.QTreeWidgetItem([name, label]))
            except Exception:
                pass
            return
        # Other types are treated as leaves

    def _format_inspector_value(self, v) -> str:
        """Short one-line summary for Inspector values."""
        try:
            import numpy as _np
        except Exception:
            _np = None
        try:
            import pyvista as _pv  # type: ignore
        except Exception:
            _pv = None

        if v is None:
            return "None"
        if isinstance(v, (bool, int, float, str)):
            return str(v)
        if isinstance(v, dict):
            return f"dict[{len(v)}]"
        if isinstance(v, (list, tuple)):
            return f"{type(v).__name__}[{len(v)}]"
        if _np is not None and isinstance(v, _np.ndarray):
            try:
                return f"ndarray shape={v.shape} dtype={v.dtype}"
            except Exception:
                return "ndarray"
        if _pv is not None and isinstance(v, _pv.DataSet):
            try:
                npts = getattr(v, 'n_points', '?')
                ncells = getattr(v, 'n_cells', '?')
                return f"{type(v).__name__} (pts={npts}, cells={ncells})"
            except Exception:
                return type(v).__name__
        # Generic fallback
        return type(v).__name__

    def _defer(self, ms: int, fn) -> None:
        """Run callable *fn* after *ms* milliseconds on the GUI thread (best-effort)."""
        try:
            QtCore.QTimer.singleShot(int(max(0, ms)), fn)
        except Exception:
            try:
                fn()
            except Exception:
                pass

    def _apply_cached_visuals(self, ds: int) -> None:
        """Reapply cached per-dataset visual properties to the live actor.

        Ensures that after a full-scene rebuild or actor creation, the dataset
        keeps its styling (point size, color mode, colormap, solid color, opacity).
        """
        v = getattr(self, "viewer3d", None)
        if v is None:
            return
        try:
            recs = getattr(v, "_datasets", [])
            if not (isinstance(ds, int) and 0 <= ds < len(recs)):
                return
            rec = recs[ds]
            if "point_size" in rec:
                try:
                    getattr(v, "set_point_size", lambda *_: None)(int(rec.get("point_size", 3)), ds)
                except Exception:
                    pass
            # Point budget / visible percentage for points datasets
            if "point_budget" in rec:
                try:
                    getattr(v, "set_point_budget", lambda *_: None)(int(rec.get("point_budget", 100)), ds)
                except Exception:
                    pass
            if "color_mode" in rec:
                try:
                    getattr(v, "set_color_mode", lambda *_: None)(str(rec.get("color_mode")), ds)
                except Exception:
                    pass
            if "colormap" in rec:
                try:
                    getattr(v, "set_colormap", lambda *_: None)(str(rec.get("colormap")), ds)
                except Exception:
                    pass
            if "solid_color" in rec:
                try:
                    r, g, b = rec["solid_color"]
                    if all(isinstance(c, float) and 0.0 <= c <= 1.0 for c in (r, g, b)):
                        r, g, b = int(r*255), int(g*255), int(b*255)
                    getattr(v, "set_dataset_color", lambda *_: None)(ds, int(r), int(g), int(b))
                except Exception:
                    pass
            if "opacity" in rec:
                try:
                    getattr(v, "set_mesh_opacity", lambda *_: None)(ds, int(rec.get("opacity", 100)))
                except Exception:
                    pass
            # Mesh representation (e.g., 'Surface', 'Wireframe')
            if "representation" in rec:
                try:
                    getattr(v, "set_mesh_representation", lambda *_: None)(ds, str(rec.get("representation")))
                except Exception:
                    pass
            # Points rendering style (if exposed)
            try:
                if "points_as_spheres" in rec:
                    getattr(v, "set_points_as_spheres", lambda *_: None)(bool(rec.get("points_as_spheres", True)))
            except Exception:
                pass
            # Honor cached normals visibility without enabling when False
            try:
                if bool(rec.get("normals_visible", False)):
                    getattr(v, "set_normals_visibility", lambda *_: None)(ds, True)
                else:
                    self._hide_normals_actor(ds)
            except Exception:
                pass
        except Exception:
            pass

    def _hide_normals_actor(self, ds: int) -> None:
        """Ensure normals are hidden for dataset *ds* (actor off + cache flag)."""
        v = getattr(self, "viewer3d", None)
        if v is None:
            return
        try:
            getattr(v, "set_normals_visibility", lambda *_: None)(ds, False)
        except Exception:
            pass
        try:
            recs = getattr(v, "_datasets", [])
            if 0 <= ds < len(recs):
                rec = recs[ds]
                rec["normals_visible"] = False
                act = rec.get("actor_normals") or rec.get("normals_actor")
                if act is not None:
                    try:
                        act.SetVisibility(0)
                    except Exception:
                        try:
                            act.visibility = False  # type: ignore[attr-defined]
                        except Exception:
                            pass
        except Exception:
            pass

    def _reset_viewer3d_from_tree(self) -> None:
        """
        Clear the 3D scene and rebuild visibility/actors of all objects based solely on each node's own check state.
        Also restores normals visibility based on a dedicated 'normals' node or per-dataset state.
        """
        v = getattr(self, "viewer3d", None)
        if v is None:
            return

        # 1) Clear scene
        try:
            if hasattr(v, "clear"):
                v.clear()
            else:
                getattr(v, "refresh", lambda: None)()
        except Exception:
            pass

        # 2) Re-apply according to each dataset node's own check (no propagation)
        normals_requests: list[tuple[int, bool]] = []  # (ds, visible)
        try:
            def recurse(node: QtWidgets.QTreeWidgetItem) -> None:
                data = node.data(0, QtCore.Qt.ItemDataRole.UserRole)
                if isinstance(data, dict):
                    kind = data.get("kind")
                    ds = data.get("ds")
                    if ds is not None and kind in ("points", "mesh", "normals"):
                        visible = self._node_self_checked(node)
                        # Build (or ensure) actors for data-bearing kinds
                        if kind in ("points", "mesh"):
                            # Ensure actor exists before toggling visibility
                            if bool(visible):
                                self._viewer_ensure_actor(kind, int(ds))
                            # Reapply cached visuals after actor creation
                            self._apply_cached_visuals(int(ds))
                            self._viewer_set_visibility(kind, int(ds), bool(visible))
                        elif kind == "normals":
                            normals_requests.append((int(ds), bool(visible)))
                # Recurse
                for i in range(node.childCount()):
                    recurse(node.child(i))

            for i in range(self.treeMCTS.topLevelItemCount()):
                recurse(self.treeMCTS.topLevelItem(i))
        except Exception:
            pass

        # 3) Apply normals visibility AFTER points actors exist
        try:
            for ds, on in normals_requests:
                getattr(v, "set_normals_visibility", lambda *_: None)(ds, bool(on))
                if not on:
                    self._hide_normals_actor(ds)
        except Exception:
            pass

        # 4) If no explicit normals node exists, restore per-dataset state (best effort)
        try:
            recs = getattr(v, "_datasets", [])
            for ds, rec in enumerate(recs):
                want = bool(rec.get("normals_visible", False))
                # Only if we didn't already apply an explicit request for this ds
                if all(ds_req != ds for ds_req, _ in normals_requests):
                    if want:
                        getattr(v, "set_normals_visibility", lambda *_: None)(ds, True)
        except Exception:
            pass

        # 5) Restore overlays (grid, units) and refresh
        try:
            self._restore_default_overlays()
        except Exception:
            pass
        try:
            v.refresh()
        except Exception:
            pass

    def _node_self_checked(self, item: QtWidgets.QTreeWidgetItem) -> bool:
        """Return True if the item's own checkbox is checked (no parent/child propagation)."""
        try:
            return item.checkState(0) == QtCore.Qt.CheckState.Checked
        except Exception:
            return False

    def _reapply_overlays_safe(self) -> None:
        """
        Ask the viewer to re-apply overlays (grid, units overlay, etc.) without changing camera.
        Safe no-op if the viewer does not implement it.
        """
        v = getattr(self, "viewer3d", None)
        if v is None:
            return
        # Prefer a public method if available; otherwise accept a private one.
        for name in ("reapply_overlays", "_apply_overlays"):
            fn = getattr(v, name, None)
            if callable(fn):
                try:
                    fn()
                    return
                except Exception:
                    continue
        # Fallback: ripristina come in _restore_default_overlays
        try:
            self._restore_default_overlays()
        except Exception:
            pass
        
    def _restore_default_overlays(self) -> None:
        """
        Re-enable default viewer overlays after a full scene reset:
        - Grid (according to toolbar toggle)
        - Units overlay/ruler if UnitsPlugin is available
        """
        v = getattr(self, "viewer3d", None)
        if v is None:
            return
        # Grid
        try:
            on = bool(self.act_toggle_grid.isChecked())
        except Exception:
            on = True
        try:
            for name in ("set_grid_visible", "toggle_grid", "show_grid"):
                fn = getattr(v, name, None)
                if callable(fn):
                    try:
                        fn(on)
                        break
                    except Exception:
                        continue
        except Exception:
            pass
        # Units overlay (best effort)
        try:
            pm = getattr(self, "plugin_manager", None)
            if pm is not None:
                units = None
                for getter in ("get", "plugin_by_key"):
                    fn = getattr(pm, getter, None)
                    if callable(fn):
                        units = fn("units")
                        if units:
                            break
                if units and hasattr(units, "overlay") and hasattr(units, "state"):
                    try:
                        units.overlay.show_text(units.state)
                    except Exception:
                        pass
        except Exception:
            pass
        
    def _dataset_info_from_item(self, item: QtWidgets.QTreeWidgetItem | None):
        """
        Retrieve dataset info from the given tree item or its neighbors.

        Args:
            item (QtWidgets.QTreeWidgetItem | None): The tree item to extract info from.

        Returns:
            dict | None: The dataset info dictionary if found, otherwise None.
        """
        if item is None:
            return None
        data = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if isinstance(data, dict) and data.get("ds") is not None:
            return data
        for i in range(item.childCount()):
            found = self._dataset_info_from_item(item.child(i))
            if found:
                return found
        parent = item.parent()
        if parent is not None:
            return self._dataset_info_from_item(parent)
        return None

    def _viewer_set_visibility(self, kind: str, ds: int, visible: bool) -> None:
        """Toggle dataset visibility in the 3D viewer; ensure actor exists when turning ON.
        Also: when POINTS are OFF, force normals OFF for the same dataset."""
        v = getattr(self, "viewer3d", None)
        if v is None:
            return

        # Ensure actor exists if we are turning ON after a clear()
        if visible and kind in ("points", "mesh"):
            self._viewer_ensure_actor(kind, ds)
        # Re-apply cached visuals so color/colormap don't reset to white defaults
        if visible:
            try:
                self._apply_cached_visuals(ds)
            except Exception:
                pass

        # Preferred explicit API
        for name in ("set_dataset_visibility", "set_visibility", "set_points_visibility", "set_mesh_visibility"):
            fn = getattr(v, name, None)
            if callable(fn):
                try:
                    try:
                        fn(ds, bool(visible))
                    except TypeError:
                        fn(kind, ds, bool(visible))
                    if kind == "points" and not visible:
                        getattr(v, "set_normals_visibility", lambda *_: None)(ds, False)
                    # Persist visible flag into cache + session dicts
                    try:
                        if kind == "points":
                            self._persist_dataset_prop(ds, "visible", bool(visible))
                        elif kind == "mesh":
                            self._persist_dataset_prop(ds, "visible", bool(visible))
                    except Exception:
                        pass
                    v.refresh()
                    return
                except Exception:
                    pass

        # Fallback: cached record + actor toggling
        try:
            recs = getattr(v, "_datasets", [])
            if not (isinstance(ds, int) and 0 <= ds < len(recs)):
                return
            rec = recs[ds]
            rec["visible"] = bool(visible)
            actor = rec.get("actor") or rec.get("actor_mesh") or rec.get("actor_points")
            if actor is None and visible:
                # Try again to build (another safety)
                self._viewer_ensure_actor(kind, ds)
                actor = rec.get("actor") or rec.get("actor_mesh") or rec.get("actor_points")
            # After (re)creating the actor, re-apply cached styling (solid color, colormap, rep, etc.)
            if visible:
                try:
                    self._apply_cached_visuals(ds)
                except Exception:
                    pass
            if actor is not None:
                try:
                    actor.SetVisibility(1 if visible else 0)
                except Exception:
                    try:
                        actor.visibility = bool(visible)  # type: ignore[attr-defined]
                    except Exception:
                        pass
            if kind == "points" and not visible:
                try:
                    getattr(v, "set_normals_visibility", lambda *_: None)(ds, False)
                except Exception:
                    pass
            # Persist visible flag into cache + session dicts
            try:
                if kind == "points":
                    self._persist_dataset_prop(ds, "visible", bool(visible))
                elif kind == "mesh":
                    self._persist_dataset_prop(ds, "visible", bool(visible))
            except Exception:
                pass
            v.refresh()
        except Exception:
            pass
        
    def _viewer_ensure_actor(self, kind: str, ds: int) -> None:
        """Best-effort: (re)build the actor for dataset `ds` of type `kind`.
        Used after a full viewer clear() or when an actor is missing."""
        v = getattr(self, "viewer3d", None)
        if v is None:
            return
        try:
            recs = getattr(v, "_datasets", [])
            if not (isinstance(ds, int) and 0 <= ds < len(recs)):
                return
            rec = recs[ds]
            # If an actor already exists, nothing to do
            actor = rec.get("actor") or rec.get("actor_mesh") or rec.get("actor_points")
            if actor is not None:
                return

            # Try public/safe rebuild paths first
            for name in ("rebuild_dataset", "ensure_dataset_actor", "build_dataset_actor"):
                fn = getattr(v, name, None)
                if callable(fn):
                    try:
                        fn(ds)
                        try:
                            self._apply_cached_visuals(ds)
                        except Exception:
                            pass
                        return
                    except Exception:
                        pass

            # Fall back to common internal helpers (pyvista-based viewers often have these)
            if kind == "points":
                # common internal name
                for name in ("_rebuild_points_actor", "_update_points_actor", "_ensure_points_actor"):
                    fn = getattr(v, name, None)
                    if callable(fn):
                        try:
                            fn(ds)
                            try:
                                self._apply_cached_visuals(ds)
                            except Exception:
                                pass
                            return
                        except Exception:
                            pass
                # very last resort: add from pdata directly
                pdata = rec.get("pdata") or rec.get("full_pdata")
                if pdata is not None:
                    try:
                        actor = v.plotter.add_mesh(
                            pdata, render_points_as_spheres=bool(rec.get("points_as_spheres", True)),
                            point_size=int(rec.get("point_size", 3)), name=f"points_ds{ds}"
                        )
                        rec["actor_points"] = actor
                        try:
                            self._apply_cached_visuals(ds)
                        except Exception:
                            pass
                    except Exception:
                        pass

            elif kind == "mesh":
                for name in ("_rebuild_mesh_actor", "_update_mesh_actor", "_ensure_mesh_actor"):
                    fn = getattr(v, name, None)
                    if callable(fn):
                        try:
                            fn(ds)
                            try:
                                self._apply_cached_visuals(ds)
                            except Exception:
                                pass
                            return
                        except Exception:
                            pass
                mesh = rec.get("mesh") or rec.get("pdata") or rec.get("full_pdata")
                if mesh is not None:
                    try:
                        actor = v.plotter.add_mesh(
                            mesh,
                            opacity=float(rec.get("opacity", 100))/100.0,
                            name=f"mesh_ds{ds}"
                        )
                        rec["actor_mesh"] = actor
                        try:
                            self._apply_cached_visuals(ds)
                        except Exception:
                            pass
                    except Exception:
                        pass
        except Exception:
            pass
        
    def _current_dataset_index(self) -> Optional[int]:
        """
        Return the dataset index of the currently selected tree item.

        Returns:
            Optional[int]: The dataset index if available, otherwise None.
        """
        item = self.treeMCTS.currentItem()
        info = self._dataset_info_from_item(item)
        if info:
            return info.get("ds")
        return None

    def _persist_dataset_prop(self, ds: int, key: str, value) -> None:
        """Persist a per‑dataset property both in the viewer cache (self.viewer3d._datasets)
        and in the corresponding entry of self.mcts used for session snapshots.

        Args:
            ds: Dataset index in the viewer cache.
            key: Property name (e.g., 'point_size').
            value: Property value to store.
        """
        # Update viewer cache (used by actor rebuilds)
        try:
            recs = getattr(self.viewer3d, "_datasets", [])
            if isinstance(ds, int) and 0 <= ds < len(recs):
                recs[ds][key] = value
        except Exception:
            pass

        # Update mcts registry snapshot (used by UI + sessions)
        try:
            for e in self.mcts.values():
                if e.get("ds_index") == ds:
                    e[key] = value
        except Exception:
            pass

    def _persist_dataset_color(self, ds: int, rgb_tuple: tuple[int, int, int]) -> None:
        """Specialized helper for solid RGB color persistence (kept separate for clarity)."""
        self._persist_dataset_prop(ds, "solid_color", tuple(rgb_tuple))

    def _on_point_size_changed(self, size: int) -> None:
        """
        Update the point size for the currently selected dataset.

        Args:
            size (int): The new point size to set.
        """
        ds = self._current_dataset_index()
        if ds is None:
            return
        self.viewer3d.set_point_size(int(size), ds)
        if self.mct:
            self.mct["point_size"] = int(size)
        # Persist also into the viewer cache so actor rebuilds keep the new size
        self._persist_dataset_prop(ds, "point_size", int(size))
        def _post_psize():
            try:
                self._apply_cached_visuals(ds)
            except Exception:
                pass
        self._defer(0, _post_psize)

    def _on_point_budget_changed(self, percent: int) -> None:
        """
        Update the visible points percentage (point budget) for the selected dataset.

        Args:
            percent (int): The percentage of points to display.
        """
        ds = self._current_dataset_index()
        if ds is None:
            return
        self.viewer3d.set_point_budget(int(percent), ds)
        if self.mct:
            self.mct["point_budget"] = int(percent)
        self._persist_dataset_prop(ds, "point_budget", int(percent))
        try:
            self._viewer_ensure_actor("points", ds)
        except Exception:
            pass
        def _post_budget():
            try:
                self._apply_cached_visuals(ds)
            except Exception:
                pass
        self._defer(0, _post_budget)
        try:
            self.viewer3d.refresh()
        except Exception:
            pass

    def _on_color_mode_changed(self, mode: str) -> None:
        """
        Change the color mode of the selected dataset.

        Args:
            mode (str): The color mode to set (e.g., 'Normal RGB', 'Normal Colormap').
        """
        ds = self._current_dataset_index()
        if ds is None:
            return
        self.viewer3d.set_color_mode(mode, ds)
        if self.mct:
            self.mct["color_mode"] = mode
        self._persist_dataset_prop(ds, "color_mode", str(mode))
        try:
            self.viewer3d.refresh()
        except Exception:
            pass

    def _on_solid_color_changed(self, col: QtGui.QColor) -> None:
        """
        Set the solid color for the selected dataset.

        Args:
            col (QtGui.QColor): The color to set.
        """
        ds = self._current_dataset_index()
        if ds is None or not col.isValid():
            return
        self.viewer3d.set_dataset_color(ds, col.red(), col.green(), col.blue())
        if self.mct:
            self.mct["solid_color"] = (col.red(), col.green(), col.blue())
        self._persist_dataset_color(ds, (col.red(), col.green(), col.blue()))
        try:
            self.viewer3d.refresh()
        except Exception:
            pass

    def _on_colormap_changed(self, name: str) -> None:
        """
        Update the colormap for the selected dataset.

        Args:
            name (str): The name of the colormap to apply.
        """
        ds = self._current_dataset_index()
        if ds is None:
            return
        self.viewer3d.set_colormap(name, ds)
        if self.mct:
            self.mct["colormap"] = name
        self._persist_dataset_prop(ds, "colormap", str(name))
        try:
            self.viewer3d.refresh()
        except Exception:
            pass

    def _on_mesh_rep_changed(self, mode: str) -> None:
        """
        Change the mesh representation mode for the selected mesh dataset.

        Args:
            mode (str): The mesh representation mode (e.g., 'Surface', 'Wireframe').
        """
        ds = self._current_dataset_index()
        if ds is None:
            return
        self.viewer3d.set_mesh_representation(ds, mode)
        if self.mct:
            self.mct["representation"] = mode
        self._persist_dataset_prop(ds, "representation", str(mode))
        # TEMP: rebuild scene once to avoid desync/orphan actors
        self._schedule_scene_rebuild()

    def _on_mesh_opacity_changed(self, val: int) -> None:
        """
        Update the opacity for the selected mesh dataset.

        Args:
            val (int): The opacity value (0-100).
        """
        ds = self._current_dataset_index()
        if ds is None:
            return
        self.viewer3d.set_mesh_opacity(ds, int(val))
        if self.mct:
            self.mct["opacity"] = int(val)
        self._persist_dataset_prop(ds, "opacity", int(val))
        # TEMP: rebuild scene once to avoid desync/orphan actors
        self._schedule_scene_rebuild()

    def _on_toggle_normals_clicked(self, on: bool) -> None:
        """Toggle Normals dal pulsante della toolbar sul dataset selezionato."""
        ds = self._current_dataset_index()
        if ds is None:
            # niente dataset selezionato
            return

        # Se non esiste il nodo "Normals" nel tree, crealo (non lo spunta ancora).
        normals_item = None
        try:
            normals_item = self._ensure_normals_tree_child(ds)
        except Exception:
            normals_item = None
        if normals_item is not None:
            try:
                normals_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, {"kind": "normals", "ds": int(ds)})
            except Exception:
                pass

        # Attiva/disattiva visibilità lato viewer
        try:
            getattr(self.viewer3d, "set_normals_visibility", lambda *_: None)(ds, bool(on))
        except Exception:
            return

        # Persist normals visibility in the dataset record
        try:
            self._persist_dataset_prop(ds, "normals_visible", bool(on))
        except Exception:
            pass

        def _post_normals_toggle():
            try:
                self._apply_cached_visuals(ds)
            except Exception:
                pass
            try:
                self.viewer3d.refresh()
            except Exception:
                pass
        self._defer(1, _post_normals_toggle)
        try:
            self._reapply_overlays_safe()
        except Exception:
            pass

        # Sincronizza lo stato del nodo "Normals" nell'albero (se presente)
        try:
            item = self.treeMCTS.currentItem()
            if item is not None:
                # sali al root (file)
                root = item
                while root.parent() is not None:
                    root = root.parent()
                # trova il figlio "Point cloud" e poi "Normals"
                target = None
                for i in range(root.childCount()):
                    if root.child(i).text(0) == "Point cloud":
                        target = root.child(i)
                        break
                if target is not None:
                    for i in range(target.childCount()):
                        ch = target.child(i)
                        data = ch.data(0, QtCore.Qt.ItemDataRole.UserRole)
                        if ch.text(0) == "Normals" and isinstance(data, dict) and data.get("ds") == ds:
                            ch.setCheckState(0, QtCore.Qt.CheckState.Checked if on else QtCore.Qt.CheckState.Unchecked)
                            break
        except Exception:
            pass
        # TEMP: rebuild scene once to avoid desync/orphan actors
        # self._schedule_scene_rebuild()
        
    def _build_statusbar(self) -> None:
        sb = QtWidgets.QStatusBar(self)
        self.setStatusBar(sb)

        # --- Widgets -----------------------------------------------------
        self.btnCancel = QtWidgets.QPushButton("CANCEL")
        self.btnCancel.setObjectName("buttonCANCEL")
        self.btnCancel.setEnabled(False)
        self.btnCancel.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setObjectName("barPROGRESS")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Idle")
        self.progress.setTextVisible(True)
        self.progress.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)

        self.disk = QtWidgets.QProgressBar()
        self.disk.setObjectName("diskUsageBar")
        self.disk.setRange(0, 100)
        self.disk.setValue(0)
        self.disk.setTextVisible(True)
        self.disk.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)

        # Container to control layout & stretches (so showMessage won't hide widgets)
        panel = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addWidget(self.btnCancel)
        lay.addWidget(self.progress, 3)  # ~60%
        lay.addWidget(self.disk, 2)      # ~40%

        # Add as permanent widget so temporary status messages don't hide it
        sb.addPermanentWidget(panel, 1)

    def _start_disk_timer(self) -> None:
        self._disk_timer = QtCore.QTimer(self)
        self._disk_timer.timeout.connect(self._update_disk)
        self._disk_timer.start(2000)
        self._update_disk()

    def _update_disk(self) -> None:
        used, free, percent = disk_usage_percent()
        self.disk.setValue(int(percent))
        self.disk.setFormat(f"Disk used: {percent:.0f}%")

    # ------------------------------------------------------------------
    # Progress bar helpers + background-worker slots
    # ------------------------------------------------------------------
    # --- Progress helpers (import cloud) ---------------------------------
    def _import_progress_begin(self, title: str = "Importing cloud…") -> None:
        """Begin an import progress section (safe to call multiple times)."""
        try:
            # Reuse the generic progress bar helpers already wired to the status bar
            self._progress_start(title)
        except Exception:
            pass

    def _import_progress_update(self, percent: int | None = None, message: str | None = None) -> None:
        """Update progress bar during import."""
        try:
            if percent is not None:
                v = int(max(0, min(100, percent)))
                try:
                    self.progress.setValue(v)
                except Exception:
                    pass
            if message:
                # Show a short status text while keeping the bar visible
                try:
                    self.progress.setFormat(str(message))
                except Exception:
                    pass
                try:
                    self.statusBar().showMessage(message, 2000)
                except Exception:
                    pass
        except Exception:
            pass

    def _import_progress_end(self) -> None:
        """End the import progress section."""
        try:
            self._progress_finish()
        except Exception:
            pass
    # ---------------------------------------------------------------------
    def _progress_start(self, text: str = "Working…") -> None:
        """Initialize the statusbar progress bar."""
        try:
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            self.progress.setFormat(text)
            self.progress.setTextVisible(True)
        except Exception:
            pass

    def _progress_set(self, value: int | float, text: str | None = None) -> None:
        """Update progress value (0..100) and optional text."""
        try:
            v = int(max(0, min(100, round(float(value)))))
            self.progress.setValue(v)
            if text is not None:
                self.progress.setFormat(str(text))
        except Exception:
            pass

    def _progress_finish(self, text: str = "Ready") -> None:
        """Finish the progress and reset UI bits."""
        try:
            self.progress.setValue(100)
            self.progress.setFormat(text)
        except Exception:
            pass
        # Disable cancel if it was enabled
        try:
            self.btnCancel.setEnabled(False)
        except Exception:
            pass
        # Clear active job context
        try:
            self._active_job = None
        except Exception:
            pass

    def _ensure_cancel_button(self) -> None:
        """Best-effort: ensure the Cancel button is visible/enabled for jobs."""
        try:
            self.btnCancel.setEnabled(True)
        except Exception:
            pass

    @QtCore.Slot()
    def _on_cancel_job(self) -> None:
        """User pressed CANCEL: ask the active worker to stop."""
        job = getattr(self, "_active_job", None)
        worker = None
        if isinstance(job, dict):
            worker = job.get("worker")
        try:
            if worker is not None and hasattr(worker, "request_cancel"):
                worker.request_cancel()
            self.statusBar().showMessage("Cancelling…", 2000)
        except Exception:
            pass

    # --- Slots used by _NormalsWorker (progress/message/finished) -----------
    @QtCore.Slot(int)
    def _slot_worker_progress(self, p: int) -> None:
        """Update the progress bar percentage."""
        self._progress_set(p)

    @QtCore.Slot(str)
    def _slot_worker_message(self, msg: str) -> None:
        """Reflect worker text into the progress format."""
        self._progress_set(self.progress.value(), msg)

    @QtCore.Slot(object)
    def _slot_worker_finished(self, normals_obj) -> None:
        """
        Worker finished. If `normals_obj` is an Nx3 array, attach/store it to the
        current dataset and (optionally) show normals based on the toolbar toggle.
        Always tear down the worker thread and finalize the progress bar.
        """
        # Tear down the worker thread (best-effort)
        try:
            ctx = getattr(self, "_normals_ctx", {}) or {}
            thread = ctx.get("thread")
            if thread is not None:
                try:
                    thread.quit()
                    thread.wait()
                except Exception:
                    pass
        except Exception:
            pass

        # If we got normals, persist them into the viewer's dataset
        try:
            import numpy as _np  # local import to keep module import-time light
            ds = int(self._normals_ctx.get("ds")) if hasattr(self, "_normals_ctx") else None
            if normals_obj is not None and ds is not None:
                arr = _np.asarray(normals_obj)
                if arr.ndim == 2 and arr.shape[1] == 3:
                    # Try a dedicated API first
                    set_api = getattr(self.viewer3d, "set_normals_array", None)
                    if callable(set_api):
                        try:
                            set_api(ds, arr)
                        except Exception:
                            pass
                    # Also update the cached record and point_data if accessible
                    try:
                        recs = getattr(self.viewer3d, "_datasets", [])
                        if 0 <= ds < len(recs):
                            rec = recs[ds]
                            rec["normals_array"] = arr
                            pdata = rec.get("pdata") or rec.get("full_pdata")
                            if pdata is not None and hasattr(pdata, "point_data"):
                                pdata.point_data["Normals"] = arr
                    except Exception:
                        pass

                    # Apply initial display percentage from context, default 1%
                    try:
                        percent = int(getattr(self, "_normals_ctx", {}).get("percent", 1))
                    except Exception:
                        percent = 1
                    # Persist into the dataset record so builder uses it
                    try:
                        recs = getattr(self.viewer3d, "_datasets", [])
                        if 0 <= ds < len(recs):
                            rec = recs[ds]
                            rec["normals_percent"] = percent
                    except Exception:
                        pass
                    # If the viewer exposes a setter for percent, use it; otherwise trigger a rebuild
                    try:
                        setp = getattr(self.viewer3d, "set_normals_percent", None)
                        if callable(setp):
                            # Some implementations accept (ds, percent, rebuild=True)
                            try:
                                setp(ds, percent, True)
                            except TypeError:
                                setp(ds, percent)
                        else:
                            # Best-effort: rebuild normals actor so the new percent is honored
                            rebuild = getattr(self.viewer3d, "_rebuild_normals_actor", None)
                            if callable(rebuild):
                                rebuild(ds)
                    except Exception:
                        pass

                    # --- Ensure normals are actually shown once computed: force-build actor and toggle ON ---
                    try:
                        # Persist visibility flag in cache
                        recs = getattr(self.viewer3d, "_datasets", [])
                        if 0 <= ds < len(recs):
                            recs[ds]["normals_visible"] = True
                    except Exception:
                        pass
                    try:
                        # Build or rebuild the normals actor explicitly if the viewer exposes it
                        rebuild_normals = getattr(self.viewer3d, "_rebuild_normals_actor", None)
                        if callable(rebuild_normals):
                            rebuild_normals(ds)
                    except Exception:
                        pass
                    try:
                        # Turn on the toolbar toggle (UI) without emitting recursive signals
                        self.act_toggle_normals.blockSignals(True)
                        self.act_toggle_normals.setChecked(True)
                        self.act_toggle_normals.blockSignals(False)
                    except Exception:
                        pass
                    try:
                        # Finally, ensure visibility ON at the viewer level
                        getattr(self.viewer3d, "set_normals_visibility", lambda *_: None)(ds, True)
                    except Exception:
                        pass

                    # Visibility handled above: we explicitly turned normals ON after compute.
                    # (Old code removed here.)
        except Exception:
            pass

        # Finalize UI
        self._progress_finish("Normals done")

    def _on_compute_normals(self) -> None:
        """Compute point‑cloud normals in a background thread (non‑blocking UI).

        Priority: PCA on k-NN neighborhoods with optional FAST mode (subset + propagate).
        On success, stores normals into the current dataset entry and updates the UI.
        """
        import numpy as np

        self._progress_start("Starting normals computation…")
        self._ensure_cancel_button()

        # Grab current dataset points from viewer cache if available via self.mct
        entry = getattr(self, "mct", None)
        if not entry or entry.get("ds_index") is None:
            self._append_message("[Normals] No active dataset selected.")
            self._progress_finish("Normals not computed: no dataset")
            return
        ds = int(entry["ds_index"]) if entry.get("ds_index") is not None else None

        # Try to fetch points back from viewer; fall back to stored structures if available
        P = None
        try:
            datasets = getattr(self.viewer3d, "_datasets", [])
            if isinstance(ds, int) and 0 <= ds < len(datasets):
                fp = datasets[ds].get("full_pdata") or datasets[ds].get("pdata")
                # Expect PyVista PolyData or numpy array alike
                if hasattr(fp, "points"):
                    P = np.asarray(fp.points, dtype=float)
                elif hasattr(fp, "to_numpy"):
                    P = np.asarray(fp.to_numpy(), dtype=float)
                else:
                    P = np.asarray(fp, dtype=float)
        except Exception:
            P = None

        if P is None or P.ndim != 2 or P.shape[1] != 3 or P.shape[0] == 0:
            self._append_message("[Normals] Cannot access points for current dataset.")
            self._progress_finish("Normals not computed: invalid points")
            return

        # Read fast flag
        # Fast-mode flag: prefer DisplayPanel state if present, otherwise fallback
        fast_flag = False
        try:
            if hasattr(self, "displayPanel") and self.displayPanel is not None:
                fast_flag = bool(self.displayPanel.fast_normals_enabled())
            else:
                fast_flag = bool(getattr(self, "normals_fast_enabled", False))
        except Exception:
            pass

        # Parameters (you can expose k via settings later)
        k_nn = int(getattr(self, "normals_k", 16))
        max_fast = int(getattr(self, "normals_fast_max_points", 250_000))

        # Initial display percentage for normals (default 1%)
        try:
            percent_ui = int(self.displayPanel.spinNormalsPercent.value())
        except Exception:
            percent_ui = 1

        # Harden against macOS/Accelerate and OpenMP threading issues inside background threads
        try:
            import os as _os
            _os.environ.setdefault("OMP_NUM_THREADS", "1")
            _os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
            _os.environ.setdefault("MKL_NUM_THREADS", "1")
            _os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
            _os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        except Exception:
            pass

        # Prepare worker
        worker = _NormalsWorker(points=P, k=k_nn, subset_size=80_000,
                                fast=fast_flag, fast_max_points=max_fast)
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)

        # salva contesto per lo slot di fine
        self._normals_ctx = {"P": P, "ds": ds, "entry": entry, "thread": thread, "percent": percent_ui}

        # connessioni ai nuovi slot (GUI thread garantito)
        worker.progress.connect(self._slot_worker_progress, QtCore.Qt.ConnectionType.QueuedConnection)
        worker.message.connect(self._slot_worker_message, QtCore.Qt.ConnectionType.QueuedConnection)
        worker.finished.connect(self._slot_worker_finished, QtCore.Qt.ConnectionType.QueuedConnection)

        thread.started.connect(worker.run)
        thread.start()

        # Expose active job for cancellation
        self._active_job = {"worker": worker, "thread": thread}
        self.btnCancel.clicked.connect(self._on_cancel_job, QtCore.Qt.ConnectionType.UniqueConnection)
        self.btnCancel.setEnabled(True)

    # ------------------------------------------------------------------
    # Session I/O: New / Open / Save / Save As
    # ------------------------------------------------------------------
    def _on_new_session(self) -> None:
        """Start a new empty session, clearing tree, viewer and state."""
        try:
            if self.mcts:
                ret = QtWidgets.QMessageBox.question(
                    self, "New Session",
                    "Discard current session and start a new one?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.No,
                )
                if ret != QtWidgets.QMessageBox.Yes:
                    return
        except Exception:
            pass
        # Clear UI state
        try:
            self.treeMCTS.clear()
        except Exception:
            pass
        try:
            if hasattr(self.viewer3d, "clear"):
                self.viewer3d.clear()
        except Exception:
            pass
        self.mcts = {}
        self.mct = {}
        self._session_path = None
        try:
            self.statusBar().showMessage("New session", 3000)
        except Exception:
            pass

    def _on_open_session(self) -> None:
        """Open a saved C2F4DT session (.c2f4dt.json)."""
        dlg = QtWidgets.QFileDialog(self, "Open session")
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        dlg.setNameFilters(["C2F4DT Session (*.c2f4dt.json)", "JSON (*.json)", "All files (*)"])
        if not dlg.exec():
            return
        paths = dlg.selectedFiles()
        if not paths:
            return
        self._load_session_from_file(paths[0])

    def _on_save_session(self) -> None:
        """Save the current session to disk; if untitled, fallback to Save As."""
        if not self._session_path:
            self._on_save_session_as()
            return
        data = self._session_snapshot()
        try:
            with open(self._session_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            try:
                self.statusBar().showMessage(f"Saved session to {os.path.basename(self._session_path)}", 3000)
            except Exception:
                pass
        except Exception as ex:
            QtWidgets.QMessageBox.critical(self, "Save error", str(ex))

    def _on_save_session_as(self) -> None:
        """Prompt for a path and save the session JSON there."""
        dlg = QtWidgets.QFileDialog(self, "Save session as")
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.setNameFilters(["C2F4DT Session (*.c2f4dt.json)", "JSON (*.json)", "All files (*)"])
        dlg.setDefaultSuffix("c2f4dt.json")
        if not dlg.exec():
            return
        paths = dlg.selectedFiles()
        if not paths:
            return
        self._session_path = paths[0]
        # ensure extension
        if not (self._session_path.endswith(".c2f4dt.json") or self._session_path.endswith(".json")):
            self._session_path += ".c2f4dt.json"
        self._on_save_session()

    def _session_snapshot(self) -> dict:
        """Capture a lightweight, JSON‑serializable snapshot of the current session."""
        snap: dict = {"version": 1, "datasets": [], "viewer": {}, "options": {}}
        # Viewer globals
        try:
            v = self.viewer3d
            snap["viewer"] = {
                "color_mode": getattr(v, "_color_mode", None),
                "colormap": getattr(v, "_cmap", None),
                "point_size": getattr(v, "_point_size", None),
                "view_budget_percent": getattr(v, "_view_budget_percent", None),
                "points_as_spheres": getattr(v, "_points_as_spheres", None),
            }
        except Exception:
            pass
        # App options
        try:
            snap["options"]["downsample_method"] = getattr(self, "downsample_method", None)
            snap["options"]["normals_fast_enabled"] = bool(getattr(self, "normals_fast_enabled", False))
            snap["options"]["normals_k"] = int(getattr(self, "normals_k", 16))
            snap["options"]["normals_fast_max_points"] = int(getattr(self, "normals_fast_max_points", 250_000))
        except Exception:
            pass
        # Datasets (from mcts registry)
        try:
            for name, entry in self.mcts.items():
                ds = {
                    "name": name,
                    "kind": entry.get("kind"),
                    "ds_index": entry.get("ds_index"),
                    "point_size": entry.get("point_size"),
                    "point_budget": entry.get("point_budget"),
                    "color_mode": entry.get("color_mode"),
                    "colormap": entry.get("colormap"),
                    "solid_color": entry.get("solid_color"),
                    "points_as_spheres": entry.get("points_as_spheres"),
                    # Optional: if your importers store the original path
                    "source_path": entry.get("source_path"),
                }
                snap["datasets"].append(ds)
        except Exception:
            pass
        return snap

    def _load_session_from_file(self, path: str) -> None:
        """Load a session JSON and rebuild the scene as much as possible.

        If a dataset has `source_path`, it will be re-imported automatically.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as ex:
            QtWidgets.QMessageBox.critical(self, "Open error", f"Cannot read session: {ex}")
            return
        # Reset current state
        self._on_new_session()
        self._session_path = path
        # Restore viewer/app options (best effort)
        try:
            opts = data.get("options", {})
            self.downsample_method = opts.get("downsample_method", self.downsample_method)
            self.normals_fast_enabled = bool(opts.get("normals_fast_enabled", getattr(self, "normals_fast_enabled", False)))
            self.normals_k = int(opts.get("normals_k", getattr(self, "normals_k", 16)))
            self.normals_fast_max_points = int(opts.get("normals_fast_max_points", getattr(self, "normals_fast_max_points", 250_000)))
        except Exception:
            pass
        # Re-import datasets by source_path if available
        restored = 0
        for ds in data.get("datasets", []):
            src = ds.get("source_path")
            if src and os.path.exists(src):
                try:
                    # Use programmatic import if available
                    if hasattr(self, "import_cloud_programmatic"):
                        self.import_cloud_programmatic(src)
                        restored += 1
                except Exception:
                    continue
        try:
            self.statusBar().showMessage(f"Opened session: restored {restored} dataset(s)", 5000)
        except Exception:
            pass

    def _populate_plugins_ui(self) -> None:
        """Riempi la combo e ricostruisci il menù Plugins con le azioni esposte dai plugin."""
        items = self.plugin_manager.ui_combo_items()
        self.comboPlugins.clear()
        if not items:
            self.comboPlugins.addItem("— No plugins installed —")
            self.comboPlugins.setEnabled(False)
        else:
            self.comboPlugins.setEnabled(True)

            color_map = {
                "red": QtGui.QColor("#e53935"),
                "green": QtGui.QColor("#43a047"),
                "gray": QtGui.QColor("#9e9e9e"),
                "black": QtGui.QColor("#000000"),
            }

            for it in items:
                self.comboPlugins.addItem(it["label"], userData=it.get("key"))
                idx = self.comboPlugins.count() - 1
                # tooltip e colore
                self.comboPlugins.setItemData(idx, it.get("tooltip", ""), QtCore.Qt.ItemDataRole.ToolTipRole)
                qcol = color_map.get(it.get("color", "black"), color_map["black"])
                self.comboPlugins.setItemData(idx, qcol, QtCore.Qt.ItemDataRole.TextColorRole)
                # disabilita se non disponibile
                if not it.get("enabled", True):
                    mdl = self.comboPlugins.model()
                    mitem = mdl.item(idx)
                    if mitem is not None:
                        mitem.setEnabled(False)

        # Ricostruisci il menù Plugins
        try:
            self._rebuild_plugins_menu(items)
        except Exception:
            pass

    # --------------------- Plugins wiring ---------------------------------

    def _plugin_context(self) -> dict:
        """Contesto standard passato ai plugin."""
        return {
            "window": self,
            "viewer3d": getattr(self, "viewer3d", None),
            "mcts": getattr(self, "mcts", {}),
            "mct": getattr(self, "mct", {}),
            "current_dataset": self._current_dataset_index(),
            "display": getattr(self, "displayPanel", None),
            "console": getattr(self, "console", None),
            # aggiungi qui oggetti utili che i tuoi plugin si aspettano
        }

    @QtCore.Slot(int)
    def _on_plugin_combo_activated(self, index: int) -> None:
        try:
            key = self.comboPlugins.itemData(index)  # lo impostiamo in _populate_plugins_ui
            if not key:
                return
            self._run_plugin_by_key(str(key))
        except Exception as ex:
            QtWidgets.QMessageBox.warning(self, "Plugin", f"Cannot run plugin: {ex}")

    def _run_plugin_by_key(self, key: str) -> None:
        """Trova il plugin per 'key' e prova ad eseguirlo in modo robusto (senza introspezione fragile)."""
        try:
            # 1) recupera l'oggetto plugin (lazy get)
            plugin = None
            for attr in ("get", "plugin_by_key"):
                fn_get = getattr(self.plugin_manager, attr, None)
                if callable(fn_get):
                    plugin = fn_get(key)
                    break

            # fallback: guarda nella lista items se già istanziato
            if plugin is None:
                try:
                    for it in self.plugin_manager.ui_combo_items():
                        if it.get("key") == key and it.get("plugin_obj") is not None:
                            plugin = it["plugin_obj"]
                            break
                except Exception:
                    pass

            if plugin is None:
                QtWidgets.QMessageBox.warning(self, "Plugin", f"Plugin '{key}' not found.")
                return

            ctx = self._plugin_context()

            # helper per chiamare callables in modo sicuro
            def _call_safe(fn):
                try:
                    fn(ctx)
                    return True
                except TypeError:
                    try:
                        fn()
                        return True
                    except Exception:
                        return False
                except Exception:
                    return False

            # 2) se il plugin espone azioni strutturate, usale
            actions = None
            for attr in ("actions", "get_actions"):
                getter = getattr(plugin, attr, None)
                if callable(getter):
                    try:
                        actions = getter()
                    except Exception:
                        actions = None
                    break

            if isinstance(actions, (list, tuple)) and actions:
                if len(actions) == 1:
                    self._invoke_plugin_action(plugin, actions[0], ctx)
                    return
                menu = QtWidgets.QMenu(self)
                for desc in actions:
                    act = QtGui.QAction(str(desc.get("label", "Action")), self)
                    act.triggered.connect(lambda _=False, d=desc: self._invoke_plugin_action(plugin, d, ctx))
                    menu.addAction(act)
                pt = self.comboPlugins.mapToGlobal(QtCore.QPoint(0, self.comboPlugins.height()))
                menu.exec(pt)
                return

            # 3) entry-point comuni del plugin (metodi d'istanza)
            for attr in ("run", "apply", "open", "open_dialog", "show", "__call__"):
                fn = getattr(plugin, attr, None)
                if callable(fn) and _call_safe(fn):
                    return

            # 4) modulo con funzioni globali
            import types
            if isinstance(plugin, types.ModuleType):
                for attr in ("run", "main"):
                    fn = getattr(plugin, attr, None)
                    if callable(fn) and _call_safe(fn):
                        return

            QtWidgets.QMessageBox.information(self, "Plugin", f"Plugin '{key}' does not expose any known actions.")
        except Exception as ex:
            QtWidgets.QMessageBox.critical(self, "Plugin error", str(ex))

    def _invoke_plugin_action(self, plugin, action_desc, ctx: dict) -> None:
        """Executes a single plugin action described as a dictionary:
           {'label': 'Do X', 'slot': callable} or {'label': ..., 'method': 'run'}.
        """
        try:
            slot = action_desc.get("slot")
            if callable(slot):
                # tenta (ctx) se il callable accetta argomenti
                try:
                    slot(ctx)
                except TypeError:
                    slot()
                return
            method_name = action_desc.get("method") or action_desc.get("name")
            if method_name and hasattr(plugin, method_name):
                fn = getattr(plugin, method_name)
                try:
                    fn(ctx)
                except TypeError:
                    fn()
                return
            # fallback: se c'è 'command' stringa e il plugin ha un dispatcher
            cmd = action_desc.get("command")
            if cmd and hasattr(plugin, "dispatch"):
                plugin.dispatch(cmd, ctx)
                return
            raise RuntimeError("Unsupported action descriptor")
        except Exception as ex:
            QtWidgets.QMessageBox.critical(self, "Plugin action error", str(ex))

    def _rebuild_plugins_menu(self, items: list[dict]) -> None:
        """Rigenera il menù &Plugins con le azioni dei plugin."""
        if not hasattr(self, "m_plugins") or self.m_plugins is None:
            return
        self.m_plugins.clear()
        if not items:
            act = QtGui.QAction("No plugins installed", self)
            act.setEnabled(False)
            self.m_plugins.addAction(act)
            return

        for it in items:
            key = it.get("key")
            label = it.get("label", key or "Plugin")
            tooltip = it.get("tooltip", "")
            enabled = bool(it.get("enabled", True))

            submenu = QtWidgets.QMenu(label, self.m_plugins)
            submenu.setEnabled(enabled)
            if tooltip:
                submenu.setToolTipsVisible(True)
                submenu.setToolTip(tooltip)

            # prova a ottenere il plugin e le sue azioni
            plugin = None
            get_fn = getattr(self.plugin_manager, "get", None)
            if callable(get_fn):
                try:
                    plugin = get_fn(key)
                except Exception:
                    plugin = None
            actions = None
            if plugin is not None:
                for attr in ("actions", "get_actions"):
                    getter = getattr(plugin, attr, None)
                    if callable(getter):
                        try:
                            actions = getter()
                        except Exception:
                            actions = None
                        break

            if isinstance(actions, (list, tuple)) and actions:
                # crea QAction per ciascuna azione
                for a in actions:
                    q = QtGui.QAction(str(a.get("label", "Action")), self)
                    q.setToolTip(str(a.get("tooltip", "")))
                    q.triggered.connect(lambda _=False, plug=plugin, desc=a: self._invoke_plugin_action(plug, desc, self._plugin_context()))
                    submenu.addAction(q)
            else:
                # azione di default: Run <label>
                run_act = QtGui.QAction(f"Run {label}", self)
                run_act.setToolTip("Execute default entry-point")
                run_act.triggered.connect(lambda _=False, k=key: self._run_plugin_by_key(k))
                submenu.addAction(run_act)

            self.m_plugins.addMenu(submenu)


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
        # --- Start progress bar immediately (UI feedback before heavy I/O) ---
        self._import_progress_begin("Opening file…")
        try:
            # Force the UI to repaint the progress bar before blocking I/O
            QtWidgets.QApplication.processEvents()
        except Exception:
            pass

        # Import
        from .utils.io.importers import import_file
        from .ui.import_summary_dialog import ImportSummaryDialog

        try:
            # Prefer importer with a progress callback (newer versions)
            try:
                objects = import_file(
                    path,
                    progress_cb=lambda p=None, msg=None: self._import_progress_update(
                        p if p is not None else self.progress.value(),
                        msg if msg is not None else "Reading…"
                    )
                )
            except TypeError:
                # Fallback: older importer without progress_cb
                objects = import_file(path)

            # Give a final nudge to the bar/format before showing the summary
            try:
                self._import_progress_update(100, "Parsing complete")
                QtWidgets.QApplication.processEvents()
            except Exception:
                pass

        except Exception as ex:
            # Make sure progress ends even on error
            self._import_progress_end()
            QtWidgets.QMessageBox.critical(self, "Import error", str(ex))
            return

        summary = ImportSummaryDialog(objects, self)
        if summary.exec() != QtWidgets.QDialog.Accepted:
            self._import_progress_end()
            return

        # Dopo aver letto le operazioni scelte dall’utente:
        ops = summary.operations()
        self._import_progress_update(45, "Applying options (axis / normals / budget)…")

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

        def _compute_normals_for_points(P, k=16, subset=80000, fast=True):
            """Compute PCA normals for an (N,3) numpy array `P` without touching the viewer.

            This avoids creating temporary datasets that would shift `_datasets` indices.
            """
            import numpy as _np
            from numpy.linalg import eigh as _eigh

            P = _np.asarray(P, dtype=float)
            if P.ndim != 2 or P.shape[1] != 3 or P.shape[0] == 0:
                return None
            n = P.shape[0]
            k = int(max(3, min(k, n)))

            # Try fast subset + nearest propagation when large and SciPy is available
            if fast and n > max(10000, subset):
                try:
                    from scipy.spatial import cKDTree as _KD  # type: ignore
                    rng = _np.random.default_rng(42)
                    idx_sub = rng.choice(n, size=subset, replace=False)
                    Psub = P[idx_sub]

                    tree_sub = _KD(Psub)
                    # compute normals on subset
                    Nsub = _np.empty_like(Psub)
                    # kNN within subset
                    _, knn_idx = tree_sub.query(Psub, k=min(k, Psub.shape[0]))
                    if knn_idx.ndim == 1:
                        knn_idx = knn_idx[:, None]
                    for i in range(Psub.shape[0]):
                        nbrs = Psub[knn_idx[i]]
                        C = _np.cov(nbrs.T)
                        w, v = _eigh(C)
                        nrm = v[:, 0]
                        if _np.dot(nrm, Psub[i] - nbrs.mean(axis=0)) < 0:
                            nrm = -nrm
                        Nsub[i] = nrm
                    # propagate to full set
                    tree_full = _KD(Psub)
                    _, j = tree_full.query(P, k=1)
                    return Nsub[j]
                except Exception:
                    # fall back to full computation
                    pass

            # Full PCA normals (no SciPy dependency)
            try:
                # Brute-force kNN; for large N you may replace with a KD-tree if available
                N = _np.empty_like(P)
                for i in range(n):
                    d2 = _np.sum((P - P[i]) ** 2, axis=1)
                    sel = _np.argpartition(d2, kth=k-1)[:k]
                    nbrs = P[sel]
                    C = _np.cov(nbrs.T)
                    w, v = _eigh(C)
                    nrm = v[:, 0]
                    if _np.dot(nrm, P[i] - nbrs.mean(axis=0)) < 0:
                        nrm = -nrm
                    N[i] = nrm
                return N
            except Exception:
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
                    n = _compute_normals_for_points(obj.points, k=int(getattr(self, "normals_k", 16)))
                    if n is not None:
                        obj.normals = n
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
                cap = 2_200_000 if points_as_spheres else 8_000_000

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

        self._import_progress_update(65, "Adding objects to the scene…")

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
                ds_index = self.viewer3d.add_points(
                    obj.points, obj.colors, getattr(obj, "normals", None)
                )
            elif obj.kind == "mesh" and obj.pv_mesh is not None:
                ds_index = self.viewer3d.add_pyvista_mesh(obj.pv_mesh)
            
            try:
                self._reapply_overlays_safe()
            except Exception:
                pass
            
            # Tree: hierarchical, checkable, with metadata.
            self.treeMCTS.blockSignals(True)
            root = QtWidgets.QTreeWidgetItem([obj.name])
            root.setFlags(
                root.flags()
                | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                | QtCore.Qt.ItemFlag.ItemIsAutoTristate
            )
            root.setCheckState(0, QtCore.Qt.CheckState.Checked)
            self.treeMCTS.addTopLevelItem(root)

            if obj.kind == "points":
                # Point cloud child
                it_points = QtWidgets.QTreeWidgetItem(["Point cloud"])
                it_points.setFlags(
                    it_points.flags()
                    | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                    # | QtCore.Qt.ItemFlag.ItemIsAutoTristate
                )
                it_points.setCheckState(0, QtCore.Qt.CheckState.Checked)
                it_points.setData(0, QtCore.Qt.ItemDataRole.UserRole, {"kind": "points", "ds": ds_index})
                root.addChild(it_points)
                # Normals child (if available)
                if getattr(obj, "normals", None) is not None:
                    it_normals = QtWidgets.QTreeWidgetItem(["Normals"])
                    it_normals.setFlags(
                        it_normals.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                    )
                    it_normals.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
                    it_normals.setData(0, QtCore.Qt.ItemDataRole.UserRole, {"kind": "normals", "ds": ds_index})
                    it_points.addChild(it_normals)
            else:
                # Mesh node
                it_mesh = QtWidgets.QTreeWidgetItem(["Mesh"])
                it_mesh.setFlags(it_mesh.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                it_mesh.setCheckState(0, QtCore.Qt.CheckState.Checked)
                it_mesh.setData(0, QtCore.Qt.ItemDataRole.UserRole, {"kind": "mesh", "ds": ds_index})
                root.addChild(it_mesh)

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
                    "ds_index": ds_index,
                    "source_path": path,  # keep original file path for session reopen
                }
                if obj.kind == "points":
                    entry.update(
                        {
                            "point_size": getattr(self.viewer3d, "_point_size", 3),
                            "point_budget": getattr(
                                self.viewer3d, "_view_budget_percent", 100
                            ),
                            "color_mode": getattr(self.viewer3d, "_color_mode", "Normal RGB"),
                            "colormap": getattr(self.viewer3d, "_cmap", "viridis"),
                            "solid_color": self.viewer3d._datasets[ds_index].get(
                                "solid_color", (1.0, 1.0, 1.0)
                            ),
                            "points_as_spheres": getattr(
                                self.viewer3d, "_points_as_spheres", False
                            ),
                        }
                    )
                else:
                    entry.update(
                        {
                            "representation": "Surface",
                            "opacity": 100,
                            "solid_color": (1.0, 1.0, 1.0),
                        }
                    )
                self.mcts[obj.name] = entry
                self.mct = entry
                # Always select and check the "Point cloud" node if it exists
                selected = root
                for i in range(root.childCount()):
                    child = root.child(i)
                    data = child.data(0, QtCore.Qt.ItemDataRole.UserRole)
                    if isinstance(data, dict) and data.get("kind") == "points":
                        child.setCheckState(0, QtCore.Qt.CheckState.Checked)
                        selected = child
                        break
                try:
                    self.treeMCTS.setCurrentItem(selected)
                except Exception:
                    pass
            except Exception:
                pass

            self.statusBar().showMessage(f"Imported {len(objects)} object(s) from {path}", 5000)
            # 
            self._refresh_tree_visibility()
            try:
                self._reapply_overlays_safe()
            except Exception:
                pass
            self._import_progress_update(100, "Import finished")
            self._import_progress_end()
            try:
                if self.progress.value() < 100:
                    self._import_progress_end()
            except Exception:
                pass


    def _refresh_tree_visibility(self) -> None:
        """Sync visibility of all datasets from the tree using 'effective checked' (node and all ancestors)."""
        if self.treeMCTS.topLevelItemCount() == 0:
            return

        def recurse(node: QtWidgets.QTreeWidgetItem) -> None:
            data = node.data(0, QtCore.Qt.ItemDataRole.UserRole)
            eff_on = self._is_effectively_checked(node)
            if isinstance(data, dict) and data.get("ds") is not None:
                ds = int(data.get("ds"))
                kind = data.get("kind") or "points"
                # If this is the "Normals" child node, toggle normals only.
                if node.text(0) == "Normals" or data.get("normals_node", False):
                    try:
                        getattr(self.viewer3d, "set_normals_visibility", lambda *_: None)(ds, bool(eff_on))
                    except Exception:
                        pass
                else:
                    self._viewer_set_visibility(kind, ds, bool(eff_on))
                    # Safety: if points are off, ensure normals off too
                    if kind == "points" and not eff_on:
                        try:
                            getattr(self.viewer3d, "set_normals_visibility", lambda *_: None)(ds, False)
                        except Exception:
                            pass

            for i in range(node.childCount()):
                recurse(node.child(i))

        self._tree_updating = True
        try:
            for i in range(self.treeMCTS.topLevelItemCount()):
                recurse(self.treeMCTS.topLevelItem(i))
        finally:
            self._tree_updating = False
                    

    def _on_tree_item_changed(self, item: QtWidgets.QTreeWidgetItem) -> None:
        """
        TEMP: do not propagate check state between parent/children.
        Simply schedule a single full-scene rebuild from current tree state.
        """
        # Evita rientranze/ricorsioni
        if getattr(self, "_tree_updating", False):
            return
        # Assicurati non sia tristate; non toccare parent/children
        try:
            self._tree_updating = True
            item.setFlags((item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable) & ~QtCore.Qt.ItemFlag.ItemIsTristate)
        except Exception:
            pass
        finally:
            self._tree_updating = False
        # Debounced full-scene rebuild
        self._schedule_scene_rebuild()

    def _update_parent_checkstate(self, child: QtWidgets.QTreeWidgetItem) -> None:
        """
        Update ancestors according to these rules:
        - If a parent has a 'points' child, the parent's state follows ONLY the state of 'points'.
        - Otherwise: Checked if all children are Checked; Unchecked if all are Unchecked; otherwise PartiallyChecked.
        - Does NOT modify children (no downward propagation here).
        """
        if child is None:
            return

        self._tree_updating = True
        try:
            parent = child.parent()
            while parent is not None:
                # 1) prova la regola "point-centric"
                points_child = None
                for i in range(parent.childCount()):
                    c = parent.child(i)
                    data = c.data(0, QtCore.Qt.ItemDataRole.UserRole)
                    if isinstance(data, dict) and data.get("kind") == "points":
                        points_child = c
                        break

                if points_child is not None:
                    # Il parent segue SOLO lo stato del figlio "points"
                    parent.setCheckState(0, points_child.checkState(0))
                else:
                    # 2) fallback: tri-stato classico basato su tutti i figli
                    total = parent.childCount()
                    if total == 0:
                        break
                    checked = 0
                    unchecked = 0
                    for i in range(total):
                        st = parent.child(i).checkState(0)
                        if st == QtCore.Qt.CheckState.Checked:
                            checked += 1
                        elif st == QtCore.Qt.CheckState.Unchecked:
                            unchecked += 1
                    if checked == total:
                        parent.setCheckState(0, QtCore.Qt.CheckState.Checked)
                    elif unchecked == total:
                        parent.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
                    else:
                        parent.setCheckState(0, QtCore.Qt.CheckState.PartiallyChecked)

                parent = parent.parent()
        finally:
            self._tree_updating = False

    def _uncheck_descendants(self, item: QtWidgets.QTreeWidgetItem) -> None:
        """Spegni solo i discendenti (non accende nulla)."""
        for i in range(item.childCount()):
            child = item.child(i)
            if child.checkState(0) != QtCore.Qt.CheckState.Unchecked:
                child.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
            self._uncheck_descendants(child)

    def _set_descendant_checkstate(
        self, item: QtWidgets.QTreeWidgetItem, state: QtCore.Qt.CheckState
        ) -> None:
        """Set the check state of all descendants."""
        for i in range(item.childCount()):
            child = item.child(i)
            child.setCheckState(0, state)
            self._set_descendant_checkstate(child, state)

    # def _is_effectively_checked(self, item: QtWidgets.QTreeWidgetItem) -> bool:
    #     """
    #     Visible if *this* item is Checked and no ancestor is Unchecked.

    #     - The current item must be Qt.Checked.
    #     - Ancestors with Qt.PartiallyChecked do NOT block their children.
    #     - An ancestor with Qt.Unchecked disables all its descendants.
    #     """
    #     if item is None or item.checkState(0) != QtCore.Qt.CheckState.Checked:
    #         return False
    #     parent = item.parent()
    #     while parent is not None:
    #         if parent.checkState(0) == QtCore.Qt.CheckState.Unchecked:
    #             return False
    #         parent = parent.parent()
    #     return True
    def _is_effectively_checked(self, item: QtWidgets.QTreeWidgetItem) -> bool:
        """
        Return True if `item` and **all** its ancestors are checked.
        This ensures children cannot remain logically 'on' if a parent is 'off'.
        """
        cur = item
        while cur is not None:
            try:
                if cur.checkState(0) != QtCore.Qt.CheckState.Checked:
                    return False
            except Exception:
                return False
            cur = cur.parent()
        return True
    
    def _on_tree_context_menu(self, pos: QtCore.QPoint) -> None:
        item = self.treeMCTS.itemAt(pos)
        if item is None:
            return
        data = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
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
                ds_index = self.viewer3d.add_pyvista_mesh(obj.pv_mesh)
            else:
                ds_index = None

            try:
                self._reapply_overlays_safe()
            except Exception:
                pass
            try:
                if prev_mode is not None:
                    self.viewer3d.set_color_mode(prev_mode)
            except Exception:
                pass

            # 
            # Build the tree entries (same as interactive import)
            self.treeMCTS.blockSignals(True)
            root = QtWidgets.QTreeWidgetItem([obj.name])
            root.setFlags(
                root.flags()
                | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                | QtCore.Qt.ItemFlag.ItemIsAutoTristate
            )
            root.setCheckState(0, QtCore.Qt.CheckState.Checked)
            self.treeMCTS.addTopLevelItem(root)

            if obj.kind == "points":
                it_points = QtWidgets.QTreeWidgetItem(["Point cloud"])
                it_points.setFlags(
                    it_points.flags()
                    | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                    # | QtCore.Qt.ItemFlag.ItemIsAutoTristate
                )
                it_points.setCheckState(0, QtCore.Qt.CheckState.Checked)
                it_points.setData(0, QtCore.Qt.ItemDataRole.UserRole, {"kind": "points", "ds": ds_index})
                root.addChild(it_points)

                if getattr(obj, "normals", None) is not None:
                    it_normals = QtWidgets.QTreeWidgetItem(["Normals"])
                    it_normals.setFlags(
                        it_normals.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                    )
                    it_normals.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
                    it_normals.setData(0, QtCore.Qt.ItemDataRole.UserRole, {"kind": "normals", "ds": ds_index})
                    it_points.addChild(it_normals)
            else:
                it_mesh = QtWidgets.QTreeWidgetItem(["Mesh"])
                it_mesh.setFlags(it_mesh.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                it_mesh.setCheckState(0, QtCore.Qt.CheckState.Checked)
                it_mesh.setData(0, QtCore.Qt.ItemDataRole.UserRole, {"kind": "mesh", "ds": ds_index})
                root.addChild(it_mesh)

            self.treeMCTS.blockSignals(False)
            # 
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
    
    # ----------------------- Normals: UI helpers ----------------------------
    def _invoke_set_progress_value(self, v: int) -> None:
        try:
            val = int(v)
            QtCore.QTimer.singleShot(0, lambda: self.progress.setValue(val))
        except Exception:
            pass

    def _invoke_set_progress_format(self, text: str) -> None:
        try:
            msg = str(text)
            QtCore.QTimer.singleShot(0, lambda: self.progress.setFormat(msg))
        except Exception:
            pass

    def _invoke_append_message(self, text: str) -> None:
        try:
            msg = str(text)
            QtCore.QTimer.singleShot(0, lambda: self.txtMessages.appendPlainText(msg))
        except Exception:
            pass

    def _append_message(self, text: str) -> None:
        self._invoke_append_message(text)

    # def _progress_start(self, text: str) -> None:
    #     try:
    #         self.progress.setRange(0, 100)
    #         self._invoke_set_progress_value(0)
    #         self._invoke_set_progress_format(text)
    #     except Exception:
    #         pass

    def _progress_update(self, value: int, text: Optional[str] = None) -> None:
        try:
            v = max(0, min(100, int(value)))
            self._invoke_set_progress_value(v)
            if text is not None:
                self._invoke_set_progress_format(text)
        except Exception:
            pass

    # def _progress_finish(self, text: str) -> None:
    #     try:
    #         self._invoke_set_progress_value(100)
    #         self._invoke_set_progress_format(text)
    #     except Exception:
    #         pass

    # @QtCore.Slot(int)
    # def _slot_worker_progress(self, pct: int) -> None:
    #     try:
    #         txt = getattr(self, "_last_progress_text", "")
    #         v = max(0, min(100, int(pct)))
    #         self._progress_update(v, txt)
    #     except Exception:
    #         pass

    # @QtCore.Slot(str)
    # def _slot_worker_message(self, msg: str) -> None:
    #     try:
    #         self._last_progress_text = str(msg)
    #         self._invoke_set_progress_format(self._last_progress_text)
    #         self._invoke_append_message(self._last_progress_text)
    #     except Exception:
    #         pass

    # def _ensure_cancel_button(self) -> None:
    #     """Enable the CANCEL button for an active job."""
    #     try:
    #         self.btnCancel.setEnabled(True)
    #     except Exception:
    #         pass

    # def _on_cancel_job(self) -> None:
    #     """Request cancellation of the active job, if supported by the worker."""
    #     job = getattr(self, "_active_job", None)
    #     if not job:
    #         return
    #     worker = job.get("worker")
    #     if hasattr(worker, "request_cancel"):
    #         worker.request_cancel()
    #     self._append_message("[Job] Cancel requested by user.")

    def _ensure_normals_tree_child(self, ds_index: int) -> None:
        """Ensure a 'Normals' child exists under the current file node for the active dataset."""
        try:
            item = self.treeMCTS.currentItem()
            if item is None:
                return
            # Ascend to root file node
            root = item
            while root.parent() is not None:
                root = root.parent()
            # Look for a child labeled 'Point cloud' or existing 'Normals'
            target = None
            for i in range(root.childCount()):
                c = root.child(i)
                if c.text(0) == "Point cloud":
                    target = c
                if c.text(0) == "Normals":
                    return  # already present at root level (older structure)
            if target is None:
                # Create the 'Point cloud' node if missing
                target = QtWidgets.QTreeWidgetItem(["Point cloud"])
                target.setFlags(target.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsAutoTristate)
                target.setCheckState(0, QtCore.Qt.CheckState.Checked)
                target.setData(0, QtCore.Qt.ItemDataRole.UserRole, {"kind": "points", "ds": ds_index})
                root.addChild(target)
            # Add Normals child if not present
            for i in range(target.childCount()):
                if target.child(i).text(0) == "Normals":
                    return
            it_normals = QtWidgets.QTreeWidgetItem(["Normals"])
            it_normals.setFlags(it_normals.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            it_normals.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
            it_normals.setData(0, QtCore.Qt.ItemDataRole.UserRole, {"kind": "normals", "ds": ds_index})
            target.addChild(it_normals)
        except Exception:
            pass
    

    # --------- Normals: helpers ---------------------------------------------

    def _current_ds_index(self) -> int | None:
        """Return the current dataset index or None if nothing is selected."""
        try:
            ds = self._current_dataset_index()
            return int(ds) if ds is not None else None
        except Exception:
            return None

    def _ensure_normals_visible(self, ds: int) -> None:
        """Ensure normals actor exists/visible for dataset ds before applying edits."""
        v3d = self.viewer3d
        # Prova API moderna
        set_vis = getattr(v3d, "set_normals_visibility", None)
        if callable(set_vis):
            set_vis(ds, True)
            # Sincronizza anche il toggle della toolbar se esiste
            try:
                self.act_toggle_normals.blockSignals(True)
                self.act_toggle_normals.setChecked(True)
                self.act_toggle_normals.blockSignals(False)
            except Exception:
                pass
            return
        # Fallback: prova a ricostruire direttamente
        rb = getattr(v3d, "_rebuild_normals_actor", None)
        if callable(rb):
            rb(ds)
        try:
            # best effort: attivalo come “visibile” nello stato locale
            rec = v3d._datasets[ds]
            rec["normals_visible"] = True
        except Exception:
            pass

    def _apply_normals_rebuild(self, ds: int) -> None:
        """Chiama il rebuild con i parametri correnti del dataset."""
        v3d = self.viewer3d
        # Se la API granulari esistono, non serve forzare il rebuild manuale
        rb = getattr(v3d, "_rebuild_normals_actor", None)
        if callable(rb):
            # Recupera parametri correnti (con fallback a default)
            try:
                rec = v3d._datasets[ds]
                style = str(rec.get("normals_style", getattr(v3d, "_normals_style", "Uniform")))
                color = tuple(rec.get("normals_color", getattr(v3d, "_normals_color", (1.0, 0.2, 0.2))))
                percent = int(rec.get("normals_percent", getattr(v3d, "_normals_percent", 1)))
                scale = int(rec.get("normals_scale", getattr(v3d, "_normals_scale", 20)))
            except Exception:
                style, color, percent, scale = "Uniform", (1.0, 0.2, 0.2), 50, 20
            rb(ds, style=style, color=color, percent=percent, scale=scale)

    # --------- Normals: handlers from DisplayPanel --------------------------

    # ------------------------------
    # Normals display live updates
    # ------------------------------
    # ----------------------- Normals: handlers ----------------------------
    def _on_normals_style_changed(self, mode: str) -> None:
        """
        Change the visualization style of normals for the currently selected dataset.

        Args:
            mode (str): The style mode to apply. Options include:
                - 'Uniform': Uniform color for all normals.
                - 'Axis RGB': Color normals based on their axis orientation.
                - 'RGB Components': Color normals based on their RGB components.
        """
        ds = self._current_dataset_index()
        if ds is None:
            return
        # Update the internal MCT (Metadata Context Table) state if available
        try:
            if self.mct is not None:
                self.mct["normals_style"] = mode
        except Exception:
            pass
        # Attempt to use the viewer's public API; if unavailable, fallback to rebuilding
        try:
            fn = getattr(self.viewer3d, "set_normals_style", None)
            if callable(fn):
                fn(ds, mode)
            else:
                # Fallback: force a rebuild with the new parameters while maintaining current visibility
                self._apply_normals_update(ds, style=mode)
        except Exception:
            pass

    def _on_normals_color_changed(self, col: QtGui.QColor) -> None:
        """
        Change the uniform color of normals. This is only applicable if the style is set to 'Uniform'.

        Args:
            col (QtGui.QColor): The new color to apply.
        """
        if col is None or not col.isValid():
            return
        ds = self._current_dataset_index()
        if ds is None:
            return
        rgb = (col.red(), col.green(), col.blue())
        # Update the internal MCT state if available
        try:
            if self.mct is not None:
                self.mct["normals_color"] = rgb
        except Exception:
            pass
        # Attempt to use the viewer's public API; if unavailable, fallback to rebuilding
        try:
            fn = getattr(self.viewer3d, "set_normals_color", None)
            if callable(fn):
                fn(ds, *rgb)
            else:
                self._apply_normals_update(ds, color=rgb)
        except Exception:
            pass

    def _on_normals_percent_changed(self, percent: int) -> None:
        """
        Change the percentage of normals displayed for the currently selected dataset.

        Args:
            percent (int): The percentage of normals to display (1 to 100).
        """
        ds = self._current_dataset_index()
        if ds is None:
            return
        p = int(max(1, min(100, percent)))  # Clamp the value between 1 and 100
        # Update the internal MCT state if available
        try:
            if self.mct is not None:
                self.mct["normals_percent"] = p
        except Exception:
            pass
        # Attempt to use the viewer's public API; if unavailable, fallback to rebuilding
        try:
            fn = getattr(self.viewer3d, "set_normals_percent", None)
            if callable(fn):
                fn(ds, p)
            else:
                self._apply_normals_update(ds, percent=p)
        except Exception:
            pass

    def _on_normals_scale_changed(self, scale: int) -> None:
        """
        Change the scale (vector size) of normals for the currently selected dataset.

        Args:
            scale (int): The scale factor for normals. Valid range is 1 to 200.
        """
        ds = self._current_dataset_index()
        if ds is None:
            return
        s = int(max(1, min(200, scale)))  # Clamp the value between 1 and 200
        # Update the internal MCT state if available
        try:
            if self.mct is not None:
                self.mct["normals_scale"] = s
        except Exception:
            pass
        # Attempt to use the viewer's public API; if unavailable, fallback to rebuilding
        try:
            fn = getattr(self.viewer3d, "set_normals_scale", None)
            if callable(fn):
                fn(ds, s)
            else:
                self._apply_normals_update(ds, scale=s)
        except Exception:
            pass

    # Helper: apply changes to normals by rebuilding the glyph actor if necessary
    def _apply_normals_update(self, ds: int, *, style: str | None = None,
                              color: tuple[int, int, int] | None = None,
                              percent: int | None = None,
                              scale: int | None = None) -> None:
        """
        Updates the normals parameters in the viewer's dataset record and forces the
        reconstruction of the glyph actor while maintaining the current visibility state.

        Args:
            ds (int): The dataset index to update.
            style (str | None): The visualization style for normals (e.g., 'Uniform', 'Axis RGB').
            color (tuple[int, int, int] | None): The RGB color for normals, as integers in the range 0-255.
            percent (int | None): The percentage of normals to display (1 to 100).
            scale (int | None): The scale factor for normals (1 to 200).
        """
        try:
            recs = getattr(self.viewer3d, "_datasets", [])
            if not (0 <= ds < len(recs)):
                return
            rec = recs[ds]
            # Update per-dataset state
            if style is not None:
                rec["normals_style"] = style
            if color is not None:
                # Normalize color to float 0..1 if the viewer expects it, otherwise keep 0..255
                try:
                    rec["normals_color"] = tuple(float(c)/255.0 for c in color)
                except Exception:
                    rec["normals_color"] = color
            if percent is not None:
                rec["normals_percent"] = int(max(1, min(100, percent)))
            if scale is not None:
                rec["normals_scale"] = int(max(1, min(200, scale)))

            # If normals are currently visible, rebuild the actor; otherwise, do nothing
            # (the actor will be rebuilt when toggled ON).
            visible = bool(rec.get("normals_visible", False))
            if visible:
                # Prefer public API if it exists
                rb = getattr(self.viewer3d, "_rebuild_normals_actor", None)
                if callable(rb):
                    rb(
                        ds,
                        style=str(rec.get("normals_style", "Axis RGB")),
                        color=tuple(rec.get("normals_color", (0.9, 0.9, 0.2))),
                        percent=int(rec.get("normals_percent", 1)),
                        scale=int(rec.get("normals_scale", 20)),
                    )
                else:
                    # As a fallback, force set_normals_visibility(True), which internally triggers a rebuild
                    getattr(self.viewer3d, "set_normals_visibility", lambda *_: None)(ds, True)
            # Perform a lightweight refresh of the viewer
            try:
                self.viewer3d.refresh()
            except Exception:
                pass
        except Exception:
            pass
    

    def _on_fast_normals_toggled(self, enabled: bool) -> None:
        """
        Persist the user's preference for 'Fast normals' in the window state.

        Args:
            enabled (bool): Whether the 'Fast normals' mode is enabled or not.
        """
        try:
            self.normals_fast_enabled = bool(enabled)
        except Exception:
            pass
    # ------------------------------------------------------------------
    # ViewSettingsDialog: edit 3D viewer settings (background, grid, colorbar, points style)
    # ------------------------------------------------------------------
    class ViewSettingsDialog(QtWidgets.QDialog):
        """Dialog to customize 3D view preferences (background, grid, colorbar, points style)."""
        def __init__(self, parent=None, state: dict | None = None):
            super().__init__(parent)
            self.setWindowTitle("3D View Settings")
            self.setModal(True)
            self._state = dict(state or {})

            lay = QtWidgets.QVBoxLayout(self)

            # Background color picker
            row_bg = QtWidgets.QHBoxLayout()
            row_bg.addWidget(QtWidgets.QLabel("Background:"))
            self.btnBg = QtWidgets.QPushButton()
            self.btnBg.setFixedWidth(120)
            self.btnBg.clicked.connect(self._pick_bg)
            row_bg.addWidget(self.btnBg)
            row_bg.addStretch(1)
            lay.addLayout(row_bg)

            # Grid + points style
            self.chkGrid = QtWidgets.QCheckBox("Show grid")
            self.chkPtsSpheres = QtWidgets.QCheckBox("Render points as spheres")
            lay.addWidget(self.chkGrid)
            lay.addWidget(self.chkPtsSpheres)

            # Colorbar controls
            gb = QtWidgets.QGroupBox("Colorbar")
            gl = QtWidgets.QFormLayout(gb)
            self.cmbBar = QtWidgets.QComboBox()
            self.cmbBar.addItems(["Hidden", "Horizontal (bottom-right)", "Vertical (top-right)"])
            self.edBarTitle = QtWidgets.QLineEdit()
            gl.addRow("Mode:", self.cmbBar)
            gl.addRow("Title:", self.edBarTitle)
            lay.addWidget(gb)

            # Buttons
            btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Apply)
            btns.accepted.connect(self.accept)
            btns.rejected.connect(self.reject)
            btns.button(QtWidgets.QDialogButtonBox.Apply).clicked.connect(self._apply_only)
            lay.addWidget(btns)

            self._load_state()

        def _load_state(self):
            """Populate widgets from the provided state dict."""
            bg = tuple(self._state.get("bg", (30, 30, 30)))
            self._set_btn_bg(bg)
            self.chkGrid.setChecked(bool(self._state.get("grid", True)))
            self.chkPtsSpheres.setChecked(bool(self._state.get("points_as_spheres", True)))
            mode = str(self._state.get("colorbar_mode", "vertical-tr"))
            idx = {"hidden":0, "horizontal-br":1, "vertical-tr":2}.get(mode, 2)
            self.cmbBar.setCurrentIndex(idx)
            self.edBarTitle.setText(str(self._state.get("colorbar_title", "")))

        def _set_btn_bg(self, rgb: tuple[int, int, int]):
            """Update the background button swatch."""
            try:
                r, g, b = map(int, rgb)
                self.btnBg.setText(f"RGB {r},{g},{b}")
                self.btnBg.setStyleSheet(f"background-color: rgb({r},{g},{b}); color: white;")
            except Exception:
                pass

        def _pick_bg(self):
            """Open a QColorDialog to pick a background color."""
            try:
                c0 = QtGui.QColor(*self._state.get("bg", (30, 30, 30)))
                col = QtWidgets.QColorDialog.getColor(c0, self, "Pick background color")
                if col.isValid():
                    self._state["bg"] = (col.red(), col.green(), col.blue())
                    self._set_btn_bg(self._state["bg"])
            except Exception:
                pass

        def values(self) -> dict:
            """Return current dialog values as a dict."""
            mode_idx = int(self.cmbBar.currentIndex())
            mode = {0:"hidden", 1:"horizontal-br", 2:"vertical-tr"}.get(mode_idx, "vertical-tr")
            return {
                "bg": tuple(self._state.get("bg", (30,30,30))),
                "grid": bool(self.chkGrid.isChecked()),
                "points_as_spheres": bool(self.chkPtsSpheres.isChecked()),
                "colorbar_mode": mode,
                "colorbar_title": self.edBarTitle.text().strip(),
            }

        def _apply_only(self):
            self.done(2)  # custom code for Apply

    def _on_open_view_settings(self) -> None:
        """Open the 3D view settings dialog and apply changes (Apply/OK)."""
        st = dict(getattr(self, "_view_prefs", {}))
        dlg = self.ViewSettingsDialog(self, state=st)
        code = dlg.exec()

        def _apply(vals: dict):
            v3d = getattr(self, "viewer3d", None)
            if v3d is None:
                return
            # Background
            try:
                getattr(v3d, "set_background_color", lambda *_: None)(vals["bg"])  # (r,g,b)
            except Exception:
                pass
            # Grid
            try:
                on = bool(vals.get("grid", True))
                for name in ("set_grid_enabled", "set_grid_visible", "toggle_grid", "show_grid"):
                    fn = getattr(v3d, name, None)
                    if callable(fn):
                        try:
                            fn(on)
                            break
                        except Exception:
                            continue
            except Exception:
                pass
            # Points as spheres
            try:
                getattr(v3d, "set_points_as_spheres", lambda *_: None)(bool(vals.get("points_as_spheres", True)))
            except Exception:
                pass
            # Colorbar placement
            try:
                getattr(v3d, "set_colorbar_mode", lambda *_: None)(str(vals.get("colorbar_mode", "vertical-tr")), str(vals.get("colorbar_title", "")))
            except Exception:
                # Fallback: try vertical/hide helper if available
                mode = str(vals.get("colorbar_mode", "vertical-tr"))
                if mode == "hidden":
                    try:
                        v3d.set_colorbar_vertical(False)
                    except Exception:
                        pass
                elif mode == "vertical-tr":
                    try:
                        v3d.set_colorbar_vertical(True, str(vals.get("colorbar_title", "")))
                    except Exception:
                        pass
            
            try:
                v3d._apply_background(); v3d._apply_scalarbar()
            except Exception:
                pass
            # Keep a copy for next time
            try:
                self._view_prefs.update(vals)
            except Exception:
                pass
            try:
                v3d.refresh()
            except Exception:
                pass

        # Apply immediately for Apply/OK
        if code == 2 or code == QtWidgets.QDialog.Accepted:
            _apply(dlg.values())