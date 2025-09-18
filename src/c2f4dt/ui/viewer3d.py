from __future__ import annotations

from PySide6 import QtWidgets
import numpy as np
try:
    from scipy.spatial import cKDTree as KDTree
except Exception:
    KDTree = None

def _ensure_surface_mesh(mesh):
    """Return a PolyData suitable for surface rendering, extracting if needed.

    For UnstructuredGrid/StructuredGrid/ImageData, uses extract_surface(pass_pointid=True)
    so we can remap point-data to the surface using vtkOriginalPointIds.
    """
    try:
        import pyvista as pv
    except Exception:
        return mesh
    if mesh is None:
        return None
    if isinstance(mesh, pv.PolyData):
        return mesh
    try:
        if hasattr(mesh, "extract_surface"):
            surf = mesh.extract_surface(pass_pointid=True, pass_cellid=True)
            if isinstance(surf, pv.PolyData):
                return surf
    except Exception:
        pass
    try:
        if hasattr(mesh, "extract_geometry"):
            geo = mesh.extract_geometry()
            if isinstance(geo, pv.PolyData):
                return geo
    except Exception:
        pass
    return mesh


class Viewer3DPlaceholder(QtWidgets.QFrame):
    """Placeholder for the 3D viewer area.

    This placeholder keeps the UI functional when PyVista/VTK is not installed.
    All public methods are no-op to avoid AttributeError when actions are triggered.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)
        self._view_budget_percent: int = 100  # on-screen LOD budget, per-dataset

    # --- Camera helpers (no-op) ---
    def view_fit(self) -> None:
        """No-op fit view."""
        pass

    def view_axis(self, axis: str) -> None:
        """No-op axis view.

        Args:
            axis: One of "+X", "-X", "+Y", "-Y", "+Z", "-Z".
        """
        pass

    def view_iso(self, positive: bool = True) -> None:
        """No-op isometric view.

        Args:
            positive: If True use isometric+, otherwise isometric−.
        """
        pass

    def invert_view(self) -> None:
        """No-op invert camera direction."""
        pass

    def refresh(self) -> None:
        """No-op refresh renderer."""
        pass

    # --- Display controls (no-op) ---
    def set_point_size(self, size: int, dataset_index: int | None = None) -> None:
        """No-op set point size."""
        pass

    def set_point_budget(self, percent: int, dataset_index: int | None = None) -> None:
        """No-op set visible points percentage."""
        pass

    def set_color_mode(self, mode: str, dataset_index: int | None = None) -> None:
        """No-op set color mode."""
        pass

    def set_solid_color(self, r: int, g: int, b: int) -> None:
        """No-op set solid RGB color.

        Args:
            r: Red channel (0–255).
            g: Green channel (0–255).
            b: Blue channel (0–255).
        """
        self._solid_color = (r, g, b)
        try:
            self._solid_fallback = (r / 255.0, g / 255.0, b / 255.0)
        except Exception:
            self._solid_fallback = (0.78, 0.78, 0.78)
        self._refresh_datasets()

    def set_colormap(self, name: str, dataset_index: int | None = None) -> None:
        """No-op set colormap name."""
        pass

    def set_dataset_color(self, dataset_index: int, r: int, g: int, b: int) -> None:
        """Set per-dataset solid color and update actor if in Solid mode."""
        try:
            rec = self._datasets[dataset_index]
        except Exception:
            return
        rgb = (
            max(0, min(255, r)) / 255.0,
            max(0, min(255, g)) / 255.0,
            max(0, min(255, b)) / 255.0,
        )
        rec["solid_color"] = rgb
        if rec.get("kind") == "mesh":
            actor_m = rec.get("actor_mesh")
            if actor_m is not None:
                try:
                    actor_m.GetProperty().SetColor(rgb)
                    self.plotter.update()
                except Exception:
                    pass
            return
        mode = getattr(rec, "color_mode", getattr(self, "_color_mode", "Normal RGB")).lower()
        actor = rec.get("actor_points")
        if actor is not None and mode.startswith("solid"):
            try:
                actor.GetProperty().SetColor(rgb)
                self.plotter.update()
                return
            except Exception:
                pass
        # Fallback: ricrea l'attore
        try:
            if actor is not None:
                self.plotter.remove_actor(actor)
        except Exception:
            pass
        rec["actor_points"] = self._add_points_by_mode(
            rec.get("pdata"),
            rec.get("has_rgb", False),
            rec.get("solid_color", self._solid_fallback),
            rec.get("color_mode", self._color_mode),
            rec.get("cmap", self._cmap),
            rec.get("points_as_spheres", self._points_as_spheres),
        )
        self.plotter.update()

class Viewer3D(QtWidgets.QWidget):
    """PyVistaQt-based 3D viewer with camera helpers and display API.

    This widget wraps a ``pyvistaqt.QtInteractor`` when available. All public
    methods are guarded with try/except so that the UI remains responsive.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        # Lazy import so that the app can start without hard dependency at import-time.
        from pyvistaqt import QtInteractor  # type: ignore
        import pyvista as pv  # noqa: F401  (kept for future use)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter)
        # ---- macOS/Metal safety tweaks ----
        # These settings mitigate GPU driver recoveries seen on some Macs with Qt6+VTK.
        try:
            # Disable multisampling (anti-aliasing) at the render window level
            if hasattr(self.plotter, "render_window"):
                self.plotter.render_window.SetMultiSamples(0)

            # Conservative interactive update rates (throttle rendering)
            if hasattr(self.plotter, "iren"):
                try:
                    self.plotter.iren.SetDesiredUpdateRate(5.0)
                    self.plotter.iren.SetStillUpdateRate(0.5)
                except Exception:
                    pass

            # Disable depth peeling and FXAA which may stress Metal backend
            try:
                self.plotter.renderer.SetUseDepthPeeling(0)
                if hasattr(self.plotter.renderer, "SetUseFXAA"):
                    self.plotter.renderer.SetUseFXAA(False)
            except Exception:
                pass
        except Exception:
            pass
        # Initial rendering cosmetics (tracked so we can re-apply after clear)
        try:
            self._bg_rgb = (255, 255, 255)  # default background; updated by set_background_color()
            self._apply_background()
        except Exception:
            pass

        # Track overlay state so that 'clear()' doesn't permanently remove them
        self._axes_on: bool = True
        self._bounds_on: bool = True
        # Default PyVista bounds style; adjust as you like
        self._bounds_kwargs: dict = {"grid": "front", "location": "outer"}

        # Apply overlays once at startup
        try:
            self._apply_overlays()
        except Exception:
            pass

        # State for display options
        self._point_size = 3
        self._budget = 100
        self._color_mode = "Solid"
        self._solid_color = (200, 200, 200)
        self._cmap = getattr(self, "_cmap", "viridis")
        # ---- Normals (viewer defaults) ------------------------------------
        # Publicly controllable via setters; copied per-dataset at add time.
        # style: "Uniform" | "Axis RGB" | "RGB Components"
        self._normals_style: str = "Axis RGB"
        # Default uniform color in [0..1]
        self._normals_color: tuple[float, float, float] = (0.9, 0.9, 0.2)
        # Percentage of normals to visualize (1..100)
        self._normals_percent: int = 1
        # Scale slider 1..200 → glyph_len = diag * (scale / 1000.0)
        self._normals_scale: int = 20

        # Rendering style & datasets registry
        self._points_as_spheres = getattr(self, "_points_as_spheres", False)  # user reported better perf with raw points
        # Registry of datasets
        self._datasets = getattr(self, "_datasets", [])  # list of dicts: {"pdata": pv.PolyData, "actor_points": vtkActor|None, "actor_normals": vtkActor|None, "has_rgb": bool}
        # View-time LOD budget control
        self._view_budget_percent: int = 100  # global percent applied to visible datasets
        self._auto_budget_initialized: bool = False
        # Per-dataset solid color support
        self._default_palette = [
            (0.90, 0.36, 0.36),  # red-ish
            (0.39, 0.77, 0.36),  # green-ish
            (0.36, 0.53, 0.90),  # blue-ish
            (0.91, 0.72, 0.36),  # orange-ish
            (0.58, 0.39, 0.77),  # purple-ish
            (0.36, 0.77, 0.73),  # teal-ish
        ]
        # Fallback Solid (0–1) coerente con _solid_color (0–255)
        try:
            self._solid_fallback = (
                float(self._solid_color[0]) / 255.0,
                float(self._solid_color[1]) / 255.0,
                float(self._solid_color[2]) / 255.0,
            )
        except Exception:
            self._solid_fallback = (0.78, 0.78, 0.78)

        self.view_fit()

    def _target_visible_points(self) -> int:
        """Heuristic cap for total visible points based on platform/render style."""
        try:
            import sys
            if sys.platform == "darwin":
                return 2_000_000 if not self._points_as_spheres else 600_000
            return 4_000_000 if not self._points_as_spheres else 1_200_000
        except Exception:
            return 2_000_000

    def _rebalance_budget_across_datasets(self) -> None:
        """Adjust global percent so that total visible points stays under target cap."""
        try:
            total_full = 0
            for rec in self._datasets:
                if not rec.get("visible", True):
                    continue
                full = rec.get("full_pdata", rec.get("pdata"))
                if hasattr(full, "n_points"):
                    total_full += int(full.n_points)
            if total_full <= 0:
                return
            cap = self._target_visible_points()
            p = min(100, max(1, int(cap * 100 / total_full)))
            if p != self._view_budget_percent:
                self._view_budget_percent = p
        except Exception:
            pass
    
    # ---- Bounds helpers ----
    def _visible_bounds(self):
        """Compute combined bounds (xmin,xmax,ymin,ymax,zmin,zmax) of visible datasets.

        Returns:
            tuple[float,float,float,float,float,float] | None
        """
        try:
            import numpy as np
        except Exception:
            np = None

        xmin = ymin = zmin = float("inf")
        xmax = ymax = zmax = float("-inf")
        any_ok = False

        for rec in self._datasets:
            if not rec.get("visible", True):
                continue
            b = None
            kind = rec.get("kind", "points")
            try:
                if kind == "points":
                    pd = rec.get("full_pdata", rec.get("pdata"))
                    if pd is not None and hasattr(pd, "bounds"):
                        b = pd.bounds  # (xmin,xmax,ymin,ymax,zmin,zmax)
                elif kind == "mesh":
                    m = rec.get("mesh")
                    if m is not None and hasattr(m, "bounds"):
                        b = m.bounds
            except Exception:
                b = None

            if not b:
                continue

            try:
                xmin = min(xmin, float(b[0])); xmax = max(xmax, float(b[1]))
                ymin = min(ymin, float(b[2])); ymax = max(ymax, float(b[3]))
                zmin = min(zmin, float(b[4])); zmax = max(zmax, float(b[5]))
                any_ok = True
            except Exception:
                continue

        if not any_ok:
            return None

        # Guard against degenerate extents (make them at least epsilon wide)
        eps = 1e-9
        if xmax <= xmin: xmax = xmin + eps
        if ymax <= ymin: ymax = ymin + eps
        if zmax <= zmin: zmax = zmin + eps

        return (xmin, xmax, ymin, ymax, zmin, zmax)
    
    # ---- Camera helpers ----
    def view_fit(self) -> None:
        """Fit camera to the bounds of *visible* datasets only."""
        try:
            b = self._visible_bounds()
            if b is None:
                # Fallback: nothing visible → default reset
                try:
                    self.plotter.reset_camera()
                except Exception:
                    pass
                return

            ren = getattr(self.plotter, "renderer", None)
            if ren is not None and hasattr(ren, "ResetCamera"):
                # ResetCamera(xmin,xmax,ymin,ymax,zmin,zmax)
                ren.ResetCamera(b[0], b[1], b[2], b[3], b[4], b[5])
                try:
                    ren.ResetCameraClippingRange()
                except Exception:
                    pass
            else:
                # Fallback if renderer API is not available
                self.plotter.reset_camera()

            # Final render
            try:
                if hasattr(self.plotter, "render"):
                    self.plotter.render()
                else:
                    self.plotter.update()
            except Exception:
                pass
        except Exception:
            # Never raise from a UI callback
            pass

    def view_axis(self, axis: str) -> None:
        """Set camera to an orthographic axis view.

        Args:
            axis: One of "+X", "-X", "+Y", "-Y", "+Z", "-Z".
        """
        try:
            if axis == "+X":
                self.plotter.view_xz(negative=False)
            elif axis == "-X":
                self.plotter.view_xz(negative=True)
            elif axis == "+Y":
                self.plotter.view_yz(negative=False)
            elif axis == "-Y":
                self.plotter.view_yz(negative=True)
            elif axis == "+Z":
                self.plotter.view_xy(negative=False)
            elif axis == "-Z":
                self.plotter.view_xy(negative=True)
            # self.plotter.reset_camera()
        except Exception:
            pass

    def view_iso(self, positive: bool = True) -> None:
        """Set an isometric camera preset.

        Args:
            positive: If True, use a ( +1, +1, +1 ) view direction; if False use a mirrored variant.
        """
        try:
            if positive:
                # Keep PyVista's built-in for the “+” preset
                self.plotter.view_isometric()
                try:
                    self.plotter.reset_camera_clipping_range()
                except Exception:
                    pass
                # Render
                try:
                    self.plotter.render()
                except Exception:
                    pass
            else:
                # A mirrored isometric direction (you can change to (-1, +1, +1) if you prefer)
                self._set_camera_iso(direction=(-1.0, -1.0, 1.0))
        except Exception:
            pass

    def _scene_center_radius(self):
        """Return (center_xyz, diag_len) computed from visible bounds.

        Falls back to plotter bounds if none are visible.
        """
        try:
            b = self._visible_bounds()
            if b is None and hasattr(self.plotter, "bounds"):
                b = self.plotter.bounds  # (xmin,xmax,ymin,ymax,zmin,zmax)
            if not b:
                return (0.0, 0.0, 0.0), 1.0
            cx = 0.5 * (float(b[0]) + float(b[1]))
            cy = 0.5 * (float(b[2]) + float(b[3]))
            cz = 0.5 * (float(b[4]) + float(b[5]))
            dx = float(b[1]) - float(b[0])
            dy = float(b[3]) - float(b[2])
            dz = float(b[5]) - float(b[4])
            import math
            diag = max(1e-9, math.sqrt(dx*dx + dy*dy + dz*dz))
            return (cx, cy, cz), diag
        except Exception:
            return (0.0, 0.0, 0.0), 1.0

    def _set_camera_iso(self, direction=(1.0, 1.0, 1.0)) -> None:
        """Place camera along an isometric direction, scaled to scene size."""
        import math
        # center & size
        ctr, diag = self._scene_center_radius()
        cx, cy, cz = ctr

        # normalize direction
        dx, dy, dz = map(float, direction)
        n = math.sqrt(dx*dx + dy*dy + dz*dz) or 1.0
        dx, dy, dz = dx/n, dy/n, dz/n

        # pick an 'up' that is not parallel to the view dir
        up = (0.0, 0.0, 1.0)
        dot = dx*up[0] + dy*up[1] + dz*up[2]
        if abs(dot) > 0.95:   # almost parallel → use Y-up
            up = (0.0, 1.0, 0.0)

        # distance factor: a bit larger than diagonal so everything fits
        dist = 2.0 * diag

        pos = (cx + dx*dist, cy + dy*dist, cz + dz*dist)
        try:
            self.plotter.camera_position = [pos, (cx, cy, cz), up]
            # make sure clipping range adapts; avoid full reset_camera() to keep the pose
            if hasattr(self.plotter, "reset_camera_clipping_range"):
                self.plotter.reset_camera_clipping_range()
        except Exception:
            pass
        try:
            if hasattr(self.plotter, "render"):
                self.plotter.render()
            else:
                self.plotter.update()
        except Exception:
            pass
    def invert_view(self) -> None:
        """Invert the camera position through the focal point."""
        try:
            pos, foc, up = self.plotter.camera_position
            inv = (2 * foc[0] - pos[0], 2 * foc[1] - pos[1], 2 * foc[2] - pos[2])
            self.plotter.camera_position = [inv, foc, up]
            self.plotter.reset_camera_clipping_range()
        except Exception:
            pass

    def refresh(self) -> None:
        """Request a render update (prefer immediate render if available)."""
        try:
            if hasattr(self.plotter, "render"):
                self.plotter.render()
            else:
                self.plotter.update()
        except Exception:
            pass

    # ---- Display controls ----
    def set_mesh_representation(self, ds: int, mode: str) -> None:
        """Set mesh representation for dataset *ds*.

        Args:
            ds: Dataset index in the internal cache.
            mode: One of {'Points','Wireframe','Surface','Surface with Edges'}.
        """
        recs = getattr(self, "_datasets", [])
        if not (isinstance(ds, int) and 0 <= ds < len(recs)):
            return
        rec = recs[ds]
        pdata = rec.get("mesh_surface") or rec.get("mesh") or rec.get("pdata") or rec.get("full_pdata")
        if pdata is None or self.plotter is None:
            return

        rec["representation"] = mode

        # Remove previous actor
        try:
            act = rec.get("actor") or rec.get("actor_mesh") or rec.get("actor_points")
            if act is not None:
                self.plotter.remove_actor(act)
        except Exception:
            pass

        scalars = rec.get("active_scalars", None)
        cmap = rec.get("colormap", "Viridis")
        clim = rec.get("clim", None)
        scalar_bar = bool(rec.get("scalar_bar", False))

        style = None
        show_edges = False
        if mode == "Points":
            style = "points"
        elif mode == "Wireframe":
            style = "wireframe"
        elif mode == "Surface with Edges":
            show_edges = True

        try:
            actor = self.plotter.add_mesh(
                pdata,
                scalars=scalars,
                cmap=cmap,
                clim=clim,
                style=style,
                show_edges=show_edges or bool(rec.get("edge_visibility", False)),
                edge_color=tuple(rec.get("edge_color", (0, 0, 0))),
                opacity=float(rec.get("opacity", 100)) / 100.0,
                point_size=int(rec.get("point_size", 3)),
                line_width=int(rec.get("line_width", 1)),
                lighting=bool(rec.get("lighting", True)),
                name=rec.get("name", f"ds{ds}"),
                scalar_bar_args={"title": rec.get("active_scalar_name", "")} if scalar_bar else None,
                copy_mesh=False,
                reset_camera=False,
            )
            rec["actor"] = actor
        except Exception:
            return

        # Solid color fallback (if no scalars)
        if scalars is None:
            try:
                rgb = rec.get("solid_color", (1.0, 1.0, 1.0))
                if all(isinstance(c, int) for c in rgb):
                    rgb = (rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0)
                actor.prop.color = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
            except Exception:
                pass

        try:
            self.plotter.render()
        except Exception:
            pass

    def set_dataset_color(self, ds: int, r: int, g: int, b: int) -> None:
        """Apply solid RGB color to dataset *ds* and persist.

        Args:
            ds: Dataset index.
            r, g, b: 0..255 components.
        """
        recs = getattr(self, "_datasets", [])
        if not (isinstance(ds, int) and 0 <= ds < len(recs)):
            return
        rec = recs[ds]
        rec["solid_color"] = (int(r), int(g), int(b))
        act = rec.get("actor")
        if act is not None:
            try:
                act.mapper.scalar_visibility = False  # force solid
            except Exception:
                pass
            try:
                act.prop.color = (r/255.0, g/255.0, b/255.0)
            except Exception:
                pass
            try:
                self.plotter.render()
            except Exception:
                pass

    def set_colormap(self, name: str, ds: int) -> None:
        """Set the colormap name for dataset *ds* and refresh scalar mapping."""
        recs = getattr(self, "_datasets", [])
        if not (isinstance(ds, int) and 0 <= ds < len(recs)):
            return
        rec = recs[ds]
        rec["colormap"] = str(name)
        self._refresh_scalars(ds)

    def set_color_mode(self, mode: str, ds: int) -> None:
        """Set 'color mode' for dataset *ds*.

        Supported modes:
        - "Solid Color"
        - "PointData/<ARRAY>"
        - "CellData/<ARRAY>"  (TODO: mapping via vtkOriginalCellIds)
        """
        recs = getattr(self, "_datasets", [])
        if not (isinstance(ds, int) and 0 <= ds < len(recs)):
            return
        rec = recs[ds]
        rec["color_mode"] = mode

        # Solid ⇒ disattiva scalars e aggiorna
        if mode == "Solid Color":
            rec["active_scalars"] = None
            rec["active_scalar_name"] = ""
            self._refresh_scalars(ds)
            return

        # Parse association/name
        assoc = "POINT"
        array_name = None
        if mode.startswith("PointData/"):
            assoc = "POINT"
            array_name = mode.split("/", 1)[1]
        elif mode.startswith("CellData/"):
            assoc = "CELL"
            array_name = mode.split("/", 1)[1]
        else:
            array_name = mode  # raw name, assume point

        # Build scalars mapped to the render surface
        scal = None
        if assoc == "POINT" and array_name:
            vmode = rec.get("vector_mode", "Magnitude")
            scal = self._map_point_scalars_to_surface_rec(rec, array_name, vmode)
        else:
            # TODO: support CellData via vtkOriginalCellIds and per-face averaging if needed
            scal = None

        if scal is None:
            # fallback to solid
            rec["active_scalars"] = None
            rec["active_scalar_name"] = ""
        else:
            rec["active_scalars"] = scal
            rec["active_scalar_name"] = array_name

        self._refresh_scalars(ds)

    def _map_point_scalars_to_surface_rec(self, rec: dict, name: str, vector_mode: str = "Magnitude"):
        """Return per-point scalars mapped to the render surface.

        Priority:
        1) rec['mesh_surface'].point_data['vtkOriginalPointIds'] → direct gather
        2) KDTree( mesh.points ) → nearest-neighbor on surface.points
        3) pass-through if sizes match
        Handles vector arrays (Magnitude / X / Y / Z).
        """
        try:
            mesh = rec.get("mesh_orig") or rec.get("mesh") or rec.get("pdata") or rec.get("full_pdata")
            surf = rec.get("mesh_surface") or rec.get("mesh") or rec.get("pdata") or rec.get("full_pdata")
            if mesh is None or surf is None:
                return None

            # fetch from original mesh (or from surface if already there)
            if hasattr(mesh, "point_data") and name in mesh.point_data:
                base = np.asarray(mesh.point_data[name])
            elif hasattr(surf, "point_data") and name in surf.point_data:
                base = np.asarray(surf.point_data[name])
            else:
                return None

            # vector handling
            if base.ndim == 2 and base.shape[1] in (2, 3):
                vm = (vector_mode or "Magnitude").title()
                if vm == "Magnitude":
                    base = np.linalg.norm(base, axis=1)
                else:
                    comp = {"X": 0, "Y": 1, "Z": 2}.get(vm, 0)
                    comp = min(comp, base.shape[1] - 1)
                    base = base[:, comp]

            # 1) mapping via OriginalPointIds
            ids = None
            if hasattr(surf, "point_data"):
                for key in ("vtkOriginalPointIds", "vtkOriginalPointID", "origids", "OriginalPointIds"):
                    if key in surf.point_data:
                        ids = np.asarray(surf.point_data[key]).astype(np.int64)
                        break
            if ids is not None:
                ids = np.clip(ids, 0, base.shape[0] - 1)
                return base[ids]

            # 2) KDTree fallback
            if KDTree is not None and hasattr(mesh, "points") and hasattr(surf, "points"):
                P_src = np.asarray(mesh.points)
                P_dst = np.asarray(surf.points)
                if P_src.size and P_dst.size:
                    tree = KDTree(P_src)
                    idx = tree.query(P_dst, k=1, workers=-1)[1]
                    idx = np.clip(np.asarray(idx, dtype=np.int64), 0, base.shape[0] - 1)
                    return base[idx]

            # 3) last resort: sizes equal → pass-through
            return base if getattr(surf, "n_points", -1) == base.shape[0] else None
        except Exception:
            return None
        
    def _refresh_scalars(self, ds: int) -> None:
        """Internal: re-apply scalar mapping and LUT/CLIM/scalar bar to current actor."""
        recs = getattr(self, "_datasets", [])
        if not (isinstance(ds, int) and 0 <= ds < len(recs)):
            return
        rec = recs[ds]
        act = rec.get("actor")
        if act is None or self.plotter is None:
            # attempt to rebuild the actor via representation
            rep = rec.get("representation", "Surface")
            self.set_mesh_representation(ds, rep)
            act = rec.get("actor")
            if act is None:
                return

        scalars = rec.get("active_scalars", None)
        cmap = rec.get("colormap", "Viridis")
        clim = rec.get("clim", None)
        sbar = bool(rec.get("scalar_bar", False))

        try:
            if scalars is None:
                act.mapper.scalar_visibility = False
                # Apply solid color when scalar visibility is disabled
                try:
                    rgb = rec.get("solid_color", (1.0, 1.0, 1.0))
                    if all(isinstance(c, int) for c in rgb):
                        rgb = (rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0)
                    act.prop.color = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
                except Exception:
                    pass
            else:
                pdata = rec.get("mesh_surface") or rec.get("mesh") or rec.get("pdata") or rec.get("full_pdata")
                try:
                    self.plotter.remove_actor(act)
                except Exception:
                    pass
                act = self.plotter.add_mesh(
                    pdata,
                    scalars=scalars,
                    cmap=cmap,
                    clim=clim,
                    style=None if rec.get("representation", "Surface") != "Wireframe" else "wireframe",
                    show_edges=(rec.get("representation", "Surface") == "Surface with Edges") or bool(rec.get("edge_visibility", False)),
                    edge_color=tuple(rec.get("edge_color", (0, 0, 0))),
                    opacity=float(rec.get("opacity", 100)) / 100.0,
                    point_size=int(rec.get("point_size", 3)),
                    line_width=int(rec.get("line_width", 1)),
                    lighting=bool(rec.get("lighting", True)),
                    name=rec.get("name", f"ds{ds}"),
                    scalar_bar_args={"title": rec.get("active_scalar_name", "")} if sbar else None,
                    copy_mesh=False,
                    reset_camera=False,
                )
                rec["actor"] = act
                if not sbar:
                    try:
                        self.plotter.remove_scalar_bar()
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            self.plotter.render()
        except Exception:
            pass

    def set_point_size(self, size: int, dataset_index: int | None = None) -> None:
        """Set point size globally or for a specific dataset."""
        try:
            sz = int(size)
        except Exception:
            return
        if dataset_index is None:
            self._point_size = sz
            for idx, rec in enumerate(self._datasets):
                if rec.get("kind") != "points":
                    continue
                rec["point_size"] = sz
                actor = rec.get("actor_points")
                if actor is not None:
                    try:
                        prop = actor.GetProperty()
                        if prop:
                            prop.SetPointSize(sz)
                    except Exception:
                        pass
            try:
                self.plotter.update()
            except Exception:
                pass
            return

        try:
            rec = self._datasets[dataset_index]
        except Exception:
            return
        if rec.get("kind") != "points":
            return
        rec["point_size"] = sz
        actor = rec.get("actor_points")
        if actor is not None:
            try:
                prop = actor.GetProperty()
                if prop:
                    prop.SetPointSize(sz)
                self.plotter.update()
            except Exception:
                pass

    # def set_color_mode(self, mode: str, dataset_index: int | None = None) -> None:
    #     """Set color mode globally or for a specific point dataset."""
    #     if dataset_index is None:
    #         self._color_mode = mode
    #         for idx, rec in enumerate(self._datasets):
    #             if rec.get("kind") == "points":
    #                 self.set_color_mode(mode, idx)
    #         return
    #     try:
    #         rec = self._datasets[dataset_index]
    #     except Exception:
    #         return
    #     if rec.get("kind") != "points":
    #         return
    #     rec["color_mode"] = mode
    #     actor = rec.get("actor_points")
    #     if actor is not None:
    #         try:
    #             self.plotter.remove_actor(actor)
    #         except Exception:
    #             pass
    #         rec["actor_points"] = None
    #     if rec.get("visible", True):
    #         rec["actor_points"] = self._add_points_by_mode(
    #             rec.get("pdata"),
    #             rec.get("has_rgb", False),
    #             rec.get("solid_color", self._solid_fallback),
    #             mode,
    #             rec.get("cmap", self._cmap),
    #             rec.get("points_as_spheres", self._points_as_spheres),
    #         )
    #         actor = rec.get("actor_points")
    #         if actor is not None:
    #             try:
    #                 prop = actor.GetProperty()
    #                 if prop:
    #                     prop.SetPointSize(rec.get("point_size", self._point_size))
    #             except Exception:
    #                 pass
    #     # Auto-show a managed scalar bar when switching to colormap mode
    #     try:
    #         if mode == "Normal Colormap":
    #             title = str(rec.get("scalar_name", "Intensity"))
    #             current = str(getattr(self, "_scalarbar_mode", "horizontal-br"))
    #             if current == "hidden":
    #                 current = "vertical-tr"  # default sensato
    #             self.set_colorbar_mode(current, title)
    #     except Exception:
    #         pass
    #     try:
    #         self._apply_scalarbar()
    #     except Exception:
    #         pass
    #     try:
    #         self.plotter.update()
    #     except Exception:
    #         pass

    def set_solid_color(self, r: int, g: int, b: int) -> None:
        """Set a solid RGB color on all actors (best-effort)."""
        self._solid_color = (r, g, b)
        self._refresh_datasets()

    # def set_colormap(self, name: str, dataset_index: int | None = None) -> None:
    #     """Set colormap globally or for a specific dataset when in colormap mode."""
    #     if dataset_index is None:
    #         self._cmap = name
    #         for idx, rec in enumerate(self._datasets):
    #             if rec.get("kind") == "points":
    #                 self.set_colormap(name, idx)
    #         return
    #     try:
    #         rec = self._datasets[dataset_index]
    #     except Exception:
    #         return
    #     if rec.get("kind") != "points":
    #         return
    #     rec["cmap"] = name
    #     if rec.get("color_mode", self._color_mode) != "Normal Colormap" and rec.get("has_rgb", False):
    #         return
    #     actor = rec.get("actor_points")
    #     if actor is not None:
    #         try:
    #             self.plotter.remove_actor(actor)
    #         except Exception:
    #             pass
    #     rec["actor_points"] = None
    #     if rec.get("visible", True):
    #         rec["actor_points"] = self._add_points_by_mode(
    #             rec.get("pdata"),
    #             rec.get("has_rgb", False),
    #             rec.get("solid_color", self._solid_fallback),
    #             rec.get("color_mode", self._color_mode),
    #             name,
    #             rec.get("points_as_spheres", self._points_as_spheres),
    #         )
    #         actor = rec.get("actor_points")
    #         if actor is not None:
    #             try:
    #                 prop = actor.GetProperty()
    #                 if prop:
    #                     prop.SetPointSize(rec.get("point_size", self._point_size))
    #             except Exception:
    #                 pass
    #     # Ensure the scalar bar is shown with the correct title in colormap mode
    #     try:
    #         if rec.get("color_mode", self._color_mode) == "Normal Colormap":
    #             title = str(rec.get("scalar_name", "Intensity"))
    #             current = str(getattr(self, "_scalarbar_mode", "horizontal-br"))
    #             if current == "hidden":
    #                 current = "vertical-tr"
    #             self.set_colorbar_mode(current, title)
    #     except Exception:
    #         pass
    #     try:
    #         self._apply_scalarbar()
    #     except Exception:
    #         pass
    #     try:
    #         self.plotter.update()
    #     except Exception:
    #         pass

    # def set_dataset_color(self, dataset_index: int, r: int, g: int, b: int) -> None:
    #     """Set per-dataset solid color and update its actor if needed.

    #     Works for both point clouds and meshes. For points, if the dataset's
    #     current color mode is Solid, the actor color is updated directly;
    #     otherwise the actor is rebuilt only when needed.
    #     """
    #     try:
    #         rec = self._datasets[dataset_index]
    #     except Exception:
    #         return

    #     # Clamp and normalize to 0..1
    #     rgb = (
    #         max(0, min(255, int(r))) / 255.0,
    #         max(0, min(255, int(g))) / 255.0,
    #         max(0, min(255, int(b))) / 255.0,
    #     )
    #     rec["solid_color"] = rgb

    #     # Mesh datasets: update actor property if present
    #     if rec.get("kind") == "mesh":
    #         actor_m = rec.get("actor_mesh")
    #         if actor_m is not None:
    #             try:
    #                 actor_m.GetProperty().SetColor(rgb)
    #                 self.plotter.update()
    #             except Exception:
    #                 pass
    #         return

    #     # Point datasets
    #     if rec.get("kind") != "points":
    #         return

    #     mode = str(rec.get("color_mode", self._color_mode))
    #     actor = rec.get("actor_points")

    #     # If currently in Solid mode and an actor exists, update in-place
    #     if actor is not None and mode == "Solid":
    #         try:
    #             actor.GetProperty().SetColor(rgb)
    #             self.plotter.update()
    #             return
    #         except Exception:
    #             pass

    #     # Otherwise, rebuild only if visible (to reflect future switch to Solid)
    #     if actor is not None:
    #         try:
    #             self.plotter.remove_actor(actor)
    #         except Exception:
    #             pass
    #         rec["actor_points"] = None

    #     if rec.get("visible", True):
    #         new_actor = self._add_points_by_mode(
    #             rec.get("pdata"),
    #             rec.get("has_rgb", False),
    #             rec.get("solid_color", self._solid_fallback),
    #             rec.get("color_mode", self._color_mode),
    #             rec.get("cmap", self._cmap),
    #             rec.get("points_as_spheres", self._points_as_spheres),
    #         )
    #         if new_actor is not None:
    #             try:
    #                 prop = new_actor.GetProperty()
    #                 if prop:
    #                     prop.SetPointSize(rec.get("point_size", self._point_size))
    #             except Exception:
    #                 pass
    #         rec["actor_points"] = new_actor
    #         try:
    #             self.plotter.update()
    #         except Exception:
    #             pass

    def set_points_as_spheres(self, enabled: bool) -> None:
        """Toggle rendering style for points and refresh existing actors."""
        self._points_as_spheres = bool(enabled)
        self._refresh_datasets()


    def add_points(self, points, colors=None, normals=None) -> int:
        """Add a point cloud to the scene with smart coloring and optional normals.

        Args:
            points: array-like (N,3)
            colors: optional (N,3) in [0,1] or [0,255]
            normals: optional (N,3) float array of unit vectors

        Returns:
            Integer dataset index that can be used to toggle/augment visuals later.
        """
        try:
            import numpy as np
            import pyvista as pv  # type: ignore

            pts = np.asarray(points, dtype=float)
            pdata = pv.PolyData(pts)

            has_rgb = False
            if colors is not None:
                cols = np.asarray(colors)
                if cols.max() > 1.5:
                    cols = cols / 255.0
                pdata["RGB"] = (cols * 255).astype(np.uint8)
                has_rgb = True

            if normals is not None:
                nrm = np.asarray(normals, dtype=float)
                if nrm.ndim == 2 and nrm.shape[1] >= 3 and nrm.shape[0] == pts.shape[0]:
                    pdata["Normals"] = nrm[:, :3].astype(np.float32)

            # fake intensity per colormap fallback
            if pts.size:
                z = pts[:, 2]
                zmin, zmax = float(z.min()), float(z.max())
                inten = (z - zmin) / (zmax - zmin) if zmax > zmin else np.zeros_like(z)
                pdata["Intensity"] = inten.astype(np.float32)

            # Auto-initialize or rebalance the view budget
            if not self._auto_budget_initialized:
                cap = self._target_visible_points()
                n = int(pts.shape[0])
                p = min(100, max(1, int(cap * 100 / max(1, n))))
                self._view_budget_percent = p
                self._auto_budget_initialized = True
            else:
                self._rebalance_budget_across_datasets()
                
            full_pdata = pdata
            pdata_for_view = self._apply_budget_to_polydata(full_pdata, has_rgb, self._view_budget_percent)
            # Scegli un colore distinto dalla palette per questo dataset
            palette = getattr(self, "_default_palette", [self._solid_fallback])
            ds_color = palette[len(self._datasets) % len(palette)] if palette else self._solid_fallback
            actor_points = self._add_points_by_mode(
                pdata_for_view,
                has_rgb,
                ds_color,
                self._color_mode,
                self._cmap,
                self._points_as_spheres,
            )
            # Ensure the new actor uses the current point size
            try:
                if actor_points is not None:
                    prop = actor_points.GetProperty()
                    if prop:
                        prop.SetPointSize(self._point_size)
            except Exception:
                pass

            rec = {
                "kind": "points",
                "full_pdata": full_pdata,
                "pdata": pdata_for_view,
                "actor_points": actor_points,
                "actor_normals": None,
                "has_rgb": has_rgb,
                "solid_color": ds_color,
                "visible": True,
                "point_size": self._point_size,
                "view_percent": self._view_budget_percent,
                "color_mode": self._color_mode,
                "cmap": self._cmap,
                "points_as_spheres": self._points_as_spheres,
                # ---- per-dataset normals state ----
                "normals_visible": False,
                "normals_style": self._normals_style,
                "normals_color": self._normals_color,
                "normals_percent": self._normals_percent,
                "normals_scale": self._normals_scale,
                "scalar_name": "Intensity",
            }
            self._datasets.append(rec)
            ds_index = len(self._datasets) - 1

            self.plotter.reset_camera()
            try:
                self._apply_overlays()
            except Exception:
                pass
            return ds_index
        except Exception:
            return -1
        

    def _add_points_by_mode(self, pdata, has_rgb: bool, solid_color=None, mode=None, cmap=None, spheres=None):
        """Internal helper to add points according to current mode and style.

        Args:
            pdata: PolyData con eventuali array 'RGB'/'Intensity'.
            has_rgb: True se 'RGB' è presente.
            solid_color: opzionale (r,g,b) in [0,1] da usare in Solid.
            mode: rendering mode (Solid, Normal RGB, Normal Colormap).
            cmap: colormap name when in colormap mode.
            spheres: render points as spheres if True.
        """
        mode = self._color_mode if mode is None else mode
        cmap = self._cmap if cmap is None else cmap
        spheres = self._points_as_spheres if spheres is None else spheres

        # Colore Solid risolto (per-dataset o fallback globale)
        solid = solid_color if solid_color is not None else self._solid_fallback

        # Modalità Solid
        if mode == "Solid":
            try:
                return self.plotter.add_points(
                    pdata,
                    color=solid,
                    render_points_as_spheres=spheres,
                    show_scalar_bar=False,
                )
            except Exception:
                pass

        # Colormap (o fallback se manca RGB)
        if mode == "Normal Colormap" or not has_rgb:
            try:
                return self.plotter.add_points(
                    pdata,
                    scalars="Intensity",
                    cmap=cmap,
                    render_points_as_spheres=spheres,
                    show_scalar_bar=False,
                )
            except Exception:
                pass

        # Default: RGB reale
        try:
            return self.plotter.add_points(
                pdata,
                scalars="RGB",
                rgb=True,
                render_points_as_spheres=spheres,
                show_scalar_bar=False,
            )
        except Exception:
            # Fallback finale: Solid
            try:
                return self.plotter.add_points(
                    pdata,
                    color=solid,
                    render_points_as_spheres=spheres,
                    show_scalar_bar=False,
                )
            except Exception:
                return None

    def _refresh_datasets(self) -> None:
        """Rebuild all actors according to current rendering settings.

        Preserves:
        - camera position/clip range,
        - point-size on point actors,
        - per-dataset normals state (visibility, style, color, percent, scale).
        """
        try:
            # Save camera
            try:
                cam_pos = self.plotter.camera_position
            except Exception:
                cam_pos = None

            # Clear scene and rebuild dataset list
            self.plotter.clear()
            # After a clear() we must re-apply axes/grid overlays
            try:
                self._apply_overlays()
            except Exception:
                pass
            try:
                self._apply_background()
                self._apply_scalarbar()
            except Exception:
                pass
            new_list: list[dict] = []

            for old_rec in self._datasets:
                kind = old_rec.get("kind", "points")

                # ---------- Mesh datasets ----------
                if kind == "mesh":
                    actor_mesh = None
                    if old_rec.get("visible", True):
                        try:
                            actor_mesh = self._add_mesh_no_bar(
                                old_rec.get("mesh"),
                                color=old_rec.get("solid_color", self._solid_fallback),
                                style="wireframe"
                                if old_rec.get("representation", "surface") == "wireframe"
                                else "surface",
                                opacity=float(old_rec.get("opacity", 100)) / 100.0,
                            )
                        except Exception:
                            actor_mesh = None

                    new_rec = dict(old_rec)
                    new_rec["actor_mesh"] = actor_mesh
                    new_list.append(new_rec)
                    continue

                # ---------- Point datasets ----------
                full_pdata = old_rec.get("full_pdata", old_rec.get("pdata"))
                has_rgb = bool(old_rec.get("has_rgb", False))
                was_visible = bool(old_rec.get("visible", True))
                view_pct = int(old_rec.get("view_percent", 100))

                # Apply view budget
                pdata_for_view = self._apply_budget_to_polydata(
                    full_pdata, has_rgb, view_pct
                )

                # Re-add points actor according to color mode/cmap
                actor_points = None
                if was_visible:
                    actor_points = self._add_points_by_mode(
                        pdata_for_view,
                        has_rgb,
                        old_rec.get("solid_color", self._solid_fallback),
                        old_rec.get("color_mode", self._color_mode),
                        old_rec.get("cmap", self._cmap),
                        old_rec.get("points_as_spheres", self._points_as_spheres),
                    )
                    # Re-apply point size
                    try:
                        if actor_points is not None:
                            prop = actor_points.GetProperty()
                            if prop:
                                prop.SetPointSize(old_rec.get("point_size", self._point_size))
                    except Exception:
                        pass

                # Base new record
                new_rec = {
                    "kind": "points",
                    "full_pdata": full_pdata,
                    "pdata": pdata_for_view,
                    "actor_points": actor_points,
                    "actor_normals": None,  # will be rebuilt below if needed
                    "has_rgb": has_rgb,
                    "solid_color": old_rec.get("solid_color", self._solid_fallback),
                    "visible": was_visible,
                    "point_size": old_rec.get("point_size", self._point_size),
                    "view_percent": view_pct,
                    "color_mode": old_rec.get("color_mode", self._color_mode),
                    "cmap": old_rec.get("cmap", self._cmap),
                    "points_as_spheres": old_rec.get(
                        "points_as_spheres", self._points_as_spheres
                    ),
                    # ---- Normals per-dataset state (carry over with sane defaults) ----
                    "normals_visible": bool(old_rec.get("normals_visible", False)),
                    "normals_style": str(old_rec.get("normals_style", getattr(self, "_normals_style", "Axis RGB"))),
                    "normals_color": tuple(old_rec.get("normals_color", getattr(self, "_normals_color", (0.9, 0.9, 0.2)))),
                    "normals_percent": int(old_rec.get("normals_percent", getattr(self, "_normals_percent", 50))),
                    "normals_scale": int(old_rec.get("normals_scale", getattr(self, "_normals_scale", 20))),
                }
                new_list.append(new_rec)

                # ---- Rebuild normals actor if it was visible ----
                try:
                    if new_rec["normals_visible"] and hasattr(pdata_for_view, "point_data") and "Normals" in pdata_for_view.point_data:
                        self._rebuild_normals_actor(
                            ds=len(new_list) - 1, # here it was data_index
                            style=new_rec["normals_style"],
                            color=new_rec["normals_color"],
                            percent=new_rec["normals_percent"],
                            scale=new_rec["normals_scale"],
                        )
                except Exception:
                    # Best-effort: keep going even if normals rebuild fails
                    pass

            # Swap registry
            self._datasets = new_list

            # Ensure point-size is applied to all point actors (paranoia pass)
            try:
                for rec in self._datasets:
                    if rec.get("kind") != "points":
                        continue
                    actor = rec.get("actor_points")
                    if actor is not None:
                        prop = actor.GetProperty()
                        if prop:
                            prop.SetPointSize(rec.get("point_size", self._point_size))
            except Exception:
                pass

            # Restore camera
            if cam_pos is not None:
                try:
                    self.plotter.camera_position = cam_pos
                    self.plotter.reset_camera_clipping_range()
                except Exception:
                    pass

            # Final refresh
            try:
                self.plotter.update()
            except Exception:
                pass

        except Exception:
            # Never raise from a refresh; keep UI responsive.
            pass

    def add_pyvista_mesh(self, mesh) -> int:
        """Add a VTK/PyVista mesh to the scene with a guaranteed surface actor.

        Stores both:
        - rec['mesh_orig']    : the original dataset (e.g. UnstructuredGrid)
        - rec['mesh_surface'] : a PolyData surface extracted for rendering

        All visual representations use the surface, so scalars and edges behave.
        """
        try:
            render_mesh = _ensure_surface_mesh(mesh)
            actor = self._add_mesh_no_bar(render_mesh)
            rec = {
                "kind": "mesh",
                "mesh_orig": mesh,
                "mesh_surface": render_mesh,
                "actor": actor,
                "visible": True,
                "representation": "Surface",
                "opacity": 100,
                "solid_color": (1.0, 1.0, 1.0),
                "active_scalars": None,
                "active_scalar_name": "",
                "colormap": "Viridis",
                "clim": None,
                "scalar_bar": False,
                "edge_visibility": False,
                "edge_color": (0, 0, 0),
                "point_size": 3,
                "line_width": 1,
                "lighting": True,
            }
            self._datasets.append(rec)
            try:
                self.plotter.reset_camera()
            except Exception:
                pass
            return len(self._datasets) - 1
        except Exception:
            return -1

    def clear(self) -> None:
        """Remove all actors from the scene."""
        try:
            self.plotter.clear()
        except Exception:
            pass
    
    def enable_safe_rendering(self, enabled: bool = True) -> None:
        """Toggle conservative rendering settings (useful on macOS/Metal).

        Args:
            enabled: If True, apply safe settings; if False, attempt to restore defaults.
        """
        try:
            if not hasattr(self.plotter, "render_window"):
                return
            if enabled:
                self.plotter.render_window.SetMultiSamples(0)
                try:
                    self.plotter.renderer.SetUseDepthPeeling(0)
                    if hasattr(self.plotter.renderer, "SetUseFXAA"):
                        self.plotter.renderer.SetUseFXAA(False)
                except Exception:
                    pass
            else:
                # Best-effort revert
                try:
                    self.plotter.render_window.SetMultiSamples(4)
                except Exception:
                    pass
        except Exception:
            pass
    
    def _apply_background(self) -> None:
        """Apply the current background RGB to the plotter (best-effort)."""
        try:
            r, g, b = getattr(self, "_bg_rgb", (255, 255, 255))
            if hasattr(self.plotter, "set_background"):
                self.plotter.set_background((int(r), int(g), int(b)))
        except Exception:
            pass
    
    def _infer_scalarbar_title(self) -> str:
        """Return a sensible scalar bar title based on visible datasets.

        Prefers the first visible point dataset in 'Normal Colormap' mode and
        uses its recorded scalar name; defaults to 'Intensity'.
        """
        try:
            for rec in self._datasets:
                if rec.get("kind") == "points" and rec.get("visible", True):
                    if rec.get("color_mode", self._color_mode) == "Normal Colormap":
                        return str(rec.get("scalar_name", "Intensity"))
        except Exception:
            pass
        return "Intensity"
    
    def _apply_scalarbar(self) -> None:
        """Apply current scalar bar mode/title to the plotter (removes existing first)."""
        try:
            mode = getattr(self, "_scalarbar_mode", "horizontal-br")
            _title = getattr(self, "_scalarbar_title", "")
            if not _title:
                _title = self._infer_scalarbar_title()
            title = _title
            if hasattr(self.plotter, "remove_scalar_bar"):
                try:
                    self.plotter.remove_scalar_bar()
                except Exception:
                    pass
            if mode == "hidden":
                return
            if mode == "vertical-tr":
                opts = dict(
                    vertical=True,
                    position_x=0.85,
                    position_y=0.1,
                    height=0.8,
                    width=0.1,
                    title=title,
                    fmt="%0.3f",
                )
            else:  # horizontal-br default
                opts = dict(
                    vertical=False,
                    position_x=0.62,
                    position_y=0.02,
                    height=0.12,
                    width=0.36,
                    title=title,
                    fmt="%0.3f",
                )
            try:
                self.plotter.add_scalar_bar(**opts)
            except Exception:
                pass
        except Exception:
            pass

    def set_background_color(self, color) -> None:
        """Set the plotter background color.

        Args:
            color: `QtGui.QColor` or an (r,g,b) tuple in 0..255 or 0..1 floats.
        """
        try:
            r = g = b = None
            try:
                from PySide6 import QtGui as _QtGui  # local import
            except Exception:
                _QtGui = None
            if _QtGui is not None and isinstance(color, _QtGui.QColor):
                r, g, b = color.red(), color.green(), color.blue()
            else:
                r, g, b = color
                if all(isinstance(c, float) and 0.0 <= c <= 1.0 for c in (r, g, b)):
                    r, g, b = int(round(r * 255)), int(round(g * 255)), int(round(b * 255))
            self._bg_rgb = (int(r), int(g), int(b))
            self._apply_background()
            self.refresh()
        except Exception:
            pass

    def set_colorbar_mode(self, mode: str = "horizontal-br", title: str = "") -> None:
        """Set scalar bar layout and apply it immediately.

        Supported modes:
            - 'hidden'         : remove scalar bar
            - 'horizontal-br'  : bottom-right horizontal bar
            - 'vertical-tr'    : top-right vertical bar
        """
        try:
            mode = str(mode).strip().lower()
            if mode not in ("hidden", "horizontal-br", "vertical-tr"):
                mode = "horizontal-br"
            self._scalarbar_mode = mode
            self._scalarbar_title = str(title)
            self._apply_scalarbar()
            self.refresh()
        except Exception:
            pass
    
    def _add_mesh_no_bar(self, *args, **kwargs):
        """Wrapper around plotter.add_mesh that guarantees no auto scalar bar is shown.

        Forces show_scalar_bar=False so that colorbar placement is managed exclusively
        by `_apply_scalarbar()` / `set_colorbar_mode()`.
        """
        try:
            kwargs = dict(kwargs)
            kwargs["show_scalar_bar"] = False
            return self.plotter.add_mesh(*args, **kwargs)
        except Exception:
            try:
                return self.plotter.add_mesh(*args)
            except Exception:
                return None
            
    # ---- Overlays (axes + bounds grid) ---------------------------------
    def _apply_overlays(self) -> None:
        """(Re)apply overlays (axes, bounds grid) based on current flags."""
        try:
            # Remove previous bounds axes if the backend keeps them around
            if hasattr(self.plotter, "remove_bounds_axes"):
                try:
                    self.plotter.remove_bounds_axes()
                except Exception:
                    pass

            if self._axes_on:
                try:
                    self.plotter.add_axes()
                except Exception:
                    pass
            if self._bounds_on:
                try:
                    self.plotter.show_bounds(**self._bounds_kwargs)
                except Exception:
                    pass
        except Exception:
            pass

    def set_colorbar_vertical(self, enabled: bool = True, title: str = "") -> None:
        """Show or hide a vertical colorbar at top-right of the scene.

        Args:
            enabled: If True, display the colorbar; if False, remove it.
            title: Optional label for the scalar bar.
        """
        try:
            if not enabled:
                if hasattr(self.plotter, "remove_scalar_bar"):
                    try:
                        self.plotter.remove_scalar_bar()
                    except Exception:
                        pass
                return

            opts = dict(
                vertical=True,
                position_x=0.85,  # right margin
                position_y=0.1,   # start a bit below the top
                height=0.8,
                width=0.1,
                title=title,
                fmt="%0.3f",
            )
            try:
                self.plotter.add_scalar_bar(**opts)
            except Exception:
                pass
        except Exception:
            pass

    def set_axes_enabled(self, enabled: bool) -> None:
        """Public toggle for axes overlay."""
        self._axes_on = bool(enabled)
        try:
            self._apply_overlays()
            self.refresh()
        except Exception:
            pass

    def set_grid_enabled(self, enabled: bool, **kwargs) -> None:
        """Public toggle for bounds grid overlay.

        You may pass additional kwargs forwarded to plotter.show_bounds(),
        e.g., grid='front', location='outer', color='black', etc.
        """
        self._bounds_on = bool(enabled)
        if kwargs:
            self._bounds_kwargs.update(kwargs)
        try:
            self._apply_overlays()
            self.refresh()
        except Exception:
            pass

    def apply_normals_properties(
        self,
        ds: int,
        *,
        style: str | None = None,
        color: tuple[float, float, float] | None = None,
        percent: int | None = None,
        scale: int | None = None,
        ensure_visible: bool = True,
    ) -> None:
        """Update per-dataset normals properties and optionally rebuild glyphs.

        Args:
            ds: Dataset index.
            style: 'Uniform' | 'Axis RGB' | 'RGB Components'.
            color: Uniform RGB (0..1 floats).
            percent: 1..100 fraction of glyphs shown.
            scale: 1..200 glyph magnitude slider value.
            ensure_visible: If True, shows/rebuilds normals actor immediately.
        """
        try:
            rec = self._datasets[ds]
        except Exception:
            return

        if style is not None:
            rec["normals_style"] = str(style)
        if color is not None:
            try:
                r, g, b = color
                rec["normals_color"] = (float(r), float(g), float(b))
            except Exception:
                pass
        if percent is not None:
            rec["normals_percent"] = int(max(1, min(100, int(percent))))
        if scale is not None:
            rec["normals_scale"] = int(max(1, min(200, int(scale))))

        if ensure_visible:
            # This will rebuild the actor with the latest properties
            self.set_normals_visibility(ds, True)
        elif rec.get("normals_visible"):
            # Rebuild without toggling visibility
            self._rebuild_normals_actor(
                ds,
                style=str(rec.get("normals_style", self._normals_style)),
                color=tuple(rec.get("normals_color", self._normals_color)),
                percent=int(rec.get("normals_percent", self._normals_percent)),
                scale=int(rec.get("normals_scale", self._normals_scale)),
            )
            
    def set_normals_visibility(self, dataset_index: int, visible: bool, scale: float | None = None) -> None:
        """Show/hide normals glyphs for a given dataset.

        This keeps per-dataset state and rebuilds the glyph actor as needed.

        Args:
            dataset_index: Target dataset index.
            visible: True to show normals, False to hide them.
            scale: Optional legacy float override (kept for backward compat).
                   If provided, it is mapped to the new integer slider domain (1..200).
        """
        try:
            rec = self._datasets[dataset_index]
        except Exception:
            return

        # Hide path
        if not visible:
            try:
                if rec.get("actor_normals") is not None:
                    self.plotter.remove_actor(rec["actor_normals"])
            except Exception:
                pass
            rec["actor_normals"] = None
            rec["normals_visible"] = False
            try:
                self.plotter.update()
            except Exception:
                pass
            return

        # Show path: rebuild regardless of whether the budgeted pdata already carries "Normals"
        # (the builder will source from rec["normals_array"] or full_pdata as needed)
        pdata = rec.get("pdata")

        # Optional legacy scale override
        if scale is not None:
            try:
                s = int(max(1, min(200, round(float(scale) * 1000.0))))
                rec["normals_scale"] = s
            except Exception:
                pass

        self._rebuild_normals_actor(
            dataset_index,
            style=str(rec.get("normals_style", self._normals_style)),
            color=tuple(rec.get("normals_color", self._normals_color)),
            percent=int(rec.get("normals_percent", self._normals_percent)),
            scale=int(rec.get("normals_scale", self._normals_scale)),
        )
        rec["normals_visible"] = True
        try:
            self.plotter.update()
        except Exception:
            pass
    
    # ---- Public normals setters -------------------------------------------
    def set_normals_style(self, dataset_index: int, style: str) -> None:
        """Set normals color style for a dataset and rebuild if visible."""
        try:
            rec = self._datasets[dataset_index]
        except Exception:
            return
        style = str(style).strip()
        if style not in ("Uniform", "Axis RGB", "RGB Components"):
            style = "Axis RGB"
        rec["normals_style"] = style
        if rec.get("normals_visible", False):
            self._rebuild_normals_actor(
                dataset_index,
                style=style,
                color=tuple(rec.get("normals_color", self._normals_color)),
                percent=int(rec.get("normals_percent", self._normals_percent)),
                scale=int(rec.get("normals_scale", self._normals_scale)),
            )
            try:
                self.plotter.update()
            except Exception:
                pass

    def set_normals_color(self, dataset_index: int, r: int, g: int, b: int) -> None:
        """Set uniform color for normals (used only in 'Uniform' style)."""
        try:
            rec = self._datasets[dataset_index]
        except Exception:
            return
        color = (
            max(0, min(255, int(r))) / 255.0,
            max(0, min(255, int(g))) / 255.0,
            max(0, min(255, int(b))) / 255.0,
        )
        rec["normals_color"] = color
        if rec.get("normals_visible", False) and rec.get("normals_style") == "Uniform":
            self._rebuild_normals_actor(
                dataset_index,
                style="Uniform",
                color=color,
                percent=int(rec.get("normals_percent", self._normals_percent)),
                scale=int(rec.get("normals_scale", self._normals_scale)),
            )
            try:
                self.plotter.update()
            except Exception:
                pass

    def set_normals_fraction(self, dataset_index: int, percent: int) -> None:
        """Set the percentage (1..100) of normals to draw and rebuild if visible."""
        try:
            rec = self._datasets[dataset_index]
        except Exception:
            return
        p = max(1, min(100, int(percent)))
        rec["normals_percent"] = p
        if rec.get("normals_visible", False):
            self._rebuild_normals_actor(
                dataset_index,
                style=str(rec.get("normals_style", self._normals_style)),
                color=tuple(rec.get("normals_color", self._normals_color)),
                percent=p,
                scale=int(rec.get("normals_scale", self._normals_scale)),
            )
            try:
                self.plotter.update()
            except Exception:
                pass

    def set_normals_scale(self, dataset_index: int, scale: int) -> None:
        """Set the glyph scale slider value (1..200) and rebuild if visible."""
        try:
            rec = self._datasets[dataset_index]
        except Exception:
            return
        s = max(1, min(200, int(scale)))
        rec["normals_scale"] = s
        if rec.get("normals_visible", False):
            self._rebuild_normals_actor(
                dataset_index,
                style=str(rec.get("normals_style", self._normals_style)),
                color=tuple(rec.get("normals_color", self._normals_color)),
                percent=int(rec.get("normals_percent", self._normals_percent)),
                scale=s,
            )
            try:
                self.plotter.update()
            except Exception:
                pass

    def set_points_visibility(self, dataset_index: int, visible: bool) -> None:
        """Show/hide the points actor for a dataset by removing/adding the actor."""
        try:
            rec = self._datasets[dataset_index]
        except Exception:
            return

        rec["visible"] = bool(visible)
        try:
            if not visible:
                actor = rec.get("actor_points")
                if actor is not None:
                    try:
                        self.plotter.remove_actor(actor)
                    except Exception:
                        pass
                    rec["actor_points"] = None
                self.plotter.update()
                return

            # Visible: (re)create from current budgeted pdata
            pdata_for_view = self._apply_budget_to_polydata(
                rec.get("full_pdata", rec.get("pdata")),
                rec.get("has_rgb", False),
                rec.get("view_percent", 100),
            )
            rec["pdata"] = pdata_for_view
            actor = self._add_points_by_mode(
                pdata_for_view,
                bool(rec.get("has_rgb", False)),
                rec.get("solid_color", self._solid_fallback),
                rec.get("color_mode", self._color_mode),
                rec.get("cmap", self._cmap),
                rec.get("points_as_spheres", self._points_as_spheres),
            )
            # Applica la dimensione punti corrente
            try:
                if actor is not None:
                    prop = actor.GetProperty()
                    if prop:
                        prop.SetPointSize(self._point_size)
            except Exception:
                pass
            rec["actor_points"] = actor
            self.plotter.update()
        except Exception:
            pass

    def color_points_by_array(self, dataset_index: int, array_name: str, cmap: str = "viridis", show_scalar_bar: bool = True) -> None:
        """Color a point dataset by a point-data array and rebuild its actor.

        Args:
            dataset_index: Index of the target dataset (must be kind == 'points').
            array_name: Name of the array present in rec['pdata'].point_data.
            cmap: Matplotlib colormap name (e.g., 'viridis', 'plasma').
            show_scalar_bar: If True, show scalar bar titled with the array name.
        """
        recs = getattr(self, "_datasets", [])
        if not (isinstance(dataset_index, int) and 0 <= dataset_index < len(recs)):
            return
        rec = recs[dataset_index]
        if rec.get("kind") != "points":
            return

        pdata = rec.get("pdata")
        if pdata is None:
            return
        try:
            npts = int(getattr(pdata, "n_points", 0))
        except Exception:
            npts = 0
        if npts <= 0:
            return

        # Ensure the array exists on pdata
        try:
            if not (hasattr(pdata, "point_data") and array_name in pdata.point_data):
                return
        except Exception:
            return

        # Remove previous points actor
        try:
            actor = rec.get("actor_points")
            if actor is not None and hasattr(self, "plotter"):
                self.plotter.remove_actor(actor)
        except Exception:
            pass
        rec["actor_points"] = None

        # Add a new points actor using the provided scalars
        try:
            actor = self.plotter.add_points(
                pdata,
                scalars=array_name,
                cmap=cmap,
                render_points_as_spheres=bool(rec.get("points_as_spheres", False)),
                show_scalar_bar=False,
            )
            rec["actor_points"] = actor
            # Keep point size
            try:
                size = int(rec.get("point_size", getattr(self, "_point_size", 3)))
                prop = actor.GetProperty() if actor is not None else None
                if prop:
                    prop.SetPointSize(size)
            except Exception:
                pass
        except Exception:
            return

        # Book-keeping and optional scalar bar
        rec["color_mode"] = f"PointData/{array_name}"
        rec["scalar_name"] = array_name
        rec["cmap"] = cmap
        try:
            if show_scalar_bar and hasattr(self, "set_colorbar_mode"):
                self.set_colorbar_mode(getattr(self, "_scalarbar_mode", "horizontal-br"), title=array_name)
            elif hasattr(self, "set_colorbar_mode"):
                self.set_colorbar_mode("hidden")
        except Exception:
            pass

        try:
            if hasattr(self, "plotter"):
                self.plotter.render()
        except Exception:
            pass


    def reset_point_coloring(self, dataset_index: int) -> None:
        """Restore default coloring for a point dataset (Solid/RGB/Intensity heuristic)."""
        recs = getattr(self, "_datasets", [])
        if not (isinstance(dataset_index, int) and 0 <= dataset_index < len(recs)):
            return
        rec = recs[dataset_index]
        if rec.get("kind") != "points":
            return

        # Remove current actor
        try:
            actor = rec.get("actor_points")
            if actor is not None and hasattr(self, "plotter"):
                self.plotter.remove_actor(actor)
        except Exception:
            pass
        rec["actor_points"] = None

        # Rebuild via helper if available
        try:
            actor = None
            if hasattr(self, "_add_points_by_mode"):
                actor = self._add_points_by_mode(
                    rec.get("pdata"),
                    rec.get("has_rgb", False),
                    rec.get("solid_color", getattr(self, "_solid_fallback", (200, 200, 200))),
                    rec.get("color_mode", getattr(self, "_color_mode", "Solid")),
                    rec.get("cmap", getattr(self, "_cmap", "viridis")),
                    rec.get("points_as_spheres", getattr(self, "_points_as_spheres", False)),
                )
            rec["actor_points"] = actor
            # Hide scalar bar
            if hasattr(self, "set_colorbar_mode"):
                self.set_colorbar_mode("hidden")
            # Restore point size
            try:
                if actor is not None:
                    prop = actor.GetProperty()
                    if prop:
                        prop.SetPointSize(int(rec.get("point_size", getattr(self, "_point_size", 3))))
            except Exception:
                pass
            if hasattr(self, "plotter"):
                self.plotter.render()
        except Exception:
            pass
        
    def set_mesh_visibility(self, dataset_index: int, visible: bool) -> None:
        """Show or hide a mesh dataset."""
        try:
            rec = self._datasets[dataset_index]
        except Exception:
            return
        if rec.get("kind") != "mesh":
            return
        rec["visible"] = bool(visible)
        actor = rec.get("actor_mesh")
        if not visible and actor is not None:
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
            rec["actor_mesh"] = None
            try:
                self.plotter.update()
            except Exception:
                pass
            return
        if visible and actor is None:
            try:
                rec["actor_mesh"] = self._add_mesh_no_bar(
                    rec.get("mesh"),
                    color=rec.get("solid_color", self._solid_fallback),
                    style="wireframe"
                    if rec.get("representation", "surface") == "wireframe"
                    else "surface",
                    opacity=float(rec.get("opacity", 100)) / 100.0,
                )
                self.plotter.update()
            except Exception:
                pass

    def set_mesh_opacity(self, dataset_index: int, opacity: int) -> None:
        """Set mesh opacity (0-100)."""
        try:
            rec = self._datasets[dataset_index]
        except Exception:
            return
        if rec.get("kind") != "mesh":
            return
        rec["opacity"] = max(0, min(100, int(opacity)))
        actor = rec.get("actor_mesh")
        if actor is not None:
            try:
                prop = actor.GetProperty()
                if prop:
                    prop.SetOpacity(rec["opacity"] / 100.0)
                self.plotter.update()
            except Exception:
                pass
    
    def _apply_budget_to_polydata(self, full_pdata, has_rgb: bool, percent: int = 100):
        """Return a PolyData respecting view budget with smooth % changes.

        Args:
            full_pdata: PolyData with all points.
            has_rgb: True if dataset has RGB values.
            percent: Desired visible percent (1-100).

        Uses a deterministic hash-per-point probability test so tiny percent
        changes produce proportionally small visual changes (no big jumps),
        and results are stable across refreshes.
        """
        try:
            import numpy as np
            import pyvista as pv
        except Exception:
            return full_pdata

        pct = max(1, min(100, int(percent)))
        if pct >= 100:
            return full_pdata

        pts = np.asarray(full_pdata.points)
        n = pts.shape[0]
        if n == 0:
            return full_pdata

        # Quantize coordinates to integers to build a fast, deterministic hash
        q = np.round(pts * 1e4).astype(np.int64)  # 0.1 mm se unità metri
        # 32-bit mix
        h = (
            (q[:, 0] * 73856093) ^
            (q[:, 1] * 19349663) ^
            (q[:, 2] * 83492791)
        ) & 0xFFFFFFFF
        # Bucket 0..99
        bucket = (h % 100).astype(np.int32)
        mask = bucket < pct
        if not np.any(mask):
            mask[0] = True

        out = pv.PolyData(pts[mask])

        # Copia arrays allineati
        pd = full_pdata.point_data
        for key in list(pd.keys()):
            try:
                arr = np.asarray(pd[key])
                if arr.shape[0] == n:
                    out.point_data[key] = arr[mask]
            except Exception:
                pass

        return out
    
        # ---- Internals: normals helpers ---------------------------------------
    def _dataset_diag(self, pdata) -> float:
        """Return scene-space diagonal length for a PolyData's bounds."""
        try:
            import numpy as np
            pts = pdata.points
            if pts.size == 0:
                return 1.0
            bb = np.array([pts.min(axis=0), pts.max(axis=0)])
            return float(np.linalg.norm(bb[1] - bb[0])) or 1.0
        except Exception:
            return 1.0

    def _normals_subset(self, pts, nrm, percent: int):
        """Return a deterministic subset of points+normals given a percentage."""
        import numpy as np
        p = max(1, min(100, int(percent)))
        n = pts.shape[0]
        if n <= 20000 and p >= 100:
            return pts, nrm
        q = np.round(pts * 1e4).astype(np.int64)
        h = ((q[:, 0] * 73856093) ^ (q[:, 1] * 19349663) ^ (q[:, 2] * 83492791)) & 0xFFFFFFFF
        mask = (h % 100) < p
        if not mask.any():
            mask[0] = True
        return pts[mask], nrm[mask]

    def _normals_colors(self, pts, nrm, style: str, uniform_rgb: tuple[float, float, float]):
        """Compute per-glyph colors according to the selected style."""
        import numpy as np
        if style == "Axis RGB":
            c = np.abs(nrm[:, :3])
            c = c / np.maximum(1e-12, c.max(axis=1, keepdims=True))
            return c
        if style == "RGB Components":
            c = (nrm[:, :3] * 0.5) + 0.5
            return c
        return None

    import numpy as _np
    import pyvista as _pv

    def _get_normals_array(self, pdata) -> _np.ndarray | None:
        # 1) proprietà comoda
        import numpy as _np
        import pyvista as _pv
        try:
            n = getattr(pdata, "point_normals", None)
            if n is not None:
                n = _np.asarray(n)
                if n.ndim == 2 and n.shape[1] == 3:
                    return n
        except Exception:
            pass
        # 2) chiavi comuni
        try:
            pcd = getattr(pdata, "point_data", {})
            for key in ("Normals", "normals", "PointNormals"):
                if key in pcd:
                    n = _np.asarray(pcd[key])
                    if n.ndim == 2 and n.shape[1] == 3:
                        return n
        except Exception:
            pass
        return None

    def _rebuild_normals_actor(self, ds: int,
                            style: str = "Uniform",
                            color: tuple[float, float, float] = (1.0, 0.2, 0.2),
                            percent: int = 10,
                            scale: int = 30) -> None:
        """(Re)build normals glyph actor for dataset `ds`."""
        import numpy as _np
        import pyvista as _pv
        try:
            rec = self._datasets[ds]
        except Exception:
            return

        # Remove previous
        try:
            if rec.get("actor_normals") is not None:
                self.plotter.remove_actor(rec["actor_normals"])
        except Exception:
            pass
        rec["actor_normals"] = None

        pdata = rec.get("pdata")
        if pdata is None or getattr(pdata, "n_points", 0) == 0:
            print("[Normals] pdata missing or empty")
            return

        pdata_view = rec.get("pdata")
        full_pdata = rec.get("full_pdata", pdata_view)

        if full_pdata is None or getattr(full_pdata, "n_points", 0) == 0:
            print("[Normals] pdata missing or empty")
            return

        # Points from FULL dataset (aligns best with worker-produced normals)
        P = _np.asarray(full_pdata.points, dtype=float)

        # Normals priority:
        # 1) result saved by the worker (rec["normals_array"])
        # 2) arrays already present on full_pdata
        # 3) arrays on the budgeted view (if any)
        # 4) as last resort, compute in place on full_pdata
        N = rec.get("normals_array", None)

        if N is None:
            N = self._get_normals_array(full_pdata)

        if N is None and pdata_view is not None:
            N = self._get_normals_array(pdata_view)

        if N is None:
            try:
                full_pdata.compute_normals(point_normals=True, inplace=True)
                N = self._get_normals_array(full_pdata)
            except Exception:
                N = None

        if N is None:
            print("[Normals] No normals array found")
            return

        # Clean + normalize, reconcile lengths
        N = _np.asarray(N, dtype=float)
        P = _np.asarray(P, dtype=float)

        nP = int(P.shape[0]) if P.ndim == 2 and P.shape[1] == 3 else 0
        nN = int(N.shape[0]) if N.ndim == 2 and N.shape[1] == 3 else 0
        if nP == 0 or nN == 0:
            return
        if nP != nN:
            m = min(nP, nN)
            P = P[:m]
            N = N[:m]

        ok = _np.isfinite(N).all(axis=1)
        ok &= _np.isfinite(P).all(axis=1)
        if not ok.any():
            return
        N = N[ok]
        P = P[ok]
        if N.size == 0 or P.size == 0:
            return

        # Normalize normals
        norm = _np.linalg.norm(N, axis=1)
        eps = 1e-12
        nz = norm > eps
        N[nz] = (N[nz].T / norm[nz]).T
        N[~nz] = _np.array([0.0, 0.0, 1.0])

        npts = P.shape[0]
        perc = int(max(1, min(100, percent)))
        nsamp = max(1, int(round(npts * (perc / 100.0))))
        rng = _np.random.default_rng(12345 + int(ds))
        idx = rng.choice(npts, size=nsamp, replace=False) if nsamp < npts else _np.arange(npts)

        scl = float(max(1, min(200, int(scale)))) * 0.01

        try:
            # Compute a stable glyph length based on dataset diagonal (scale slider: 1..200 -> /1000)
            diag = self._dataset_diag(full_pdata)
            factor = max(diag * (float(scale) / 1000.0), 1e-9)

            # Build arrow glyphs explicitly so we can attach per-glyph RGB
            centers = _pv.PolyData(P[idx])
            centers["vectors"] = N[idx]
            arrow = _pv.Arrow()  # default small arrow geometry
            glyph = centers.glyph(orient="vectors", scale=False, factor=factor, geom=arrow)

            # Decide coloring
            if style == "Uniform":
                # Single uniform color
                actor = self._add_mesh_no_bar(
                    glyph,
                    color=(float(color[0]), float(color[1]), float(color[2])),
                    name=f"normals_ds{ds}",
                )
            # --- dentro _rebuild_normals_actor, nel blocco "else:" per gli stili non 'Uniform' ---
            else:
                # Per-glyph RGB based on style (Axis RGB / RGB Components)
                C = self._normals_colors(P[idx], N[idx], style, color)
                if C is None:
                    # Fallback a colore uniforme se stile non riconosciuto
                    actor = self._add_mesh_no_bar(
                        glyph,
                        color=(float(color[0]), float(color[1]), float(color[2])),
                        name=f"normals_ds{ds}",
                    )
                else:
                    import numpy as _np
                    # RGB per freccia in [0..255] uint8
                    rgb = _np.asarray(_np.clip(_np.round(C * 255.0), 0, 255), dtype=_np.uint8)

                    k = rgb.shape[0]  # numero di frecce (= nsamp)
                    # Tentativo 1: per-cella
                    try:
                        cells_per = max(1, int(glyph.n_cells // max(1, k)))
                        rgb_cells = _np.repeat(rgb, cells_per, axis=0)
                        # Allinea esattamente alla lunghezza richiesta
                        if rgb_cells.shape[0] < glyph.n_cells:
                            pad = _np.repeat(rgb[-1:], glyph.n_cells - rgb_cells.shape[0], axis=0)
                            rgb_cells = _np.concatenate([rgb_cells, pad], axis=0)
                        elif rgb_cells.shape[0] > glyph.n_cells:
                            rgb_cells = rgb_cells[:glyph.n_cells]

                        glyph.cell_data["RGB"] = rgb_cells
                        actor = self._add_mesh_no_bar(
                            glyph,
                            scalars="RGB",
                            rgb=True,
                            name=f"normals_ds{ds}",
                        )
                    except Exception:
                        # Tentativo 2: per-punto
                        try:
                            pts_per = max(1, int(glyph.n_points // max(1, k)))
                            rgb_pts = _np.repeat(rgb, pts_per, axis=0)
                            if rgb_pts.shape[0] < glyph.n_points:
                                pad = _np.repeat(rgb[-1:], glyph.n_points - rgb_pts.shape[0], axis=0)
                                rgb_pts = _np.concatenate([rgb_pts, pad], axis=0)
                            elif rgb_pts.shape[0] > glyph.n_points:
                                rgb_pts = rgb_pts[:glyph.n_points]

                            glyph.point_data["RGB"] = rgb_pts
                            actor = self._add_mesh_no_bar(
                                glyph,
                                scalars="RGB",
                                rgb=True,
                                name=f"normals_ds{ds}",
                            )
                        except Exception:
                            # Ultimo fallback: colore uniforme
                            actor = self._add_mesh_no_bar(
                                glyph,
                                color=(float(color[0]), float(color[1]), float(color[2])),
                                name=f"normals_ds{ds}",
                            )

            rec["actor_normals"] = actor
            rec["normals_visible"] = True
            rec["normals_style"] = style
            rec["normals_color"] = color
            rec["normals_percent"] = perc
            rec["normals_scale"] = int(scale)

            try:
                self.plotter.update()
            except Exception:
                pass
            # print(f"[Normals] ds={ds} points={npts} shown={idx.size} factor={factor:.6f}")
        except Exception as ex:
            print(f"[Normals] normals glyph failed: {ex}")
            rec["actor_normals"] = None
            rec["normals_visible"] = False

    def set_point_budget(self, percent: int, dataset_index: int | None = None) -> None:
        """Set visible percent for points globally or for a specific dataset."""
        try:
            p = int(percent)
        except Exception:
            return
        p = max(1, min(100, p))

        if dataset_index is None:
            self._view_budget_percent = p
            for idx, rec in enumerate(self._datasets):
                if rec.get("kind") != "points":
                    continue
                self.set_point_budget(p, idx)
            return

        try:
            rec = self._datasets[dataset_index]
        except Exception:
            return
        if rec.get("kind") != "points":
            return

        rec["view_percent"] = p
        full_pdata = rec.get("full_pdata", rec.get("pdata"))
        pdata_for_view = self._apply_budget_to_polydata(
            full_pdata, rec.get("has_rgb", False), p
        )
        rec["pdata"] = pdata_for_view
        actor = rec.get("actor_points")
        if actor is not None:
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        rec["actor_points"] = None
        if rec.get("visible", True):
            rec["actor_points"] = self._add_points_by_mode(
                pdata_for_view,
                rec.get("has_rgb", False),
                rec.get("solid_color", self._solid_fallback),
                rec.get("color_mode", self._color_mode),
                rec.get("cmap", self._cmap),
                rec.get("points_as_spheres", self._points_as_spheres),
            )
            actor = rec.get("actor_points")
            if actor is not None:
                try:
                    prop = actor.GetProperty()
                    if prop:
                        prop.SetPointSize(rec.get("point_size", self._point_size))
                except Exception:
                    pass
        try:
            self.plotter.update()
        except Exception:
            pass