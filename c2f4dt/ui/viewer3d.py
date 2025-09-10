from __future__ import annotations

from PySide6 import QtWidgets


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
        # Initial rendering cosmetics (best-effort)
        try:
            self.plotter.set_background("white")
            self.plotter.add_axes()
            self.plotter.show_bounds(grid="front", location="outer")
        except Exception:
            pass

        # State for display options
        self._point_size = 3
        self._budget = 100
        self._color_mode = "Solid"
        self._solid_color = (200, 200, 200)
        self._cmap = getattr(self, "_cmap", "viridis")

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
        
    # ---- Camera helpers ----
    def view_fit(self) -> None:
        """Fit camera to scene bounds."""
        try:
            self.plotter.reset_camera()
        except Exception:
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
            self.plotter.reset_camera()
        except Exception:
            pass

    def view_iso(self, positive: bool = True) -> None:
        """Set an isometric camera preset.

        Args:
            positive: If True, use the default isometric preset; otherwise use a mirrored variant.
        """
        try:
            if positive:
                self.plotter.view_isometric()
            else:
                self.plotter.camera_position = [(1, 1, 1), (0, 0, 0), (0, 0, 1)]
                self.plotter.reset_camera()
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
        """Request a render update."""
        try:
            self.plotter.update()
        except Exception:
            pass

    # ---- Display controls ----
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

    def set_color_mode(self, mode: str, dataset_index: int | None = None) -> None:
        """Set color mode globally or for a specific point dataset."""
        if dataset_index is None:
            self._color_mode = mode
            for idx, rec in enumerate(self._datasets):
                if rec.get("kind") == "points":
                    self.set_color_mode(mode, idx)
            return
        try:
            rec = self._datasets[dataset_index]
        except Exception:
            return
        if rec.get("kind") != "points":
            return
        rec["color_mode"] = mode
        actor = rec.get("actor_points")
        if actor is not None:
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
            rec["actor_points"] = None
        if rec.get("visible", True):
            rec["actor_points"] = self._add_points_by_mode(
                rec.get("pdata"),
                rec.get("has_rgb", False),
                rec.get("solid_color", self._solid_fallback),
                mode,
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

    def set_solid_color(self, r: int, g: int, b: int) -> None:
        """Set a solid RGB color on all actors (best-effort)."""
        self._solid_color = (r, g, b)
        self._refresh_datasets()

    def set_colormap(self, name: str, dataset_index: int | None = None) -> None:
        """Set colormap globally or for a specific dataset when in colormap mode."""
        if dataset_index is None:
            self._cmap = name
            for idx, rec in enumerate(self._datasets):
                if rec.get("kind") == "points":
                    self.set_colormap(name, idx)
            return
        try:
            rec = self._datasets[dataset_index]
        except Exception:
            return
        if rec.get("kind") != "points":
            return
        rec["cmap"] = name
        if rec.get("color_mode", self._color_mode) != "Normal Colormap" and rec.get("has_rgb", False):
            return
        actor = rec.get("actor_points")
        if actor is not None:
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        rec["actor_points"] = None
        if rec.get("visible", True):
            rec["actor_points"] = self._add_points_by_mode(
                rec.get("pdata"),
                rec.get("has_rgb", False),
                rec.get("solid_color", self._solid_fallback),
                rec.get("color_mode", self._color_mode),
                name,
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
            }
            self._datasets.append(rec)
            ds_index = len(self._datasets) - 1

            self.plotter.reset_camera()
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
            )
        except Exception:
            # Fallback finale: Solid
            try:
                return self.plotter.add_points(
                    pdata,
                    color=solid,
                    render_points_as_spheres=spheres,
                )
            except Exception:
                return None

    def _refresh_datasets(self) -> None:
        """Rebuild all point actors according to current rendering settings.

        Preserves the internal registry schema and restores normals visibility if it was active.
        """
        try:
            # Save camera
            try:
                cam_pos = self.plotter.camera_position
            except Exception:
                cam_pos = None

            # Clear all and re-add
            self.plotter.clear()
            new_list = []
            for old_rec in self._datasets:
                kind = old_rec.get("kind", "points")
                if kind == "mesh":
                    actor_mesh = None
                    if old_rec.get("visible", True):
                        actor_mesh = self.plotter.add_mesh(
                            old_rec.get("mesh"),
                            color=old_rec.get("solid_color", self._solid_fallback),
                            style="wireframe"
                            if old_rec.get("representation", "surface") == "wireframe"
                            else "surface",
                            opacity=float(old_rec.get("opacity", 100)) / 100.0,
                        )
                    new_rec = dict(old_rec)
                    new_rec["actor_mesh"] = actor_mesh
                    new_list.append(new_rec)
                    continue

                full_pdata = old_rec.get("full_pdata", old_rec.get("pdata"))
                has_rgb = bool(old_rec.get("has_rgb", False))
                was_visible = bool(old_rec.get("visible", True))
                view_pct = int(old_rec.get("view_percent", 100))

                pdata_for_view = self._apply_budget_to_polydata(
                    full_pdata, has_rgb, view_pct
                )

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
                    try:
                        if actor_points is not None:
                            prop = actor_points.GetProperty()
                            if prop:
                                prop.SetPointSize(old_rec.get("point_size", self._point_size))
                    except Exception:
                        pass

                new_rec = {
                    "kind": "points",
                    "full_pdata": full_pdata,
                    "pdata": pdata_for_view,
                    "actor_points": actor_points,
                    "actor_normals": None,
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
                }
                new_list.append(new_rec)

                # If normals actor was visible before, restore it when possible
                try:
                    had_normals = bool(old_rec.get("actor_normals") is not None)
                except Exception:
                    had_normals = False
                if had_normals and hasattr(pdata_for_view, "point_data") and "Normals" in pdata_for_view.point_data:
                    idx = len(new_list) - 1
                    self.set_normals_visibility(idx, True)

            self._datasets = new_list

            # Re-apply current point size to all point actors after rebuild
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
            self.plotter.update()
        except Exception:
            pass

    def add_pyvista_mesh(self, mesh) -> int:
        """Add a PyVista mesh (PolyData) to the scene."""
        try:
            actor = self.plotter.add_mesh(mesh)
            rec = {
                "kind": "mesh",
                "mesh": mesh,
                "actor_mesh": actor,
                "visible": True,
                "representation": "surface",
                "opacity": 100,
                "solid_color": (1.0, 1.0, 1.0),
            }
            self._datasets.append(rec)
            self.plotter.reset_camera()
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
    
    def set_normals_visibility(self, dataset_index: int, visible: bool, scale: float = 0.02) -> None:
        """Show/hide normals glyphs for a given dataset."""
        try:
            rec = self._datasets[dataset_index]
        except Exception:
            return

        try:
            import numpy as np
            pdata = rec["pdata"]
            if not visible:
                if rec["actor_normals"] is not None:
                    try:
                        self.plotter.remove_actor(rec["actor_normals"])
                    except Exception:
                        pass
                    rec["actor_normals"] = None
                    self.plotter.update()
                return

            if "Normals" not in pdata.point_data:
                return

            if rec["actor_normals"] is None:
                pts = pdata.points
                nrm = pdata.point_data["Normals"]
                if pts.shape[0] == 0:
                    return
                bb = np.array([pts.min(axis=0), pts.max(axis=0)])
                diag = float(np.linalg.norm(bb[1] - bb[0]))
                glyph_len = max(1e-9, diag * float(scale))

                n = pts.shape[0]
                step = max(1, int(np.ceil(n / 20000)))
                pts_s = pts[::step]
                nrm_s = nrm[::step]

                import pyvista as pv  # type: ignore
                arrows = pv.Arrow(start=pts_s, direction=nrm_s, scale=glyph_len)
                rec["actor_normals"] = self.plotter.add_mesh(arrows, color=(0.9, 0.9, 0.2))
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
                rec["actor_mesh"] = self.plotter.add_mesh(
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

    def set_mesh_representation(self, dataset_index: int, mode: str) -> None:
        """Change mesh rendering style (surface or wireframe)."""
        try:
            rec = self._datasets[dataset_index]
        except Exception:
            return
        if rec.get("kind") != "mesh":
            return
        rec["representation"] = mode.lower()
        actor = rec.get("actor_mesh")
        if actor is not None:
            try:
                prop = actor.GetProperty()
                if prop:
                    if rec["representation"] == "wireframe":
                        prop.SetRepresentationToWireframe()
                    else:
                        prop.SetRepresentationToSurface()
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
    
    # def set_point_budget(self, percent: int) -> None:
    #     """Set 1–100% of points to render per dataset (view-only LOD) and refresh."""
    #     try:
    #         p = int(percent)
    #     except Exception:
    #         return
    #     p = max(1, min(100, p))
    #     if getattr(self, "_view_budget_percent", 100) == p:
    #         return
    #     self._view_budget_percent = p
    #     self._refresh_datasets()
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