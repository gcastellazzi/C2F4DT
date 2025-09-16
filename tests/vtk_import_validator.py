#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VTK Import Validator for C2F4DT (with robust scalar mapping)

- OFF-SCREEN rendering (headless friendly)
- Representations: Points, Wireframe, Surface, Surface with Edges
- Solid color + Edges
- Colormaps (with invert)
- PointData → surface scalar mapping via vtkOriginalPointIds or KDTree fallback
- Vector array support: Magnitude / component

Usage:
  python tools/vtk_import_validator.py --vtk input.vtk --out _vtk_report

Requires:
  pip install pyvista numpy pillow scipy matplotlib
"""
from __future__ import annotations
import argparse, json, os, sys
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pyvista as pv

try:
    from scipy.spatial import cKDTree as KDTree
except Exception:
    KDTree = None

CMAPS = ["viridis", "turbo", "coolwarm", "plasma"]  # matplotlib names (lowercase)


def _screenshot(plotter: pv.Plotter, outdir: str, name: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{name}.png")
    try:
        plotter.screenshot(path)
    except Exception:
        try:
            plotter.render()
            plotter.screenshot(path)
        except Exception:
            return ""
    return path if os.path.exists(path) else ""


def _array_names(dataset: pv.DataSet) -> Tuple[List[str], List[str]]:
    pt = list(dataset.point_data.keys()) if hasattr(dataset, "point_data") else []
    cl = list(dataset.cell_data.keys()) if hasattr(dataset, "cell_data") else []
    return pt, cl


def _as_scalar(arr: np.ndarray, mode: str) -> np.ndarray:
    """Vector → scalar: Magnitude/X/Y/Z; passthrough for scalar arrays."""
    if arr.ndim == 2 and arr.shape[1] in (2, 3):
        mode = (mode or "Magnitude").title()
        if mode == "Magnitude":
            return np.linalg.norm(arr, axis=1)
        comp = {"X": 0, "Y": 1, "Z": 2}.get(mode, 0)
        comp = min(comp, arr.shape[1] - 1)
        return arr[:, comp]
    return arr


def _to_surface(mesh):
    """Return PolyData surface if possible."""
    if isinstance(mesh, pv.PolyData):
        return mesh
    try:
        if hasattr(mesh, "extract_surface"):
            s = mesh.extract_surface(pass_pointid=True, pass_cellid=True)
            if isinstance(s, pv.PolyData):
                return s
    except Exception:
        pass
    try:
        if hasattr(mesh, "extract_geometry"):
            g = mesh.extract_geometry()
            if isinstance(g, pv.PolyData):
                return g
    except Exception:
        pass
    return mesh


def _map_point_scalars_to_surface(
    mesh, surf, name: str, mode: str = "Magnitude"
) -> Optional[np.ndarray]:
    """Map mesh.point_data[name] → surf.points using OriginalPointIds or KDTree."""
    try:
        # prefer ID mapping if present
        ids = None
        if hasattr(surf, "point_data"):
            for key in ("vtkOriginalPointIds", "vtkOriginalPointID", "origids", "OriginalPointIds"):
                if key in surf.point_data:
                    ids = np.asarray(surf.point_data[key]).astype(np.int64)
                    break

        if name not in mesh.point_data:
            # some writers put it straight on the surface already
            if hasattr(surf, "point_data") and name in surf.point_data:
                base = np.asarray(surf.point_data[name])
                return _as_scalar(base, mode)
            return None

        base = np.asarray(mesh.point_data[name])
        base = _as_scalar(base, mode)

        if ids is not None:
            ids = np.clip(ids, 0, base.shape[0] - 1)
            return base[ids]

        # Fallback: KDTree nearest mapping by coordinates
        if KDTree is None:
            # last resort: if sizes coincide, passthrough
            return base if surf.n_points == mesh.n_points else None

        P_src = np.asarray(getattr(mesh, "points"))
        P_dst = np.asarray(getattr(surf, "points"))
        if P_src is None or P_dst is None or P_src.shape[0] == 0 or P_dst.shape[0] == 0:
            return None
        tree = KDTree(P_src)
        idx = tree.query(P_dst, k=1, workers=-1)[1]
        idx = np.clip(np.asarray(idx, dtype=np.int64), 0, base.shape[0] - 1)
        return base[idx]
    except Exception:
        return None


def _invert_cmap_name(name: str) -> str:
    """Return reversed cmap name if available (matplotlib), else original."""
    try:
        import matplotlib.pyplot as plt  # noqa: F401
        import matplotlib.cm as cm
        base = str(name).lower()
        if base.endswith("_r"):
            return base[:-2]
        if base in cm.cmap_d or base in cm.cmap_registry:
            rev = base + "_r"
            return rev
    except Exception:
        pass
    return name


def validate_vtk(path: str, outdir: str) -> Dict[str, Any]:
    rep_ok, lut_ok, arrays_ok, edges_ok, bar_ok = {}, {}, {}, {}, {}

    plotter = pv.Plotter(off_screen=True, window_size=(1200, 900))
    report = {
        "input": path,
        "outdir": outdir,
        "dataset_type": None,
        "n_points": None,
        "n_cells": None,
        "point_arrays": [],
        "cell_arrays": [],
        "screenshots": [],
        "checks": {},
    }

    try:
        mesh = pv.read(path)
    except Exception as ex:
        return {"error": f"Failed to read VTK: {ex}", "input": path}

    if isinstance(mesh, pv.MultiBlock):
        if len(mesh) == 0:
            return {"error": "Empty MultiBlock", "input": path}
        try:
            mesh = mesh[0]
        except Exception:
            return {"error": "Failed to access first block", "input": path}

    report["dataset_type"] = type(mesh).__name__
    report["n_points"] = int(getattr(mesh, "n_points", 0))
    report["n_cells"] = int(getattr(mesh, "n_cells", 0))

    pt_names, cl_names = _array_names(mesh)
    report["point_arrays"] = pt_names
    report["cell_arrays"] = cl_names

    surf = _to_surface(mesh)

    actor = plotter.add_mesh(surf, name="dataset", show_edges=False)
    plotter.add_text("VTK Import Validator", font_size=10)
    plotter.view_isometric(); plotter.reset_camera()
    shot = _screenshot(plotter, outdir, "00_baseline")
    if shot: report["screenshots"].append(shot)

    # Representations
    reps = [
        ("Points", {"style": "points"}),
        ("Wireframe", {"style": "wireframe"}),
        ("Surface", {"style": None}),
        ("Surface with Edges", {"style": None, "show_edges": True}),
    ]
    for name, opts in reps:
        try:
            plotter.remove_actor(actor)
        except Exception:
            pass
        try:
            actor = plotter.add_mesh(
                surf,
                style=opts.get("style", None),
                show_edges=opts.get("show_edges", False),
                name=f"rep_{name}",
            )
            plotter.reset_camera()
            _shot = _screenshot(plotter, outdir, f"rep_{name.replace(' ', '_')}")
            rep_ok[name] = bool(_shot)
            if _shot:
                report["screenshots"].append(_shot)
        except Exception:
            rep_ok[name] = False

    # Solid color + edges
    try:
        plotter.remove_actor(actor)
    except Exception:
        pass
    actor = plotter.add_mesh(surf, name="solid", style=None, show_edges=False, opacity=1.0)
    try:
        actor.prop.color = (0.7, 0.7, 0.85)
    except Exception:
        pass
    plotter.reset_camera()
    report["screenshots"].append(_screenshot(plotter, outdir, "solid_color") or "")
    edges_ok["off"] = True

    try:
        plotter.remove_actor(actor)
        actor = plotter.add_mesh(surf, name="solid_edges", style=None, show_edges=True, opacity=1.0)
        plotter.reset_camera()
        report["screenshots"].append(_screenshot(plotter, outdir, "solid_with_edges") or "")
        edges_ok["on"] = True
    except Exception:
        edges_ok["on"] = False

    # LUTs + scalar bar
    scalar_source, scalar_name = (None, None)
    if pt_names:
        scalar_source, scalar_name = "POINT", pt_names[0]
    elif cl_names:
        scalar_source, scalar_name = "CELL", cl_names[0]

    if scalar_name:
        arrays_ok["picked"] = f"{scalar_source}/{scalar_name}"
        for cmap in CMAPS:
            for invert in (False, True):
                label = f"{cmap}_{'inv' if invert else 'norm'}"
                try:
                    plotter.remove_actor(actor)
                except Exception:
                    pass
                try:
                    scalars = None
                    clim = None

                    if scalar_source == "POINT":
                        scalars = _map_point_scalars_to_surface(mesh, surf, scalar_name, "Magnitude")
                    else:
                        # TODO: implement cell mapping if needed (vtkOriginalCellIds + per-face strategy)
                        scalars = None

                    if scalars is None:
                        lut_ok[label] = False
                        continue

                    if isinstance(scalars, np.ndarray):
                        clim = (float(np.nanmin(scalars)), float(np.nanmax(scalars)))

                    cm_name = cmap
                    if invert:
                        cm_name = _invert_cmap_name(cm_name)

                    actor = plotter.add_mesh(
                        surf,
                        scalars=scalars,
                        cmap=cm_name,
                        clim=clim,
                        show_edges=False,
                        scalar_bar_args={"title": scalar_name},
                        name=f"lut_{label}",
                    )

                    plotter.reset_camera()
                    shot = _screenshot(plotter, outdir, f"lut_{label}")
                    lut_ok[label] = bool(shot)
                    if shot:
                        report["screenshots"].append(shot)
                except Exception:
                    lut_ok[label] = False

        # scalar bar toggle
        try:
            plotter.remove_scalar_bar(); bar_ok["off"] = True
        except Exception:
            bar_ok["off"] = False
        try:
            plotter.add_scalar_bar(title=scalar_name); bar_ok["on"] = True
            report["screenshots"].append(_screenshot(plotter, outdir, "scalar_bar_on") or "")
        except Exception:
            bar_ok["on"] = False
    else:
        arrays_ok["picked"] = None

    report["checks"]["representations"] = rep_ok
    report["checks"]["edges"] = edges_ok
    report["checks"]["luts"] = lut_ok
    report["checks"]["arrays"] = arrays_ok
    report["checks"]["scalar_bar"] = bar_ok

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vtk", required=True, help="Path to input VTK/VTU/VTP/etc.")
    ap.add_argument("--out", required=True, help="Output dir for screenshots and JSON")
    args = ap.parse_args()

    rep = validate_vtk(args.vtk, args.out)
    if "error" in rep:
        print("[FAIL]", rep["error"]); sys.exit(2)
    print("[OK] Report saved to:", args.out); sys.exit(0)


if __name__ == "__main__":
    main()