# make_vtk_testdata.py
# Create a small VTK test suite with points, lines, mesh and fields.
# Requires: pyvista (pip install pyvista)

from __future__ import annotations
import numpy as np
import pyvista as pv
from pathlib import Path

OUT_DIR = Path("./vtk_testdata").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 1) POLYDATA: points + a polyline + a triangulated surface
# ------------------------------------------------------------
# Random point cloud (small, fast to load)
rng = np.random.default_rng(42)
pts = rng.uniform(-1.0, 1.0, size=(300, 3))

# A simple polyline (helix)
t = np.linspace(0, 4*np.pi, 100)
helix = np.c_[0.5*np.cos(t), 0.5*np.sin(t), np.linspace(-0.8, 0.8, t.size)]

# Make a plane and triangulate it to get a surface
plane = pv.Plane(center=(0.0, 0.0, 0.0), direction=(0, 0, 1),
                 i_size=1.5, j_size=1.0, i_resolution=20, j_resolution=16).triangulate()

# Build PolyData with points
poly_pts = pv.PolyData(pts)

# Add a scalar field and a vector field on points
r = np.linalg.norm(pts, axis=1)
poly_pts.point_data["temperature"] = (1.0 - np.clip(r/np.max(r), 0, 1)) * 100.0  # 0..100
poly_pts.point_data["velocity"] = np.c_[np.sin(pts[:,0]*3), np.cos(pts[:,1]*3), np.sin(pts[:,2]*3)]

# Build a polyline from helix
polyline = pv.PolyData(helix)
# cells encoding: [npts, id0, id1, ..., idN]
n = helix.shape[0]
cells = np.empty(n+1, dtype=int)
cells[0] = n
cells[1:] = np.arange(n, dtype=int)
polyline.lines = cells
# simple scalar along the line
polyline.point_data["s"] = np.linspace(0.0, 1.0, n)

# Add normals on the surface and a cell scalar
plane = plane.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)
plane.cell_data["material_id"] = np.repeat([1, 2], repeats=plane.n_cells//2).astype(int)[:plane.n_cells]
plane.point_data["thickness"] = (plane.points[:,0]**2 + plane.points[:,1]**2)**0.5

# Save polydata pieces
poly_pts.save(OUT_DIR / "test_points.vtp")
polyline.save(OUT_DIR / "test_polyline.vtp")
plane.save(OUT_DIR / "test_surface.vtp")

# Also pack them together into a single PolyData by appending (optional)
# Note: appending different topologies creates a MultiBlock by default;
# to force a single polydata, only append polydata of same topology.
poly_mb = pv.MultiBlock()
poly_mb["points"] = poly_pts
poly_mb["polyline"] = polyline
poly_mb["surface"] = plane
poly_mb.save(OUT_DIR / "test_polydata.vtm")

# ------------------------------------------------------------
# 2) UNSTRUCTURED GRID (VTU) with fields
# ------------------------------------------------------------
# Start from a small uniform grid and cast to unstructured
grid = pv.UniformGrid(dims=(16, 12, 8), spacing=(0.2, 0.25, 0.3), origin=(-1.6, -1.5, -1.2))
X, Y, Z = np.meshgrid(
    np.linspace(grid.bounds[0], grid.bounds[1], grid.dimensions[0]),
    np.linspace(grid.bounds[2], grid.bounds[3], grid.dimensions[1]),
    np.linspace(grid.bounds[4], grid.bounds[5], grid.dimensions[2]),
    indexing="ij",
)
# Scalar and vector fields on points
density = np.exp(-0.5*((X/1.2)**2 + (Y/1.1)**2 + (Z/0.9)**2))
grid.point_data["density"] = density.ravel(order="F")
grid.point_data["flow"] = np.c_[np.sin(X).ravel("F"), np.cos(Y).ravel("F"), np.sin(Z).ravel("F")]

ugrid = grid.cast_to_unstructured_grid(copy=True)
# Cell data example
ugrid.cell_data["region"] = np.random.default_rng(0).integers(1, 4, size=ugrid.n_cells, endpoint=True)

ugrid.save(OUT_DIR / "test_unstructured.vtu")

# ------------------------------------------------------------
# 3) IMAGE DATA (VTI) for volume rendering
# ------------------------------------------------------------
img = pv.ImageData(dimensions=(64, 48, 36), spacing=(0.12, 0.12, 0.12), origin=(-3.0, -2.2, -1.8))
gx, gy, gz = np.meshgrid(
    np.linspace(-1, 1, img.dimensions[0]),
    np.linspace(-1, 1, img.dimensions[1]),
    np.linspace(-1, 1, img.dimensions[2]),
    indexing="ij",
)
# Two blobs + noise
vol = (np.exp(-((gx+0.3)**2 + (gy+0.1)**2 + (gz+0.2)**2)*6.0)
       + 0.7*np.exp(-((gx-0.4)**2 + (gy-0.3)**2 + (gz+0.1)**2)*8.0)
       + 0.05*np.random.default_rng(1).normal(size=gx.shape))
img.point_data["density"] = vol.ravel(order="F")
img.save(OUT_DIR / "test_image.vti")

# ------------------------------------------------------------
# 4) MULTIBLOCK SCENE (VTM)
# ------------------------------------------------------------
scene = pv.MultiBlock()
scene["poly_points"] = pv.read(OUT_DIR / "test_points.vtp")
scene["poly_line"] = pv.read(OUT_DIR / "test_polyline.vtp")
scene["poly_surface"] = pv.read(OUT_DIR / "test_surface.vtp")
scene["unstructured"] = pv.read(OUT_DIR / "test_unstructured.vtu")
scene["volume"] = pv.read(OUT_DIR / "test_image.vti")
scene.save(OUT_DIR / "test_scene.vtm")

print(f"\nSaved test datasets in: {OUT_DIR}\n"
      f" - {OUT_DIR/'test_points.vtp'}\n"
      f" - {OUT_DIR/'test_polyline.vtp'}\n"
      f" - {OUT_DIR/'test_surface.vtp'}\n"
      f" - {OUT_DIR/'test_unstructured.vtu'}\n"
      f" - {OUT_DIR/'test_image.vti'}\n"
      f" - {OUT_DIR/'test_scene.vtm'} (open this for a full test)\n")