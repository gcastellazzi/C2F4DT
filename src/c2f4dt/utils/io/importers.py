# c2f44DT/utils/io/importers.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Optional backends
try:  # Mesh & many point formats via VTK/PyVista
    import pyvista as pv  # type: ignore
    _HAS_PYVISTA = True
except Exception:  # pragma: no cover
    pv = None  # type: ignore
    _HAS_PYVISTA = False

try:  # LAS/LAZ
    import laspy  # type: ignore
    _HAS_LASPY = True
except Exception:  # pragma: no cover
    laspy = None  # type: ignore
    _HAS_LASPY = False

try:  # E57
    import pye57  # type: ignore
    _HAS_E57 = True
except Exception:  # pragma: no cover
    pye57 = None  # type: ignore
    _HAS_E57 = False


# -------------------- Downsampling utilities --------------------
def downsample_random(points: np.ndarray, percent: float) -> np.ndarray:
    """Randomly downsample points to the given percentage.

    Args:
        points: (N, 3) float array.
        percent: Target percentage in [1, 100].

    Returns:
        Indices of selected points (1D int array).
    """
    if points.size == 0:
        return np.empty((0,), dtype=np.int64)
    p = np.clip(percent, 1.0, 100.0) / 100.0
    n = points.shape[0]
    k = max(1, int(round(n * p)))
    idx = np.random.default_rng().choice(n, size=k, replace=False)
    return np.sort(idx)


def downsample_voxel_auto(points: np.ndarray, target_percent: float) -> np.ndarray:
    """Voxel-grid downsampling using a simple auto voxel-size heuristic.

    Heuristic: scala la dimensione del voxel con l'estensione del bounding-box e
    il fattore di riduzione desiderato. La % risultante è approssimata.

    Args:
        points: (N, 3) float array.
        target_percent: Target percentage in [1, 100].

    Returns:
        Indices of representative points (1D int array).
    """
    n = points.shape[0]
    if n == 0:
        return np.empty((0,), dtype=np.int64)
    p = np.clip(target_percent, 1.0, 100.0) / 100.0
    if p >= 0.999:
        return np.arange(n, dtype=np.int64)

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    extent = np.maximum(maxs - mins, 1e-9)

    reduction = max(1e-3, 1.0 - p)
    base = float(extent.max())
    voxel = base * reduction * 0.02  # 2% bbox @50% riduzione
    voxel = max(voxel, base * 1e-4)

    q = np.floor((points - mins) / voxel).astype(np.int64)
    key = (q[:, 0] * 73856093) ^ (q[:, 1] * 19349663) ^ (q[:, 2] * 83492791)
    order = np.argsort(key, kind="mergesort")
    key_sorted = key[order]
    mask = np.concatenate(([True], key_sorted[1:] != key_sorted[:-1]))
    kept_sorted = order[mask]
    return np.sort(kept_sorted)


@dataclass
class ImportedObject:
    """Container for imported datasets (point cloud or mesh)."""

    kind: str  # "points" | "mesh"
    name: str
    points: Optional[np.ndarray] = None        # (N, 3) float64
    colors: Optional[np.ndarray] = None        # (N, 3) float32 in [0,1]
    intensity: Optional[np.ndarray] = None     # (N,) float32
    normals: Optional[np.ndarray] = None       # (N, 3) float32 unit vectors
    faces: Optional[np.ndarray] = None         # (M, 3) int32 (triangles)
    pv_mesh: Optional[object] = None           # pyvista.PolyData or similar
    meta: Dict[str, object] = field(default_factory=dict)

    def bounds(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """Return (xmin, xmax, ymin, ymax, zmin, zmax) if available."""
        if self.points is not None and len(self.points):
            mins = self.points.min(axis=0)
            maxs = self.points.max(axis=0)
            return (
                float(mins[0]), float(maxs[0]),
                float(mins[1]), float(maxs[1]),
                float(mins[2]), float(maxs[2])
            )
        if self.pv_mesh is not None and _HAS_PYVISTA:
            try:
                return tuple(self.pv_mesh.bounds)  # type: ignore[return-value]
            except Exception:
                return None
        return None


def import_file(path: str) -> List[ImportedObject]:
    """Import a geometry file into one or more `ImportedObject`.

    Auto-detects reader by extension and available backends.

    Args:
        path: File path.

    Returns:
        List of imported objects.

    Raises:
        ValueError: On unsupported type or missing backend.
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext in {".ply", ".obj", ".vtp", ".stl", ".vtk", ".gltf", ".glb"}:
        return _import_with_pyvista(p)

    if ext in {".las", ".laz"}:
        return _import_las(p)

    if ext in {".e57"}:
        return _import_e57(p)

    raise ValueError(f"Unsupported file type: {ext}")


# -------------------- PyVista / VTK --------------------
def _import_with_pyvista(p: Path) -> List[ImportedObject]:
    """Import using PyVista/VTK readers.

    Supports meshes and point clouds. Tries to extract RGB, intensity, normals.
    """
    if not _HAS_PYVISTA:
        raise ValueError("PyVista/VTK backend not available. Please install pyvista and vtk.")
    try:
        mesh = pv.read(str(p))
    except Exception as ex:  # pragma: no cover
        raise ValueError(f"Failed to read with PyVista: {ex}")

    objs: List[ImportedObject] = []

    # Count points/faces
    try:
        n_points = int(mesh.n_points)
        n_faces = int(mesh.n_faces) if hasattr(mesh, "n_faces") else 0
    except Exception:
        n_points, n_faces = 0, 0

    # Extract colors/intensity/normals if present
    colors = None
    intensity = None
    normals = None
    try:
        pd = getattr(mesh, "point_data", {})
        # Colors
        for key in ("RGB", "rgb", "Colors", "colors"):
            if key in pd:
                arr = np.asarray(pd[key])
                if arr.ndim == 2 and arr.shape[1] >= 3:
                    a = arr[:, :3].astype(np.float32)
                    if a.max() > 1.5:
                        a /= 255.0
                    colors = a
                    break
        # Intensity
        for key in ("intensity", "Intensity", "Scalar", "scalars"):
            if key in pd:
                arr = np.asarray(pd[key]).astype(np.float32)
                intensity = arr
                break
        # Normals (common names or active normals)
        for key in ("Normals", "normals", "NxNyNz", "N", "n", "Normal"):
            if key in pd:
                arr = np.asarray(pd[key])
                if arr.ndim == 2 and arr.shape[1] >= 3:
                    normals = arr[:, :3].astype(np.float32)
                    break
        if normals is None and hasattr(mesh, "point_normals") and mesh.point_normals is not None:
            n = np.asarray(mesh.point_normals)
            if n.ndim == 2 and n.shape[1] >= 3:
                normals = n[:, :3].astype(np.float32)
    except Exception:
        pass

    meta = {
        "backend": "pyvista",
        "file": str(p),
        "n_points": n_points,
        "n_faces": n_faces,
        "arrays": list(getattr(mesh, "point_data", {}).keys()) if hasattr(mesh, "point_data") else [],
        "has_intensity": intensity is not None,
        "has_normals": normals is not None,
    }

    if n_faces == 0:
        pts = np.asarray(mesh.points, dtype=np.float64)
        objs.append(
            ImportedObject(
                kind="points",
                name=p.name,
                points=pts,
                colors=colors,
                intensity=intensity,
                normals=normals,
                meta=meta,
            )
        )
    else:
        # Faces as triangles if available (vtk face layout: n,id0,id1,id2,...)
        faces = None
        try:
            if hasattr(mesh, "faces") and isinstance(mesh.faces, np.ndarray):
                fa = mesh.faces.reshape(-1, 4)[:, 1:4]
                faces = fa.astype(np.int32, copy=False)
        except Exception:
            faces = None
        objs.append(
            ImportedObject(kind="mesh", name=p.name, pv_mesh=mesh, faces=faces, meta=meta)
        )

    return objs


# -------------------- LAS / LAZ --------------------
def _import_las(p: Path) -> List[ImportedObject]:
    """Import LAS/LAZ with laspy; extracts XYZ, optional RGB and intensity."""
    if not _HAS_LASPY:
        raise ValueError("LAS/LAZ backend not available. Please install laspy.")
    try:
        with laspy.open(str(p)) as f:
            _ = f.header  # kept for future meta
            pts = f.read()
        xyz = np.vstack([pts.x, pts.y, pts.z]).T.astype(np.float64)
        colors = None
        if hasattr(pts, "red") and hasattr(pts, "green") and hasattr(pts, "blue"):
            r = np.asarray(pts.red, dtype=np.float32)
            g = np.asarray(pts.green, dtype=np.float32)
            b = np.asarray(pts.blue, dtype=np.float32)
            scale = max(r.max(), g.max(), b.max(), 1.0)
            colors = np.vstack([r, g, b]).T / scale
        intensity = None
        if hasattr(pts, "intensity"):
            intensity = np.asarray(pts.intensity, dtype=np.float32)
        meta = {
            "backend": "laspy",
            "file": str(p),
            "n_points": int(xyz.shape[0]),
            "has_rgb": colors is not None,
            "has_intensity": intensity is not None,
            "has_normals": False,
        }
        return [
            ImportedObject(
                kind="points",
                name=p.name,
                points=xyz,
                colors=colors,
                intensity=intensity,
                normals=None,
                meta=meta,
            )
        ]
    except Exception as ex:  # pragma: no cover
        raise ValueError(f"Failed to read LAS/LAZ: {ex}")


# -------------------- E57 --------------------
def _import_e57(p: Path) -> List[ImportedObject]:
    """Import E57 with pye57; extracts XYZ and optional RGB/intensity (first scan)."""
    if not _HAS_E57:
        raise ValueError("E57 backend not available. Please install pye57.")
    try:
        e57 = pye57.E57(str(p))  # type: ignore
        data = e57.read_scan(0)  # pick first scan
        xyz = np.vstack([data["cartesianX"], data["cartesianY"], data["cartesianZ"]]).T.astype(np.float64)
        colors = None
        if all(k in data for k in ("colorRed", "colorGreen", "colorBlue")):
            r = data["colorRed"].astype(np.float32)
            g = data["colorGreen"].astype(np.float32)
            b = data["colorBlue"].astype(np.float32)
            scale = max(r.max(), g.max(), b.max(), 1.0)
            colors = np.vstack([r, g, b]).T / scale
        intensity = None
        if "intensity" in data:
            intensity = data["intensity"].astype(np.float32)
        meta = {
            "backend": "pye57",
            "file": str(p),
            "n_points": int(xyz.shape[0]),
            "has_rgb": colors is not None,
            "has_intensity": intensity is not None,
            "has_normals": False,
        }
        return [
            ImportedObject(
                kind="points",
                name=p.name,
                points=xyz,
                colors=colors,
                intensity=intensity,
                normals=None,
                meta=meta,
            )
        ]
    except Exception as ex:  # pragma: no cover
        raise ValueError(f"Failed to read E57: {ex}")