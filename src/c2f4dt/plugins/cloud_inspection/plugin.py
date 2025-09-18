"""
Cloud Inspection – Point cloud metrics plugin for C2F4DT.

Features
--------
- Opens a dialog to select metrics to compute on the active point cloud.
- Computes metrics (fast ones inline; heavier in a worker thread) with progress & cancel.
- Optionally integrates with a Metrics.py module (if present/importable).
- Adds a 32x32_cloud_inspection.png action on the right vertical toolbar to open the dialog.

This plugin is self-contained and mounts no panels; it shows a modal dialog on demand.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import time

from PySide6 import QtCore, QtWidgets, QtGui


# Short descriptions shown in the dialog when a metric is highlighted
_METRIC_DESCRIPTIONS = {
    "point_count": "Total number of points in the active cloud.",
    "bounds": "Axis-aligned bounding box: min/max for X, Y, Z.",
    "extent": "Box size along X/Y/Z (max - min).",
    "mean_nn_distance": "Mean nearest-neighbor distance over a sample set (proxy for spacing).",
    "density": "Points per unit volume inside the bounding box.",
    "horizontal_planes": "Detect approximately horizontal planes (e.g., floors, decks) via normals/RANSAC.",
    "normal_variation": "Local normal variability (edge indicator).",
    "cluster_labels": "DBSCAN cluster labels per point (noise = -1).",
    "curvature": "Local surface variation λ0/(λ0+λ1+λ2) from neighborhood PCA (per-point).",
    "planarity": "Planarity (λ1−λ0)/λ2 from neighborhood PCA (per-point).",
    "linearity": "Linearity (λ2−λ1)/λ2 from neighborhood PCA (per-point).",
    "sphericity": "Sphericity λ0/λ2 from neighborhood PCA (per-point).",
    "roughness": "Local roughness ~ sqrt(λ0) from neighborhood PCA (per-point).",
    "height_z": "Point height Z (per-point).",
    "normal_nz": "|n·z|, absolute Z component of estimated normal (per-point).",
    "normal_nx": "Estimated normal X-component (per-point).",
    "normal_ny": "Estimated normal Y-component (per-point).",
    "normal_nz": "Estimated normal Z-component (per-point).",
    "normal_axis": "Dominant normal axis label per point: 0=X, 1=Y, 2=Z.",
    "normal_var_nx": "Local variance of normal X-component (per-point).",
    "normal_var_ny": "Local variance of normal Y-component (per-point).",
    "normal_var_nz": "Local variance of normal Z-component (per-point).",
}


# ---------------------------------------------------------------------
# Helpers to attach widgets into the host DISPLAY tab
# ---------------------------------------------------------------------

def _add_to_display_panel(window, title: str, widget: QtWidgets.QWidget) -> None:
    """Attach *widget* into the DISPLAY tab with a titled box and sane sizes.

    Idempotent: will not add twice. It searches for a scroll area named
    `scrollDISPLAY` (like other plugins) and appends a group box.
    """
    # Idempotency guard
    if getattr(window, "_cloud_inspection_display_installed", False):
        return
    try:
        scroll = getattr(window, "scrollDISPLAY", None)
        if isinstance(scroll, QtWidgets.QScrollArea):
            container = scroll.widget()
            if container is None:
                container = QtWidgets.QWidget()
                scroll.setWidget(container)
            if container.layout() is None:
                container.setLayout(QtWidgets.QVBoxLayout())
            # Already present?
            for gb in container.findChildren(QtWidgets.QGroupBox):
                if gb.objectName() == "cloud_inspection.display_box":
                    return
            box = QtWidgets.QGroupBox(title)
            box.setObjectName("cloud_inspection.display_box")
            box.setMaximumWidth(300)
            lay = QtWidgets.QVBoxLayout(box)
            lay.setContentsMargins(8, 8, 8, 8)
            lay.addWidget(widget)
            widget.setMaximumWidth(300)
            widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
            box.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
            container.layout().addWidget(box)
            container.layout().addStretch(0)
            try:
                window._cloud_inspection_display_installed = True  # type: ignore[attr-defined]
            except Exception:
                pass
            return
    except Exception:
        pass

# ---------------------------------------------------------------------
# PluginManager metadata & entry point
# ---------------------------------------------------------------------
class _PluginMetaShim:
    name = "cloud_inspection"
    order = 20
    requires = tuple()
    version = "0.1.0"

PLUGIN = _PluginMetaShim()


def load_plugin(parent):
    return CloudInspectionPlugin(parent)

register = load_plugin


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def _try_import_metrics():
    """Try to import a Metrics module from various locations.

    Returns:
        module or None: The imported module, or None if not available.
    """
    mod = None
    for name in ("c2f4dt.plugins.cloud_inspection.metrics", "Metrics"):
        try:
            mod = __import__(name, fromlist=['*'])
            break
        except Exception:
            pass
    return mod


def _active_points(window) -> Optional[Tuple[Any, Optional[List[str]]]]:
    """Return (points_like, field_names) from active dataset or None.

    Tries to extract an (N,3) array-like from viewer3d._datasets[ds].pdata.
    """
    v = getattr(window, "viewer3d", None)
    if v is None:
        return None
    try:
        idx = int(window._current_dataset_index())
        recs = getattr(v, "_datasets", [])
        if not (0 <= idx < len(recs)):
            return None
        pdata = recs[idx].get("pdata")
        if pdata is None:
            return None
        # PyVista PolyData: has .points (numpy)
        pts = getattr(pdata, "points", None)
        if pts is None:
            # VTK: try GetPoints().GetData()
            try:
                vtk_pts = pdata.GetPoints().GetData()
                import numpy as np  # optional
                from vtk.util.numpy_support import vtk_to_numpy
                pts = vtk_to_numpy(vtk_pts)
            except Exception:
                return None
        # optional fields
        flds = []
        try:
            flds = list(getattr(pdata, "array_names", []) or [])
        except Exception:
            pass
        return pts, flds
    except Exception:
        return None


# ---------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------
class MetricsWorker(QtCore.QObject):
    """QThread worker that computes selected metrics with cancel & progress."""

    progress = QtCore.Signal(int, str)    # percent, message
    finished = QtCore.Signal(dict)        # results dict
    failed = QtCore.Signal(str)           # error message

    def __init__(self, points, selections: List[str], metrics_mod=None):
        super().__init__()
        self._pts = points
        self._sel = selections
        self._cancel = False
        self._metrics = metrics_mod
        self._cache: Dict[str, Any] = {}

    @QtCore.Slot()
    def run(self):
        try:
            res = {}
            total = max(1, len(self._sel))
            for i, key in enumerate(self._sel):
                if self._cancel:
                    self.progress.emit(int(100 * i / total), "Canceled")
                    break
                self.progress.emit(int(100 * i / total), f"Computing {key}…")
                # Dispatch
                val = self._compute_one(key)
                res[key] = val
                time.sleep(0.01)  # keep UI responsive
            self.progress.emit(100, "Done")
            self.finished.emit(res)
        except Exception as exc:
            self.failed.emit(str(exc))

    def cancel(self):
        self._cancel = True

    # ---------------- private ----------------
    def _compute_normals_full(self) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray"]:
        """Return (N,3) normals for the full cloud and caches.

        Uses PCA on a sample with KNN neighborhoods, then propagates normals to all points
        via nearest sampled neighbor. Returns (normals_full, normals_sample, sample_points, sample_index).
        """
        import numpy as np
        if "normals_full" in self._cache:
            return (self._cache["normals_full"],
                    self._cache["normals_sample"],
                    self._cache["normals_sample_pts"],
                    self._cache["normals_sample_idx"])  # type: ignore

        N = len(self._pts)
        sample_n = min(20000, N)
        idx = np.random.choice(N, size=sample_n, replace=False)
        S = self._pts[idx]
        try:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=min(16, sample_n-1)).fit(S)
            _, nbrs = nn.kneighbors(S)
            normals_s = np.empty((sample_n, 3), dtype=float)
            for j, row in enumerate(nbrs):
                P = S[row] - S[row].mean(axis=0)
                cov = (P.T @ P) / max(1, P.shape[0]-1)
                w, v = np.linalg.eigh(cov)
                n = v[:, 0]
                # Consistent orientation: make Z-component non-negative
                if n[2] < 0:
                    n = -n
                normals_s[j] = n
            # Propagate to full cloud by nearest sampled point
            nn2 = NearestNeighbors(n_neighbors=1).fit(S)
            _, ix = nn2.kneighbors(self._pts)
            normals_full = normals_s[ix[:, 0]]
        except Exception:
            # Fallback: all up normals
            normals_s = np.tile(np.array([0.0, 0.0, 1.0]), (sample_n, 1))
            normals_full = np.tile(np.array([0.0, 0.0, 1.0]), (N, 1))

        self._cache["normals_full"] = normals_full
        self._cache["normals_sample"] = normals_s
        self._cache["normals_sample_pts"] = S
        self._cache["normals_sample_idx"] = idx
        return normals_full, normals_s, S, idx

    def _compute_one(self, key: str):
        # Prefer external Metrics module if it exposes the function
        if self._metrics is not None and hasattr(self._metrics, key):
            fn = getattr(self._metrics, key)
            return fn(self._pts)

        # Built-in fallbacks (fast approximations)
        import numpy as np
        if key == "point_count":
            return int(len(self._pts))
        if key == "bounds":
            mins = np.min(self._pts, axis=0).tolist()
            maxs = np.max(self._pts, axis=0).tolist()
            return {"min": mins, "max": maxs}
        if key == "extent":
            b = self._compute_one("bounds")
            mins, maxs = b["min"], b["max"]
            return [float(maxs[i]-mins[i]) for i in range(3)]
        if key == "mean_nn_distance":
            # Very rough: sample K points and compute distance to nearest neighbors
            K = min(2000, len(self._pts))
            if K <= 1:
                return 0.0
            idx = np.random.choice(len(self._pts), size=K, replace=False)
            sample = self._pts[idx]
            # kd-tree if available
            try:
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(self._pts)
                dist, _ = nn.kneighbors(sample)
                d = dist[:,1]
            except Exception:
                # fallback: brute-force on small sample
                from scipy.spatial.distance import cdist
                D = cdist(sample, self._pts)
                D[D==0] = float("inf")
                d = D.min(axis=1)
            return float(np.mean(d))
        if key == "density":
            b = self._compute_one("bounds")
            mins, maxs = b["min"], b["max"]
            vol = max(1e-12, float(maxs[0]-mins[0]) * float(maxs[1]-mins[1]) * float(maxs[2]-mins[2]))
            return float(len(self._pts) / vol)
        if key == "horizontal_planes":
            # Lightweight horizontal plane detection.
            # Strategy: estimate normals if available via PCA on a random subset; select points
            # whose normals have |nz| >= 0.95, then cluster heights into modes (peaks).
            import numpy as np
            N = len(self._pts)
            if N < 50:
                return []
            sample_n = min(3000, N)
            idx = np.random.choice(N, size=sample_n, replace=False)
            S = self._pts[idx]
            # Estimate normals via local PCA (k=10) on the sample only
            try:
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=min(10, sample_n-1)).fit(S)
                dists, nbrs = nn.kneighbors(S)
                nz = []
                for row in nbrs:
                    P = S[row] - S[row].mean(axis=0)
                    cov = (P.T @ P) / max(1, P.shape[0]-1)
                    w, v = np.linalg.eigh(cov)
                    n = v[:, 0]  # smallest eigenvalue → normal
                    nz.append(abs(float(n[2])))
                nz = np.asarray(nz)
                mask = nz >= 0.95
                Z = S[mask][:, 2]
                if Z.size == 0:
                    return []
                # Cluster Z with 1D k-means (k up to 3)
                try:
                    from sklearn.cluster import KMeans
                    k = int(min(3, max(1, Z.size // 200)))
                    km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(Z.reshape(-1,1))
                    centers = sorted([float(c[0]) for c in km.cluster_centers_])
                except Exception:
                    centers = [float(np.median(Z))]
                return centers
            except Exception:
                # Fallback: use Z histogram modes
                hist, edges = np.histogram(self._pts[:,2], bins=20)
                tops = np.argsort(hist)[-3:]
                centers = [float(0.5*(edges[i]+edges[i+1])) for i in sorted(tops)]
                return centers

        if key == "normal_variation":
            # Local normal variation magnitude (edge indicator): stddev of normals around each point.
            import numpy as np
            N = len(self._pts)
            if N < 10:
                return np.array([], dtype=float)
            # Sample (cap for speed); compute on sample then scatter back as approx
            sample_n = min(20000, N)
            idx = np.random.choice(N, size=sample_n, replace=False)
            S = self._pts[idx]
            try:
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=min(16, sample_n-1)).fit(S)
                dists, nbrs = nn.kneighbors(S)
                var = np.empty(sample_n, dtype=float)
                for j, row in enumerate(nbrs):
                    P = S[row] - S[row].mean(axis=0)
                    cov = (P.T @ P) / max(1, P.shape[0]-1)
                    w, v = np.linalg.eigh(cov)
                    n = v[:, 0]  # smallest eigenvalue → normal
                    # approximate variation by ratio of small/large eigenvalues
                    # higher ratio → flatter; use 1 - flatness as edge-ness
                    flatness = float(w[0] / max(1e-12, w[-1]))
                    var[j] = 1.0 - flatness
                # Scatter to full size via nearest sample mapping
                try:
                    nn2 = NearestNeighbors(n_neighbors=1).fit(S)
                    _, ix = nn2.kneighbors(self._pts)
                    return var[ix[:,0]]
                except Exception:
                    return var
            except Exception:
                # Fallback: simple gradient along Z on a coarse grid proxy
                z = self._pts[:, 2]
                return (z - np.median(z))**2

        if key == "cluster_labels":
            # Quick clustering labels for coloring (DBSCAN on a sample, propagated to all points).
            import numpy as np
            N = len(self._pts)
            if N < 10:
                return np.full((N,), -1, dtype=int)
            sample_n = min(50000, N)
            idx = np.random.choice(N, size=sample_n, replace=False)
            S = self._pts[idx]
            # Heuristic eps from mean NN distance
            try:
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=2).fit(S)
                d, _ = nn.kneighbors(S)
                eps = float(np.median(d[:,1])) * 1.5
            except Exception:
                eps = 0.05
            try:
                from sklearn.cluster import DBSCAN
                lab_s = DBSCAN(eps=eps, min_samples=5).fit_predict(S)
            except Exception:
                lab_s = np.full((sample_n,), -1, dtype=int)
            # Propagate labels to full cloud by nearest sampled point
            try:
                from sklearn.neighbors import NearestNeighbors
                nn2 = NearestNeighbors(n_neighbors=1).fit(S)
                _, ix = nn2.kneighbors(self._pts)
                return lab_s[ix[:,0]]
            except Exception:
                return np.full((N,), -1, dtype=int)

        if key == "height_z":
            import numpy as np
            return np.asarray(self._pts[:, 2])

        if key in {"curvature", "planarity", "linearity", "sphericity", "roughness", "normal_nz"}:
            import numpy as np
            N = len(self._pts)
            if N < 10:
                return np.array([], dtype=float)
            sample_n = min(20000, N)
            idx = np.random.choice(N, size=sample_n, replace=False)
            S = self._pts[idx]
            try:
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=min(16, sample_n-1)).fit(S)
                _, nbrs = nn.kneighbors(S)
                vals = np.empty(sample_n, dtype=float)
                for j, row in enumerate(nbrs):
                    P = S[row] - S[row].mean(axis=0)
                    cov = (P.T @ P) / max(1, P.shape[0]-1)
                    w, v = np.linalg.eigh(cov)
                    # Sort ascending just in case
                    w = np.sort(w)
                    lam0, lam1, lam2 = float(w[0]), float(w[1]), float(w[2])
                    if key == "curvature":
                        denom = max(1e-12, lam0 + lam1 + lam2)
                        vals[j] = lam0 / denom
                    elif key == "planarity":
                        denom = max(1e-12, lam2)
                        vals[j] = (lam1 - lam0) / denom
                    elif key == "linearity":
                        denom = max(1e-12, lam2)
                        vals[j] = (lam2 - lam1) / denom
                    elif key == "sphericity":
                        denom = max(1e-12, lam2)
                        vals[j] = lam0 / denom
                    elif key == "roughness":
                        vals[j] = np.sqrt(max(0.0, lam0))
                    elif key == "normal_nz":
                        n = v[:, 0]
                        vals[j] = abs(float(n[2]))
                # Propagate to the full cloud by nearest sample mapping
                try:
                    nn2 = NearestNeighbors(n_neighbors=1).fit(S)
                    _, ix = nn2.kneighbors(self._pts)
                    return vals[ix[:, 0]]
                except Exception:
                    return vals
            except Exception:
                # Fallbacks: simple proxies if neighbors/PCA fail
                if key == "normal_nz":
                    return np.ones((N,), dtype=float)
                if key == "roughness":
                    z = self._pts[:, 2]
                    return (z - np.median(z)) ** 2
                return np.zeros((N,), dtype=float)

        if key in {"normal_nx", "normal_ny", "normal_nz", "normal_axis", "normal_var_nx", "normal_var_ny", "normal_var_nz"}:
            import numpy as np
            normals_full, normals_s, S, idx = self._compute_normals_full()
            if key == "normal_nx":
                return normals_full[:, 0]
            if key == "normal_ny":
                return normals_full[:, 1]
            if key == "normal_nz":
                return normals_full[:, 2]
            if key == "normal_axis":
                # 0=X, 1=Y, 2=Z by absolute component
                ax = np.argmax(np.abs(normals_full), axis=1)
                return ax.astype(int)
            # Variance components computed on the sampled neighborhoods, then propagated
            try:
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=min(16, len(S)-1)).fit(S)
                _, nbrs = nn.kneighbors(S)
                var_s = np.empty((len(S), 3), dtype=float)
                for j, row in enumerate(nbrs):
                    nbh = normals_s[row]
                    var_s[j] = nbh.var(axis=0)
                nn2 = NearestNeighbors(n_neighbors=1).fit(S)
                _, ix = nn2.kneighbors(self._pts)
                var_full = var_s[ix[:, 0]]
            except Exception:
                var_full = np.zeros_like(normals_full)
            if key == "normal_var_nx":
                return var_full[:, 0]
            if key == "normal_var_ny":
                return var_full[:, 1]
            if key == "normal_var_nz":
                return var_full[:, 2]

        # Unknown metric
        return None


# ---------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------
class MetricsDialog(QtWidgets.QDialog):
    def _list_point_datasets(self) -> List[tuple[int, str]]:
        """Return a list of (index, label) for point-cloud datasets in the viewer."""
        out: List[tuple[int, str]] = []
        try:
            host = self.parent(); v = getattr(host, "viewer3d", None)
            recs = getattr(v, "_datasets", []) if v is not None else []
            for i, rec in enumerate(recs):
                if rec.get("kind") == "points" and rec.get("pdata") is not None:
                    label = str(rec.get("name") or rec.get("filename") or f"Dataset {i}")
                    out.append((i, label))
        except Exception:
            pass
        return out

    def _points_from_dataset(self, ds_index: int) -> Optional[Any]:
        """Return (N,3) points array for the given dataset index, or None."""
        try:
            host = self.parent(); v = getattr(host, "viewer3d", None)
            recs = getattr(v, "_datasets", []) if v is not None else []
            if not (0 <= ds_index < len(recs)):
                return None
            pdata = recs[ds_index].get("pdata")
            if pdata is None:
                return None
            pts = getattr(pdata, "points", None)
            if pts is None:
                try:
                    vtk_pts = pdata.GetPoints().GetData()
                    import numpy as np
                    from vtk.util.numpy_support import vtk_to_numpy
                    pts = vtk_to_numpy(vtk_pts)
                except Exception:
                    return None
            return pts
        except Exception:
            return None

    def _store_results_on_dataset(self, results: Dict[str, Any], ds_index: Optional[int] = None) -> None:
        """Attach results to the active dataset record for later reuse/export.

        The data is stored under `viewer3d._datasets[idx]['inspection']` (scalars)
        and `['inspection_per_point']` (per-point arrays).
        Silent no-op if viewer/dataset is not available.
        """
        try:
            host = self.parent(); v = getattr(host, "viewer3d", None)
            if v is None:
                return
            idx = int(ds_index) if ds_index is not None else int(host._current_dataset_index())
            recs = getattr(v, "_datasets", [])
            if not (0 <= idx < len(recs)):
                return
            rec = recs[idx]
            per_point = {}
            scalars = {}
            for k, val in results.items():
                try:
                    import numpy as np
                    if isinstance(val, np.ndarray) and val.ndim == 1 and val.shape[0] == len(rec.get("pdata").points):
                        per_point[k] = val
                    else:
                        scalars[k] = val
                except Exception:
                    scalars[k] = val
            rec.setdefault("inspection", {}).update(scalars)
            rec.setdefault("inspection_per_point", {}).update(per_point)
            # Notify the Cloud Info panel (if mounted) to refresh its combo
            try:
                panel = getattr(host, "_cloud_inspection_panel", None)
                if panel is None:
                    # fallback: search by objectName
                    for w in host.findChildren(QtWidgets.QWidget):
                        if w.objectName() == "cloud_inspection.CloudInfoPanel":
                            panel = w; break
                if panel is not None and hasattr(panel, "refresh"):
                    panel.refresh()
            except Exception:
                pass
        except Exception:
            pass

    def _update_tree(self, results: Dict[str, Any], ds_index: Optional[int] = None) -> None:
        """Ensure an 'Inspection' branch exists under the active dataset and fill metrics.

        Structure:
            <dataset>
              - Inspection
                  - <metric>: <value>
        Values are rendered compactly via `_render_value_inline`.
        """
        try:
            host = self.parent()
            tree = getattr(host, "treeDatasets", None) or getattr(host, "tree", None)
            if tree is None or not hasattr(tree, "invisibleRootItem"):
                return

            # Locate dataset item by selection or by name
            root = tree.invisibleRootItem()
            ds_item = tree.currentItem() or None
            # Prefer target dataset name if ds_index provided
            cur_name = None
            try:
                if ds_index is not None:
                    v = getattr(host, "viewer3d", None)
                    recs = getattr(v, "_datasets", []) if v is not None else []
                    if 0 <= int(ds_index) < len(recs):
                        cur_name = recs[int(ds_index)].get("name") or recs[int(ds_index)].get("filename")
            except Exception:
                cur_name = None
            if ds_item is None:
                try:
                    v = getattr(host, "viewer3d", None)
                    idx = int(host._current_dataset_index())
                    recs = getattr(v, "_datasets", []) if v is not None else []
                    # Use cur_name if available, otherwise fallback to current dataset
                    if cur_name:
                        for i in range(root.childCount()):
                            it = root.child(i)
                            if it.text(0) == cur_name:
                                ds_item = it; break
                    elif 0 <= idx < len(recs):
                        fallback_name = recs[idx].get("name") or recs[idx].get("filename")
                        if fallback_name:
                            for i in range(root.childCount()):
                                it = root.child(i)
                                if it.text(0) == fallback_name:
                                    ds_item = it; break
                except Exception:
                    ds_item = None
            if ds_item is None:
                ds_item = root

            # Find or create 'Inspection' node
            def ensure_child(parent, label: str):
                for i in range(parent.childCount()):
                    it = parent.child(i)
                    if it.text(0) == label:
                        return it
                it = QtWidgets.QTreeWidgetItem([label])
                parent.addChild(it)
                return it

            insp = ensure_child(ds_item, "Inspection")

            # Fill/update metrics (idempotent update by label)
            def set_kv(parent, key: str, value: str):
                # find existing
                for i in range(parent.childCount()):
                    it = parent.child(i)
                    if it.text(0) == key:
                        it.setText(1, value)
                        return it
                it = QtWidgets.QTreeWidgetItem([key, value])
                parent.addChild(it)
                return it

            # Ensure two columns visible
            if tree.columnCount() < 2:
                tree.setColumnCount(2)
            header = tree.headerItem()
            if header is not None and header.columnCount() < 2:
                header.setText(0, "Name"); header.setText(1, "Value")

            # Ordered insert (baseline first)
            baseline = [
                "point_count", "bounds", "extent", "mean_nn_distance", "density", "horizontal_planes"
            ]
            keys = list(results.keys())
            ordered = [k for k in baseline if k in results] + sorted([k for k in keys if k not in baseline])
            for k in ordered:
                vstr = self._render_value_inline(results.get(k))
                set_kv(insp, k, vstr)

            tree.expandItem(ds_item)
            tree.expandItem(insp)
        except Exception:
            pass
    """Modal dialog to select and compute metrics for the active point cloud."""

    def __init__(self, parent, metrics_mod=None):
        super().__init__(parent)
        self.setWindowTitle("Cloud Inspection")
        self.setModal(True)
        self.resize(520, 520)
        self._metrics = metrics_mod
        self._target_ds_index: Optional[int] = None
        self._build_ui()

    def _build_ui(self):
        lay = QtWidgets.QVBoxLayout(self)
        # Dataset chooser (when multiple clouds are loaded)
        topForm = QtWidgets.QFormLayout()
        self.cboDataset = QtWidgets.QComboBox()
        # Populate with available point datasets
        items = self._list_point_datasets()
        for idx, label in items:
            self.cboDataset.addItem(label, idx)
        # Preselect current dataset if present
        try:
            cur = int(self.parent()._current_dataset_index())
            pos = next((k for k,(i,_) in enumerate(items) if i == cur), -1)
            if pos >= 0:
                self.cboDataset.setCurrentIndex(pos)
        except Exception:
            pass
        topForm.addRow("Dataset:", self.cboDataset)
        lay.addLayout(topForm)
        # Info
        self.lblInfo = QtWidgets.QLabel("Select the properties to compute on the active point cloud:")
        self.lblInfo.setWordWrap(True)
        lay.addWidget(self.lblInfo)

        # Metric list
        self.list = QtWidgets.QListWidget()
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)

        # Left/Right splitter: metrics list (left) + info panel (right)
        self.split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.split.addWidget(self.list)

        self.infoPanel = QtWidgets.QWidget()
        infoLay = QtWidgets.QVBoxLayout(self.infoPanel)
        infoLay.setContentsMargins(8, 8, 8, 8)
        self.lblTitle = QtWidgets.QLabel("—")
        f = self.lblTitle.font(); f.setBold(True); self.lblTitle.setFont(f)
        self.txtDesc = QtWidgets.QTextEdit(); self.txtDesc.setReadOnly(True)
        self.btnQuickTest = QtWidgets.QPushButton("Quick Test")
        infoLay.addWidget(self.lblTitle)
        infoLay.addWidget(self.txtDesc, 1)
        infoLay.addWidget(self.btnQuickTest)
        self.split.addWidget(self.infoPanel)
        self.split.setStretchFactor(0, 1)
        self.split.setStretchFactor(1, 1)
        lay.addWidget(self.split, 1)

        # Populate metrics (fallback set + any extra in Metrics module)
        base = [("point_count","Point count"),("bounds","Bounds"),("extent","Extent"),
                ("mean_nn_distance","Mean NN distance"),("density","Density"),("horizontal_planes","Horizontal planes")]
        # Per-point properties (computed on demand)
        base += [("normal_variation", "Normal variation (edge-ness)"),
                 ("cluster_labels", "Cluster labels (DBSCAN)")]
        base += [("curvature", "Curvature (PCA variation)"),
                 ("planarity", "Planarity (PCA)"),
                 ("linearity", "Linearity (PCA)"),
                 ("sphericity", "Sphericity (PCA)"),
                 ("roughness", "Roughness (PCA λ0)"),
                 ("height_z", "Height Z"),
                 ("normal_nz", "|n·z| (normal Z)")]
        base += [("normal_nx", "Normal nx"),
                 ("normal_ny", "Normal ny"),
                 ("normal_nz", "Normal nz"),
                 ("normal_axis", "Normal dominant axis (0/1/2)"),
                 ("normal_var_nx", "Var(n.x)"),
                 ("normal_var_ny", "Var(n.y)"),
                 ("normal_var_nz", "Var(n.z)")]
        known = {k for k,_ in base}
        if self._metrics is not None:
            for name in dir(self._metrics):
                if name.startswith("_") or name in known:
                    continue
                if callable(getattr(self._metrics, name)):
                    base.append((name, name.replace("_"," ").title()))
        for key, label in base:
            it = QtWidgets.QListWidgetItem(label)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked if key in ("point_count","bounds","mean_nn_distance") else QtCore.Qt.Unchecked)
            it.setData(QtCore.Qt.UserRole, key)
            it.setData(QtCore.Qt.UserRole+1, _METRIC_DESCRIPTIONS.get(key, ""))
            self.list.addItem(it)

        # Buttons + progress
        row = QtWidgets.QHBoxLayout()
        self.btnSelectAll = QtWidgets.QPushButton("Select All")
        self.btnSelectNone = QtWidgets.QPushButton("Select None")
        row.addWidget(self.btnSelectAll); row.addWidget(self.btnSelectNone); row.addStretch(1)
        lay.addLayout(row)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setObjectName("barPROGRESS")
        self.progress.setRange(0, 100)
        lay.addWidget(self.progress)

        # Add a label under the progress bar for the current metric
        self.lblProgressMsg = QtWidgets.QLabel("")
        self.lblProgressMsg.setObjectName("lblProgressMsg")
        self.lblProgressMsg.setStyleSheet("color: #e6e6e6;")  # match progress bar text color
        lay.addWidget(self.lblProgressMsg)
        # lay.addWidget(self.progress)

        row2 = QtWidgets.QHBoxLayout()
        self.btnCompute = QtWidgets.QPushButton("Compute")
        self.btnCancel = QtWidgets.QPushButton("Cancel")
        row2.addStretch(1); row2.addWidget(self.btnCompute); row2.addWidget(self.btnCancel)
        lay.addLayout(row2)

        # Wire
        self.btnSelectAll.clicked.connect(self._select_all)
        self.btnSelectNone.clicked.connect(self._select_none)
        self.btnCompute.clicked.connect(self._on_compute)
        self.btnCancel.clicked.connect(self.reject)

        # React to selection changes
        self.list.currentItemChanged.connect(self._on_metric_highlight)
        self.btnQuickTest.clicked.connect(self._on_quick_test)

        # --- Stylesheet: match global dark theme for progress widgets ---
        self.setStyleSheet(
            """
            QProgressBar#barPROGRESS {
                border: 1px solid #3c3f41;
                border-radius: 3px;
                text-align: center;
                background: #2b2b2b;
                color: #e6e6e6;
            }
            QProgressBar#barPROGRESS::chunk {
                background-color: #43a047; /* same green as main bar */
            }
            QLabel#lblProgressMsg {
                color: #e6e6e6;
            }
            """
        )

    # --- helpers ---
    def _selected_keys(self) -> List[str]:
        keys = []
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.checkState() == QtCore.Qt.Checked:
                keys.append(it.data(QtCore.Qt.UserRole))
        return keys

    def _select_all(self):
        for i in range(self.list.count()):
            self.list.item(i).setCheckState(QtCore.Qt.Checked)

    def _select_none(self):
        for i in range(self.list.count()):
            self.list.item(i).setCheckState(QtCore.Qt.Unchecked)

    def _on_metric_highlight(self, current: QtWidgets.QListWidgetItem, previous: Optional[QtWidgets.QListWidgetItem] = None):
        """Update the info panel when the highlighted metric changes."""
        if current is None:
            self.lblTitle.setText("—"); self.txtDesc.setPlainText(""); return
        key = current.data(QtCore.Qt.UserRole)
        title = current.text()
        desc = current.data(QtCore.Qt.UserRole+1) or _METRIC_DESCRIPTIONS.get(key, "")
        self.lblTitle.setText(title)
        self.txtDesc.setPlainText(desc)

    def _on_quick_test(self):
        """Run a very small preview computation for the highlighted metric.

        Intended to validate that a metric is callable and show a tiny sample
        result quickly (non-blocking). Heavy work is avoided here.
        """
        item = self.list.currentItem()
        if item is None:
            return
        key = item.data(QtCore.Qt.UserRole)
        # Use the dataset selected in the combo
        ds_index = int(self.cboDataset.currentData()) if self.cboDataset.currentData() is not None else None
        if ds_index is None:
            QtWidgets.QMessageBox.warning(self, "Cloud Inspection", "No dataset selected.")
            return
        pts = self._points_from_dataset(ds_index)
        if pts is None:
            QtWidgets.QMessageBox.warning(self, "Cloud Inspection", "Selected dataset has no points.")
            return
        # Small subsample for speed
        import numpy as np
        K = min(500, len(pts))
        if K <= 1:
            QtWidgets.QMessageBox.information(self, "Cloud Inspection", "Not enough points for a test.")
            return
        idx = np.random.choice(len(pts), size=K, replace=False)
        sub = pts[idx]
        # Prefer external Metrics entry if available
        mod = _try_import_metrics()
        val = None
        try:
            if mod is not None and hasattr(mod, key):
                val = getattr(mod, key)(sub)
            else:
                # Fallback: reuse worker's private dispatcher on a tiny array
                worker = MetricsWorker(sub, [key], metrics_mod=None)
                val = worker._compute_one(key)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Quick Test failed", str(exc))
            return
        # Show a concise preview
        from pprint import pformat
        QtWidgets.QMessageBox.information(self, "Quick Test", f"{key} →\n" + pformat(val, compact=True, width=80))

    # --- compute ---
    def _on_compute(self):
        # Resolve target dataset from the combo (stable even if user changes selection later)
        ds_index = int(self.cboDataset.currentData()) if self.cboDataset.currentData() is not None else None
        if ds_index is None:
            QtWidgets.QMessageBox.warning(self, "Cloud Inspection", "No dataset selected.")
            return
        pts = self._points_from_dataset(ds_index)
        if pts is None:
            QtWidgets.QMessageBox.warning(self, "Cloud Inspection", "Selected dataset has no points.")
            return
        self._target_ds_index = ds_index
        sel = self._selected_keys()
        if not sel:
            QtWidgets.QMessageBox.information(self, "Cloud Inspection", "Select at least one property to compute.")
            return

        # Start worker
        self.btnCompute.setEnabled(False)
        self.worker = MetricsWorker(pts, sel, metrics_mod=_try_import_metrics())
        self.thread = QtCore.QThread(self)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.start()

    def _on_progress(self, pct: int, msg: str):
        self.progress.setValue(int(pct))
        self.progress.setFormat((msg or "Working") + " (%p%)")
        self.lblProgressMsg.setText(msg)   # NEW

    def _on_done(self, results: Dict[str, Any]):
        # Persist on dataset and update the dataset tree
        self._store_results_on_dataset(results, ds_index=self._target_ds_index)
        self._update_tree(results, ds_index=self._target_ds_index)
        # Show ordered, readable report and also write to the message window
        self._show_results(results)
        self.accept()

    def _on_failed(self, err: str):
        QtWidgets.QMessageBox.critical(self, "Cloud Inspection – Error", err)
        self.btnCompute.setEnabled(True)

    def _render_value_inline(self, val) -> str:
        """Render a metric value compactly for list/reporting purposes.

        The goal is to keep one-liners readable: small lists/tuples inline,
        dicts as key=value fragments; large arrays summarized with shape.
        """
        try:
            import numpy as np  # optional for type checks
        except Exception:
            np = None  # type: ignore

        # Numpy arrays → summarize
        if np is not None and isinstance(val, np.ndarray):
            if val.size <= 6:
                return np.array2string(val, precision=4, separator=", ")
            return f"ndarray shape={val.shape} dtype={val.dtype}"

        # Simple containers
        if isinstance(val, (list, tuple)):
            if len(val) <= 6 and all(not isinstance(x, (list, tuple, dict)) for x in val):
                return "[" + ", ".join(self._render_value_inline(x) for x in val) + "]"
            return f"list(len={len(val)})"
        if isinstance(val, dict):
            items = list(val.items())
            if len(items) <= 6 and all(not isinstance(v, (list, tuple, dict)) for _, v in items):
                frag = ", ".join(f"{k}={self._render_value_inline(v)}" for k, v in items)
                return "{" + frag + "}"
            return f"dict(keys={len(items)})"
        if isinstance(val, float):
            return f"{val:.6g}"
        return str(val)

    def _show_results(self, results: Dict[str, Any]) -> None:
        """Present results in an ordered, human-friendly list and log to message window.

        Ordering: baseline metrics first in a fixed order, then any extra metrics
        from custom modules sorted alphabetically.
        """
        # Determine order
        baseline = [
            "point_count", "bounds", "extent", "mean_nn_distance", "density", "horizontal_planes"
        ]
        keys = list(results.keys())
        ordered = [k for k in baseline if k in results] + sorted([k for k in keys if k not in baseline])

        # Build a dialog with a QTreeWidget
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Cloud Inspection – Report")
        dlg.resize(640, 520)
        v = QtWidgets.QVBoxLayout(dlg)
        tree = QtWidgets.QTreeWidget()
        tree.setColumnCount(2)
        tree.setHeaderLabels(["Metric", "Value"])
        tree.setUniformRowHeights(True)
        v.addWidget(tree, 1)

        # Fill tree and collect plain-text report
        lines = []
        per_point_keys = {"normal_variation", "cluster_labels", "curvature", "planarity", "linearity", "sphericity", "roughness", "height_z", "normal_nz", "normal_nx", "normal_ny", "normal_axis", "normal_var_nx", "normal_var_ny", "normal_var_nz"}
        for k in ordered:
            val = results.get(k)
            vstr = self._render_value_inline(val)
            if k in per_point_keys:
                # Do not flood the table; show compact hint row
                item = QtWidgets.QTreeWidgetItem([k, vstr])
                tree.addTopLevelItem(item)
            else:
                item = QtWidgets.QTreeWidgetItem([k, vstr])
                tree.addTopLevelItem(item)
            tree.resizeColumnToContents(0)
            lines.append(f"- {k}: {vstr}")

        # Buttons
        row = QtWidgets.QHBoxLayout()
        btnCopy = QtWidgets.QPushButton("Copy to clipboard")
        btnClose = QtWidgets.QPushButton("Close")
        row.addStretch(1); row.addWidget(btnCopy); row.addWidget(btnClose)
        v.addLayout(row)

        def _copy():
            QtWidgets.QApplication.clipboard().setText("\n".join(lines))
        btnCopy.clicked.connect(_copy)
        btnClose.clicked.connect(dlg.accept)

        # Log to message window if present
        try:
            host = self.parent()
            if hasattr(host, "txtMessages"):
                host.txtMessages.appendPlainText("Cloud Inspection – Report\n" + "\n".join(lines))
        except Exception:
            pass

        dlg.exec()
# ---------------------------------------------------------------------
# Cloud Info panel for coloring by per-point properties
# ---------------------------------------------------------------------
class CloudInfoPanel(QtWidgets.QWidget):
    """Small panel to color the active point cloud by inspection properties."""
    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.window = window
        self._build_ui()

    def _build_ui(self):
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(8, 8, 8, 8)
        self.cboProp = QtWidgets.QComboBox()
        self.cboCmap = QtWidgets.QComboBox(); self.cboCmap.addItems(["viridis","plasma","magma","inferno","jet"])  # best-effort
        self.btnRefresh = QtWidgets.QPushButton("Refresh")
        self.btnApply = QtWidgets.QPushButton("Apply")
        self.btnClear = QtWidgets.QPushButton("Clear")
        f = QtWidgets.QFormLayout(); f.addRow("Property:", self.cboProp); f.addRow("Colormap:", self.cboCmap)
        v.addLayout(f)
        row = QtWidgets.QHBoxLayout(); row.addWidget(self.btnRefresh); row.addStretch(1); row.addWidget(self.btnApply); row.addWidget(self.btnClear)
        v.addLayout(row)
        v.addStretch(1)
        self.btnRefresh.clicked.connect(self.refresh)
        self.btnApply.clicked.connect(self.apply_coloring)
        self.btnClear.clicked.connect(self.clear_coloring)
        self.refresh()

    def _active_rec(self):
        try:
            v = getattr(self.window, "viewer3d", None)
            idx = int(self.window._current_dataset_index())
            recs = getattr(v, "_datasets", []) if v is not None else []
            if 0 <= idx < len(recs):
                return recs[idx]
        except Exception:
            pass
        return None

    def refresh(self):
        current = self.cboProp.currentText()
        self.cboProp.clear()
        rec = self._active_rec()
        props = []
        if rec is not None:
            pp = rec.get("inspection_per_point", {}) or {}
            props = sorted(pp.keys())
        self.cboProp.addItems(props)
        # try to restore previous selection
        if current and current in props:
            idx = self.cboProp.findText(current)
            if idx >= 0:
                self.cboProp.setCurrentIndex(idx)

    def apply_coloring(self):
        rec = self._active_rec()
        if rec is None:
            return
        pdata = rec.get("pdata")
        if pdata is None:
            return
        key = self.cboProp.currentText()
        if not key:
            return
        arr = (rec.get("inspection_per_point") or {}).get(key)
        if arr is None:
            return

        # attach as point-data array on the budgeted view
        try:
            pdata.point_data["cloud_info__" + key] = arr
        except Exception:
            try:
                pdata["cloud_info__" + key] = arr
            except Exception:
                return

        # apply via viewer
        try:
            ds = int(self.window._current_dataset_index())
        except Exception:
            ds = 0
        cmap = self.cboCmap.currentText() or "viridis"
        try:
            self.window.viewer3d.color_points_by_array(ds, "cloud_info__" + key, cmap=cmap, show_scalar_bar=True)
        except Exception:
            pass


    def clear_coloring(self):
        rec = self._active_rec()
        if rec is None:
            return
        try:
            ds = int(self.window._current_dataset_index())
        except Exception:
            ds = 0
        try:
            self.window.viewer3d.reset_point_coloring(ds)
        except Exception:
            pass



# ---------------------------------------------------------------------
# Plugin main class
# ---------------------------------------------------------------------
class CloudInspectionPlugin(QtCore.QObject):
    """Cloud Inspection plugin: opens a dialog to compute point cloud metrics."""

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self._install_toolbar_button()
        self._install_shortcut()
        try:
            panel = CloudInfoPanel(window)
            panel.setObjectName("cloud_inspection.CloudInfoPanel")
            setattr(window, "_cloud_inspection_panel", panel)
            _add_to_display_panel(window, "Cloud Info", panel)
        except Exception:
            pass
        self._log("INFO", "[cloud_inspection] ready")


    # --------------------- integration points -------------------------
    def _install_toolbar_button(self):
        # Idempotent add to right toolbar if present
        tb = None
        for attr in ("barVERTICAL_right", "right_toolbar", "toolBarRight"):
            tb = getattr(self.window, attr, None)
            if isinstance(tb, QtWidgets.QToolBar):
                break
        if tb is None:
            return
        # Check existing
        name = "cloud_inspection.actOpenDialog"
        for a in tb.actions():
            if a.objectName() == name:
                return
        act = QtGui.QAction(self._qicon("32x32_cloud_inspection.png"), "Cloud Inspection", self.window)
        act.setObjectName(name)
        act.triggered.connect(self._open_dialog)
        tb.addAction(act)

    def _install_shortcut(self):
        try:
            QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+I"), self.window, activated=self._open_dialog)
        except Exception:
            pass

    def _qicon(self, name: str) -> QtGui.QIcon:
        try:
            from c2f4dt.utils.icons import qicon
            ic = qicon(name)
            if isinstance(ic, QtGui.QIcon):
                return ic
        except Exception:
            pass
        return QtGui.QIcon()

    def _open_dialog(self):
        dlg = MetricsDialog(self.window, metrics_mod=_try_import_metrics())
        dlg.exec()

    # -------------------------- actions for PluginManager --------------
    def run(self, *args, **ctx) -> bool:
        self._open_dialog()
        return True

    def exec(self, *args, **ctx) -> bool:
        return self.run(*args, **ctx)

    def execute(self, *args, **ctx) -> bool:
        return self.run(*args, **ctx)

    def show(self, *args, **ctx) -> bool:
        return self.run(*args, **ctx)

    def __call__(self, *args, **kwargs) -> bool:  # alias
        return self.run(**kwargs)

    # ------------------------------ utils ------------------------------
    def _log(self, level: str, text: str) -> None:
        try:
            if hasattr(self.window, "txtMessages"):
                self.window.txtMessages.appendPlainText(text)
        except Exception:
            pass
        print(f"[{level}] {text}")
