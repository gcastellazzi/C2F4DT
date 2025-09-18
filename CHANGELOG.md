# Changelog

## [Unreleased]

### Added
- **Cloud Inspection Plugin**
  - Computes global & per-point metrics for point clouds (point count, bounds, NN distance, curvature, normals, clusters, etc.).
  - Interactive dialog with metric selection, progress tracking, and quick tests.
  - Results stored under the dataset tree in an **Inspection** branch.
  - New **Cloud Info panel** in the Display tab to colorize point clouds by computed properties.
  - Toolbar button with `32x32_cloud_inspection.png`.

- **Cloud2FEM Plugin**
  - Slicing and FEM preparation from point clouds.
  - Slice options: axis, thickness, spacing (fixed step, fixed count, custom).
  - Computation of centroids, polylines, polygons, and FEM-ready grids.

- **Transformations Plugin**
  - Apply translation, rotation, and scaling to datasets.
  - Supports incremental and batch operations.
  - Reset option to restore original dataset position.

- **Units Plugin**
  - Engineering unit conversions (SI ↔ Imperial).
  - Quick access via toolbar.

- **Import VTK Plugin**
  - Import `.vtk` and `.vtu` datasets directly.
  - Automatic dataset tree integration.
  - Supports multi-file selection.

### Documentation
- New plugin guide pages under `docs/plugins/`:
  - `units.md`
  - `cloud_inspection.md`
  - `cloud2fem.md`
  - `transformations.md`
  - `import_vtk.md`

Each page includes:
- Description
- Toolbar icon
- Features
- Usage workflow
- Options/parameters

---

## [0.1.0] – Initial release
- Base viewer (PyVista + PySide6 integration).
- Dataset tree with point cloud and mesh support.
- Core toolbar actions (import, export, views).