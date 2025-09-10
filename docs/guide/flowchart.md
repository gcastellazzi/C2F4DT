## Flowchart of main components

The following Mermaid flowchart illustrates the main components and data flow within the Cloud2FEMi application, covering GUI elements, data input/output, processing steps, data models, grid and FEM generation, visualization overlays, and optional tools.

```mermaid
flowchart TD
  %% Grouped by responsibility
  subgraph "GUI / UX"
    A1["Main Window (PySide6) — cloud2fem.ui.main_window"]
    A2["3D Viewer (PyVista/VTK) — cloud2fem.ui.viewers.viewer"]
    A3["2D Slice Window — cloud2fem.ui.viewers.pyvista_backend"]
  end

  subgraph "I/O"
    B1["Point Cloud Import/Export — cloud2fem.io.pcl"]
    B2["Boundary Conditions Export — cloud2fem.io.bcs"]
  end

  subgraph "Data Model"
    C1["TwinState / SliceSet / Slice — cloud2fem.model.types"]
    C2["Materials & Boundary Conditions — cloud2fem.model.types"]
  end

  subgraph "Processing"
    D1["Create Slices — cloud2fem.ops.slices"]
    D2["Centroid & Feature Extraction — cloud2fem.ops.geometry"]
    D3["Normals Estimation — cloud2fem.ops.normals"]
  end

  subgraph "Grid & FEM"
    E1["Grid Model — cloud2fem.grid.grid_model"]
    E2["FEM Generation / Export — (placeholder)"]
  end

  subgraph "Visualization"
    F1["2D Overlays (Points / Labels) — cloud2fem.viz.overlays"]
    F2["3D Overlays (Points / Labels) — cloud2fem.viz.overlays_3d"]
  end

  subgraph "Optional Tools"
    G1["Additional Utilities — cloud2fem.tools.optional"]
  end
  ```


```mermaid
  A1 -->|Open / Load| B1
  B1 -->|NDArray + Channels| C1
  A1 -->|Slice Parameters| D1
  D1 -->|SliceSet| C1
  D1 -->|Indices per Slice| D2
  D2 -->|Centroids / Polylines / Polygons| C1
  A1 -->|Compute Normals| D3
  D3 -->|Normals Channel| C1
  C1 -->|Slice Polygons| E1
  E1 -->|Slice-Aligned Grid| E2
  C2 -->|Attach BCs| E2
  C2 -->|Export BCs| B2
  C1 -->|Derived Layers| A2
  F1 -->|Points / Legend| A2
  F2 -->|3D Points / Legend| A2
  A1 -->|Open 2D View| A3
  C1 -->|Slice Layers| A3
  G1 -->|Optional Utilities| A1
```