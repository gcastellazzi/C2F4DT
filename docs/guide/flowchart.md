## Flowchart of main components

The following Mermaid flowchart illustrates the main components and data flow within the **C2F4DT** application.  
C2F4DT acts as a viewer and plugin host. The diagram shows GUI elements, data input/output, processing steps, data models, grid and FEM generation, visualization overlays, and optional tools.  
**Cloud2FEM** is represented as a built-in plugin extending these capabilities.


```mermaid
flowchart TD
  %% Grouped by responsibility
  subgraph "GUI / UX"
    A1["Main Window (PySide6) — c2f4dt.main_window"]
    A2["3D Viewer (PyVista/VTK) — c2f4dt.ui.viewer3d"]
    A3["2D Slice Window — c2f4dt.ui.display_panel"]
  end

  subgraph "I/O"
    B1["Point Cloud Import/Export — c2f4dt.utils.io.importers"]
    B2["Boundary Conditions Export — c2f4dt.plugins.cloud2fem (BCs)"]
  end

  subgraph "Data Model"
    C1["TwinState / SliceSet / Slice — c2f4dt.plugins.cloud2fem.model"]
    C2["Materials & Boundary Conditions — c2f4dt.plugins.cloud2fem.model"]
  end

  subgraph "Processing"
    D1["Create Slices — c2f4dt.plugins.cloud2fem.slices"]
    D2["Centroid & Feature Extraction — c2f4dt.plugins.cloud2fem.geometry"]
    D3["Normals Estimation — c2f4dt.plugins.cloud2fem.normals"]
  end

  subgraph "Grid & FEM"
    E1["Grid Model — c2f4dt.plugins.cloud2fem.grid_model"]
    E2["FEM Generation / Export — c2f4dt.plugins.cloud2fem (FEM)"]
  end

  subgraph "Visualization"
    F1["2D Overlays (Points / Labels) — c2f4dt.ui.display_panel"]
    F2["3D Overlays (Points / Labels) — c2f4dt.ui.viewer3d"]
  end

  subgraph "Optional Tools"
    G1["Additional Utilities / Plugins — c2f4dt.plugins.*"]
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
```mermaid