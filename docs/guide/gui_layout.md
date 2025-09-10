# Cloud2FEMi — GUI Layout (Qt Object Names)

This document provides a structured overview of the main GUI elements in **Cloud2FEMi**, focusing on Qt `objectName`s to assist developers when wiring widgets, signals, and extensions.

---

# Cloud2FEMi — GUI Layout (Tree View)

```text
MainWindow (QMainWindow)
│
├── Menu bar (menubar)
│
├── Toolbars
│   ├── barTOPCOMMAND              # Top horizontal toolbar
│   ├── barVERTICALCOMMAND_left    # Left vertical toolbar
│   ├── barVERTICALCOMMAND_right   # Right vertical toolbar
│   └── barVIEWER3D                # Small toolbar inside 3D viewer
│
├── Central area
│   ├── tabMain (QTabWidget)            # Main tab container
│   │
│   ├── tabDisplay (QWidget)            # Tab: Display
│   │   └── scrollDISPLAY_CONTENT       # Scrollable content area for Display
│   │
│   ├── tabSLICING (QWidget)            # Tab: Slices
│   │   └── scrollSLICING_CONTENT       # Scrollable content area for Slices
│   │
│   ├── tabFEM (QWidget)                # Tab: FEM / Mesh
│   │   └── scrollFEM_CONTENT           # Scrollable content area for FEM
│   │
│   ├── tabRESULTS (QWidget)            # Tab: Results
│   │   └── scrollRESULTS_CONTENT       # Scrollable content area for Results
│   │
│   ├── tabINSPECTOR (QWidget)          # Tab: Inspector / Tree
│   │   └── treeMCT (QTreeWidget)       # Main MCT tree
│   │
│   └── tabCONSOLE (QTabWidget)         # Console tab (can host multiple pages)
│       └── (page "Console") → layout with custom console widget
│
├── 3D Viewer Panel
│   └── VIEWER3D (QWidget, QVBoxLayout)
│       ├── barVIEWER3D (QToolBar)      # Viewer toolbar
│       └── (placeholder → replaced by QtInteractor for PyVista)
│
├── Secondary Trees / Data
│   └── treeMCTS (QTreeWidget)          # Optional: second dataset tree
│
├── Status bar area
│   ├── barPROGRESS (QProgressBar)      # Progress indicator
│   └── buttonCANCEL (QPushButton)      # Cancel action
│
└── Messages
    └── txtMessages (QPlainTextEdit)    # Log / messages output
```

---

## 1. Main Window Structure

- **QMainWindow**
  - **Top toolbar** → `barTOPCOMMAND`
  - **Left vertical toolbar** → `barVERTICALCOMMAND_left`
  - **Right vertical toolbar** → `barVERTICALCOMMAND_right`
  - **Viewer 3D area** → `VIEWER3D`
    - Contains toolbar → `barVIEWER3D`
  - **Message log** → `txtMessages`
  - **Progress bar** → `barPROGRESS`
  - **Cancel button** → `buttonCANCEL`
  - **Tabs (main)** → `tabMain`

---

## 2. Tab Pages and Containers

- **Display tab** → `tabDisplay`
  - Scroll container: `scrollDISPLAY_CONTENT`
- **Slices tab** → `tabSLICING`
  - Scroll container: `scrollSLICING_CONTENT`
- **FEM tab** → `tabFEM`
  - Scroll container: `scrollFEM_CONTENT`
- **Results tab** → `tabRESULTS`
  - Scroll container: `scrollRESULTS_CONTENT`
- **Inspector tab** → `tabINSPECTOR`
  - Tree widget: `treeMCT`
- **Console tab** → `tabCONSOLE`
  - Replaces placeholder with console widget
- **MCTS tab** → `tabMCTS`
  - Tree widget: `treeMCTS`

---

## 3. Actions and Menus

- File:
  - `actionNew`, `actionOpen`, `actionSave`, `actionSaveAs`
- Import cloud:
  - `actionImportCloud`
- Grid tools:
  - `actionCreateGrid`, `actionToggleGrid`
- Normals:
  - `actionToggleNormals`
- Tab navigation:
  - `actionOpen_Display_Tab`
  - `actionOpen_Slices_Tab`
  - `actionOpen_FEM_Tab`
  - `actionOpen_Inspector_Tab`
  - `actionOpen_2dView`

---

## 4. Slice Visualization Actions

- `toggle_current_slice_3D_view`
- `toggle_all_slices_3D_view`
- `toggle_centroids_view`
- `toggle_polylines_view`
- `toggle_polygons_view`
- `toggle_mesh_view`

---

# Abbreviated Map (Cheat Sheet)

| Area              | Object name(s) |
|-------------------|----------------|
| Top toolbar       | `barTOPCOMMAND` |
| Left toolbar      | `barVERTICALCOMMAND_left` |
| Right toolbar     | `barVERTICALCOMMAND_right` |
| Viewer container  | `VIEWER3D` |
| Viewer toolbar    | `barVIEWER3D` |
| Message editor    | `txtMessages` |
| Progress bar      | `barPROGRESS` |
| Cancel button     | `buttonCANCEL` |
| Tabs root         | `tabMain` |
| Display container | `scrollDISPLAY_CONTENT` |
| Slicing container | `scrollSLICING_CONTENT` |
| FEM container     | `scrollFEM_CONTENT` |
| Results container | `scrollRESULTS_CONTENT` |
| Inspector tree    | `treeMCT` |
| MCTS tree         | `treeMCTS` |
| Console tab       | `tabCONSOLE` |

---

### Quick reference for slice tools:

- Current slice overlay: `toggle_current_slice_3D_view`
- All slices overlay: `toggle_all_slices_3D_view`
- Centroids: `toggle_centroids_view`
- Polylines: `toggle_polylines_view`
- Polygons: `toggle_polygons_view`
- Mesh: `toggle_mesh_view`
