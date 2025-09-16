# C2F4DT — Developer Guide
<img src="assets/logo-cloud2fem.png" alt="C2F4DT Logo" width="300" height="300" align="right"> Welcome to the **C2F4DT** developer documentation. This site collects guidelines, code architecture, theory, and an **auto-generated API reference**.  

**C2F4DT** is a modular software package designed as a 3D viewer and interaction environment built on VTK. Its primary goal is to provide a framework where plugins can be developed to visualize and interact with point clouds and finite element models. Within this architecture, **Cloud2FEM** is included as one of the plugins, enabling users to transform point cloud data into finite element meshes.

The concept behind C2F4DT is to serve as a host for digital twins of finite element models that can be initialized or continuously updated using point cloud data or IoT (Internet of Things) streams. This first release focuses on the core viewer capabilities, establishing a foundation for visualization, interaction, and plugin integration.

Future versions will extend the package with additional plugins and extensions, enhancing functionality and enabling more advanced workflows for digital twin management and structural analysis.

## Features
<img src="assets/image3.png" alt="C2F4DT Logo" width="300" height="300" align="right">
- Import and preprocess 3D point clouds from multiple formats
- Slice point clouds into meaningful cross-sections
- Generate centroids and polygons to represent structural elements
- Create grids and finite element meshes for structural analysis
- Define boundary conditions (BCs) and load cases
- Visualize point clouds, meshes, and simulation results within the tool

## Publications

Cloud2FEMi and its underlying methodologies have been described and validated in several key publications:

- [Cloud2FEM: A finite element mesh generator based on point clouds of existing/historical structures](https://www.sciencedirect.com/science/article/pii/S235271102200067X). Castellazzi, G., et al. (2022). *SoftwareX*.
- [An innovative numerical modeling strategy for the structural analysis of historical monumental buildings](https://www.sciencedirect.com/science/article/pii/S0141029616312627). Castellazzi, G., et al. (2017). *Engineering Structures*.
- [From Laser Scanning to Finite Element Analysis of Complex Buildings by Using a Semi-Automatic Procedure](https://www.mdpi.com/1424-8220/15/8/18360). Castellazzi, G., et al. (2015). *Sensors*.

## Getting Started

To begin exploring Cloud2FEMi, we recommend reviewing the following sections:
<img src="assets/image4.png" alt="C2F4DT Logo" width="300" height="300" align="right">
- [Installation](guide/installation.md)
- [Overview](guide/overview.md)
- [Theory & Equations](guide/theory.md)
- [Architecture](guide/architecture.md)
- [Flowchart](guide/flowchart.md)

## Quick start
```bash
# Local preview
pip install mkdocs-material "mkdocstrings[python]" pymdown-extensions
mkdocs serve
```

## Goals
- Documentation that is easy to evolve during development (Markdown + MathJax)
- Always up-to-date Python API via **mkdocstrings**
- Automatic deploy to **GitHub Pages**
