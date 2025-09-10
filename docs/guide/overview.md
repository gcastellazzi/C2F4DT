# Overview

This guide describes project goals, scope, and current status.

- **Repository**: `Cloud2FEMi`
- **Python package**: `cloud2fem`
- **Viewer**: PyVista/VTK + PySide6
- **Scope**: point-cloud import, slicing, inspection, GUI, and Cloud→FEM pipeline

## Project Goals and Significance

Cloud2FEMi aims to provide a comprehensive and user-friendly software framework to facilitate the processing of point-cloud data for finite element method (FEM) analysis. The project focuses on enabling efficient import, slicing, inspection, and conversion of complex 3D point clouds into FEM models, streamlining the workflow from raw data acquisition to structural analysis.

Key goals include:

- Developing a robust pipeline for processing and slicing large-scale point clouds.
- Integrating advanced visualization tools for inspection and validation of data.
- Providing a graphical user interface (GUI) that simplifies interaction with the pipeline.
- Extending existing methodologies into a modern, Python-based framework that supports extensibility and ease of use.

The significance of Cloud2FEMi lies in its ability to bridge the gap between raw 3D scanning data and engineering simulations, facilitating more accurate and efficient structural assessments and designs.

## Scope

- Importing and managing diverse point-cloud datasets.
- Implementing slicing algorithms to extract meaningful cross-sections.
- Enabling detailed inspection and visualization through PyVista and VTK.
- Integrating a GUI based on PySide6 for enhanced user experience.
- Automating the pipeline from point-cloud data to FEM-ready models.

## Relevant Published Works

Cloud2FEMi builds upon a series of foundational methodologies developed and published by Castellazzi and colleagues:

- [Cloud2FEM: A finite element mesh generator based on point clouds of existing/historical structures](https://www.sciencedirect.com/science/article/pii/S235271102200067X). Castellazzi, G., et al. (2022). *SoftwareX*.
- [An innovative numerical modeling strategy for the structural analysis of historical monumental buildings](https://www.sciencedirect.com/science/article/pii/S0141029616312627). Castellazzi, G., et al. (2017). *Engineering Structures*.
- [From Laser Scanning to Finite Element Analysis of Complex Buildings by Using a Semi-Automatic Procedure](https://www.mdpi.com/1424-8220/15/8/18360). Castellazzi, G., et al. (2015). *Sensors*.

These works establish the theoretical and practical background for the techniques implemented in Cloud2FEMi.

## Note

Cloud2FEMi extends these established methodologies by integrating them into a modern Python-based software framework, featuring a graphical user interface built with PySide6 and visualization capabilities powered by VTK/PyVista. This integration enhances usability, flexibility, and accessibility for researchers and practitioners in the field. 

!!! note "Project status"
    Under development - not a stable release.
