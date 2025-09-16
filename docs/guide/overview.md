# Overview

This guide describes project goals, scope, and current status.

- **Repository**: `C2F4DT`
- **Python package**: `c2f4dt`
- **Viewer**: PyVista/VTK + PySide6
- **Scope**: point-cloud import, slicing, inspection, GUI, and plugin-based FEM workflows

## Project Goals and Significance

C2F4DT is designed as a core viewer and host framework that enables the integration and execution of specialized plugins for finite element method (FEM) analysis and related workflows. It provides a flexible and user-friendly environment to import, slice, inspect, and visualize complex 3D point-cloud data, while supporting extensible plugins such as Cloud2FEM for FEM model generation.

Key goals include:

- Developing a robust core framework for processing and slicing large-scale point clouds.
- Integrating advanced visualization tools for inspection and validation of data.
- Providing a graphical user interface (GUI) that simplifies interaction with the viewer and plugins.
- Supporting extensibility through a plugin architecture that includes FEM workflows like Cloud2FEM.

The significance of C2F4DT lies in its ability to serve as a versatile platform bridging raw 3D scanning data and engineering simulations, facilitating more accurate and efficient structural assessments and designs through modular plugins.

## Scope

- Importing and managing diverse point-cloud datasets.
- Implementing slicing algorithms to extract meaningful cross-sections.
- Enabling detailed inspection and visualization through PyVista and VTK.
- Integrating a GUI based on PySide6 for enhanced user experience.
- Providing a plugin infrastructure to support FEM-ready model generation and other workflows.

## Relevant Published Works

Cloud2FEMi builds upon a series of foundational methodologies developed and published by Castellazzi and colleagues:

- [Cloud2FEM: A finite element mesh generator based on point clouds of existing/historical structures](https://www.sciencedirect.com/science/article/pii/S235271102200067X). Castellazzi, G., et al. (2022). *SoftwareX*.
- [An innovative numerical modeling strategy for the structural analysis of historical monumental buildings](https://www.sciencedirect.com/science/article/pii/S0141029616312627). Castellazzi, G., et al. (2017). *Engineering Structures*.
- [From Laser Scanning to Finite Element Analysis of Complex Buildings by Using a Semi-Automatic Procedure](https://www.mdpi.com/1424-8220/15/8/18360). Castellazzi, G., et al. (2015). *Sensors*.

These works establish the theoretical and practical background for the techniques implemented in Cloud2FEMi.

## Note

C2F4DT provides the core viewer and plugin infrastructure, with Cloud2FEM integrated as one of the key plugins enabling FEM workflows. This modular approach enhances usability, flexibility, and accessibility for researchers and practitioners in the field.

!!! note "Project status"
    Under development - not a stable release.
