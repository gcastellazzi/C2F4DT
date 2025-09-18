"""
Cloud2FEM plugin for C2F4DT.

This package exposes a compact right-toolbar panel that orchestrates
Cloud2FEMi pipelines (denoise, clustering, verticals detection, Hex8 grid,
and export) integrated with the host viewer and progress system.

Author: C2F4DT team
License: same as C2F4DT
"""
from .plugin import Cloud2FEMPlugin

__all__ = ["Cloud2FEMPlugin"]