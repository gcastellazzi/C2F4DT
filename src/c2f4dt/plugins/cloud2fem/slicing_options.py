# -*- coding: utf-8 -*-
"""Default slicing & contour options for Cloud2FEM.

This module centralizes the default configuration for the slicing pipeline
and related visualization. The function :func:`default_slice_options` returns
an in-memory dictionary so callers can copy and mutate safely.

All values are deliberately explicit to make their intent clear. Keep the
structure stable because both GUI and non-GUI code read these keys.
"""
from __future__ import annotations

from typing import Dict, Any


def default_slice_options() -> Dict[str, Any]:
    """Return the default options for slicing and contour extraction.

    Returns:
        dict: Nested dictionary with default parameters for:
            - slices: axis, thickness, spacing mode, etc.
            - centroids: clustering and tolerance parameters.
            - polylines: splitting and simplification.
            - polygons: polygonization and validation.
            - viz: visualization defaults (3D/2D overlays for slices).
    """
    return {
        "_version": 2,
        "slices": {
            "axis": "Z",                   # X, Y, Z, CUSTOM:...
            "thickness": 0.01,
            "spacing_mode": "fixed_count",  # fixed_step | fixed_count | custom
            "fixed_step": 0.02,
            "fixed_count": 20,
            "align_to_grid": True,
            "inplane_filter": True,
        },
        "centroids": {
            "min_wall_thick": 0.18,
            "min_pts_slice": 10,           # tolsl
            "min_pts_cluster": 2,          # tolpt
            "seed_tol": 0.10,              # tol
            "check_fraction": 0.10,        # checkpts
            "tol_incr": 1.35,
        },
        "polylines": {
            "split_jump": 0.18,            # min_wall_thick
            "simplify_tol": 0.02,
            "check_angles": False,
            "min_angle": 80,
            "check_fitting": False,
            "fit_tol": 0.05,
            "method": "baseline",          # registry key
        },
        "polygons": {
            "enable": True,                # forced False if Shapely is missing
            "simplify_tol": 0.035,
            "invalid_limit_factor": 2.5,
        },
        # Visualization options unified here (used by main_window + plugins)
        "viz": {
            # ALL SLICES (3D overlay)
            "all_slices_points_size": 6.0,
            "all_slices_points_color": (0, 120, 255),          # rgb tuple
            "all_slices_points_spheres": True,
            # CURRENT SLICE (3D overlay)
            "current_slice_points_size": 8.0,
            "current_slice_points_color": (255, 20, 147),      # deep pink
            "current_slice_points_spheres": True,
            # Optional label color for current slice text in 3D
            "current_slice_label_color": (255, 20, 147),
            # 2D viewer defaults (extend as needed)
            "centroids_color": (220, 30, 20),
            "centroids_size": 10.0,
        },
    }