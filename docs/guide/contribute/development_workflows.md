
## Development Workflow

1. Create a feature branch
2. Implement code with clear docstrings
3. Update pages in `docs/guide/`
4. `mkdocs serve` for local preview
5. Open a PR with screenshots/gifs when helpful

## Docstring style (Google)
```python
def compute_normals(points, k=16):
    """Compute normals for a point cloud.

    Args:
        points (ndarray): (N, 3) array of XYZ.
        k (int): Neighborhood size.

    Returns:
        ndarray: (N, 3) array of normals.
    """
    ...
```

## Processing Workflow

The processing workflow in Cloud2FEMi consists of the following 10 steps:

1. **Point Cloud Import**  
   Load the raw point cloud data (e.g., from LiDAR or photogrammetry) into the system.

2. **Slice Creation**  
   Divide the point cloud into slices, typically along a chosen axis, to simplify further processing.

3. **Centroid Generation**  
   Compute centroids for each slice, representing the central position of points within a slice.

4. **Polyline Generation**  
   Connect centroids or selected points within each slice to form polylines that outline key features.

5. **Polygon Creation**  
   Convert polylines into closed polygons to define cross-sectional shapes.

6. **Grid Generation**  
   Generate a computational grid (mesh) within each polygon, suitable for FEM analysis.

7. **Grid Extrusion/Adaptation**  
   Extrude or adapt the 2D grids into 3D elements, following the geometry of the original point cloud.

8. **FEM Generation**  
   Assemble the full finite element model, assigning nodes and elements based on the extruded grid.

9. **Boundary Conditions and Export**  
   Define boundary conditions and export the FEM model in a format ready for simulation.

10. **Visualization**  
    Visualize the processed data and FEM model to verify geometry and mesh quality before simulation.
