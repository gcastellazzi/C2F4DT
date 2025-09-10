# Code Architecture

## Package layout (indicative)
```
cloud2fem/
  ui/             # Qt .ui files and PySide6 widgets for user interface
  viz/            # Viewer interface and PyVista/VTK backend for visualization
  io/             # Input/output operations, import/export of point clouds and models
  model/          # Finite Element Model construction and representation
  ops/            # Core processing operations such as slicing, clustering, and feature extraction
  grid/           # Grid and mesh generation for FEM modeling
  tools/          # Utility scripts and command-line tools for various tasks
  utils/          # Helper functions and common utilities across the project
  native/         # Native extensions and bindings for performance-critical code
examples/
  run_app.py      # Minimal launcher
```

## Conventions
- Docstrings: **Google** or **NumPy** style
- Clear names for Qt signals/slots and viewer methods
- Add lightweight tests where feasible
- Code style: Follow **PEP8** guidelines for Python code formatting
- Typing: Use type hints consistently for function signatures
- Logging: Employ Python’s logging module for runtime information and debugging
- Separation of concerns: Maintain modularity by separating UI, logic, and data handling
- Testing strategy: Write unit and integration tests; place tests alongside modules or in a dedicated `tests/` directory



