# Code Architecture

## Package layout (indicative)
```
c2f4dt/
  ui/             # Qt .ui files and PySide6 widgets for the main application interface
  viewer/         # Core 3D viewer classes and VTK/PyVista rendering backend
  plugins/        # Built-in plugins (Cloud2FEM and others)
  utils/          # Shared helper functions and utilities
  app/            # Application entry points and high-level orchestration
  data/           # Assets (icons, styles, default resources)
examples/
  run_app.py      # Minimal launcher
extensions/       # Optional external extensions (distributed separately)
tests/            # Unit and integration tests
```

## Plugins
- **Cloud2FEM** is included as a plugin that transforms point clouds into finite element meshes.
- Additional plugins can extend the viewer with new features (e.g., clustering, import/export, custom visualization).
- External **extensions** can be distributed in separate repositories and integrated at runtime.

## Conventions
- **Docstrings**: Follow **Google** or **NumPy** style for clarity.
- **Naming**: Use clear names for Qt signals/slots, viewer methods, and plugin entry points.
- **Tests**: Add lightweight tests where feasible; include both unit and integration tests.
- **Style**: Follow **PEP8** guidelines for Python code formatting.
- **Typing**: Use type hints consistently for function signatures.
- **Logging**: Employ Python’s `logging` module for runtime information and debugging.
- **Separation of concerns**: Maintain modularity by separating UI, core viewer, plugins, and data handling.
- **Testing strategy**: Place tests alongside modules or in a dedicated `tests/` directory; ensure plugin APIs are tested as well.



