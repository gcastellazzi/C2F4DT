# C2F4DT — Plugin/Extensions Developer Guide

This guide shows how to build **drop-in extensions** for C2F4DT without modifying `main_window.py`.  
Extensions live in the `./extensions/` folder and are loaded automatically at startup.  
The **Cloud2FEM** plugin is included as an example, but the framework is designed for any plugin or extension.

## TL;DR
- Create a folder under `extensions/your_plugin/`
- Add a small `extension.json` manifest
- Provide a Python module with a factory `create_extension()` that returns an object implementing:
  - `name: str`, `version: str`
  - `activate(ctx: HostContext) -> None`
  - `deactivate() -> None`

---

## 1) Folder layout

```
extensions/
  your_plugin/
    extension.json
    __init__.py
    your_plugin.py
```

**Example `extension.json`:**
```json
{
  "name": "My Plugin",
  "package": "your_plugin",
  "entry": "create_extension",
  "version": "1.0.0",
  "requires_api": "1.0"
}
```

- `package`: python module (filename without `.py`)
- `entry`: factory function returning the extension instance
- `requires_api`: host API version your plugin expects

---

## 2) Host API (what you get)

Extensions receive a `HostContext` at `activate()`:

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class HostContext:
    window: Any                 # MainWindow (QMainWindow)
    registry: dict[str, Any]    # Shared services registry
    add_tab: callable           # add_tab(title, widget, target="main")
    log: callable               # log(level, message)
```

*(In the current codebase, `HostContext` is defined in `c2f4dt.plugins.manager` and re-used by plugins.)*

- Use `ctx.add_tab(widget_title, widget, target="main")` to add whole tabs to `tabMain`.
- Access existing content panels with:
  - `scrollDISPLAY_CONTENT`, `scrollSLICING_CONTENT`, `scrollFEM_CONTENT`, `scrollRESULTS_CONTENT`
- Store persistent UI prefs in `ctx.window.view_opts` (a dict), e.g., `view_opts["slice_point_size"]`.

**Stability**: the host exposes `APP_PLUGIN_API_VERSION` to gate incompatible plugins.

---

## 3) Minimal plugin template

```python
# extensions/your_plugin/your_plugin.py
from __future__ import annotations
from typing import Optional
from PySide6 import QtWidgets
from c2f4dt.plugins.manager import HostContext

class MyPlugin:
    name = "My Plugin"
    version = "1.0.0"

    def __init__(self):
        self._ctx: Optional[HostContext] = None
        self._panel: Optional[QtWidgets.QGroupBox] = None

    def activate(self, ctx: HostContext) -> None:
        self._ctx = ctx
        ctx.log("INFO", "Activating My Plugin")

        # 1) Create UI
        container = ctx.window.findChild(QtWidgets.QWidget, "scrollDISPLAY_CONTENT")
        if container is None:
            ctx.log("WARN", "scrollDISPLAY_CONTENT not found")
            return

        self._panel = QtWidgets.QGroupBox("My Plugin Panel", container)
        form = QtWidgets.QFormLayout(self._panel)

        btn = QtWidgets.QPushButton("Do something", self._panel)
        btn.clicked.connect(self._on_click)
        form.addRow(btn)

        # 2) Mount into the panel container
        lay = container.layout() or QtWidgets.QVBoxLayout(container)
        lay.addWidget(self._panel)

    def deactivate(self) -> None:
        if self._ctx and self._panel:
            self._panel.setParent(None)
            self._panel.deleteLater()
        self._panel = None
        self._ctx = None

    def _on_click(self):
        self._ctx.log("INFO", "Button clicked!")

def create_extension():
    return MyPlugin()
```

> Prefer embedding controls inside existing `scrollXXX_CONTENT` containers for a cohesive UX.  
> Use `ctx.add_tab(...)` if your feature needs a dedicated full tab.

---

## 4) UI targets & object names

Common insertion points (object names):

- Display: `scrollDISPLAY_CONTENT`
- Slices: `scrollSLICING_CONTENT`
- FEM: `scrollFEM_CONTENT`
- Results: `scrollRESULTS_CONTENT`

Adding toolbar actions is allowed (use `ctx.window.menuBar()` or find toolbars by name), but keep toolbar minimal and consistent.

---

## 5) Reading & setting view options

Use `ctx.window.view_opts` to store simple preferences:

```python
v = getattr(ctx.window, "view_opts", {}) or {}
v["myplugin_threshold"] = 0.25
ctx.window.view_opts = v
```

Renderers (3D viewer, etc.) can read those values to affect visualization.  
Example keys used by the Slices visualization extension:

- `slice_point_size` / `slice_point_color` (all slices)
- `slice_current_point_size` / `slice_current_point_color` (current slice)

---

## 6) Integrating with existing logic (optional)

If you need to sync with host behavior, you can look up helpful widgets or actions:

- `cbSliceIndex` (`QComboBox`): current slice index
- `toggle_current_slice_3D_view` / `toggle_all_slices_3D_view` (`QAction`)

Prefer **host-agnostic** logic; call internal helpers only when necessary and available.

---

## 7) Logging & messages

- Use `ctx.log("INFO", "message")` for plugin logs.
- For user-facing messages, use the host’s message area:
  ```python
  txt = ctx.window.findChild(QtWidgets.QPlainTextEdit, "txtMessages")
  if txt:
      txt.appendPlainText("[MyPlugin] Something happened…")
  ```

---

## 8) Versioning & compatibility

- Set `requires_api` in `extension.json`.
- When the host bumps `APP_PLUGIN_API_VERSION`, incompatible plugins are skipped.
- Keep your plugin’s `version` in sync with your changes.

---

## 9) Testing checklist

- [ ] No crashes if target containers are missing (fail gracefully).
- [ ] No global state required; clean up in `deactivate()`.
- [ ] Works with/without a loaded point cloud.
- [ ] Honors dark theme where possible (avoid hard-coded colors).
- [ ] No blocking calls on the UI thread for long operations.

---

## 10) Packaging & distribution

- Ship your plugin folder (`your_plugin/`) as a zip; users unzip into `./extensions/`.
- Avoid heavy dependencies; if needed, handle absence gracefully.
- Keep names unique to avoid clashes.

---

## 11) Advanced tips

- **Hot reload (dev):** add a host command to call `plugin_manager.unload_all(); plugin_manager.load_all()`.
- **Shared services:** use `ctx.registry` to publish/reuse services between plugins.
- **Targeted tabs:** the host’s `add_tab` supports `target="main"` and can be extended for additional tab stacks.
