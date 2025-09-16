"""
Example plugin: adds a panel with a button inside the scrollDISPLAY (DisplayPanel).

HOW INTEGRATION WITH YOUR PluginManager WORKS
---------------------------------------------
- Your PluginManager:
    1) Scans the "plugins/" folder looking for subfolders.
    2) For each folder, reads "plugin.yaml" (if present) and determines:
         - name, version, order, requires, entry_point = "module:attribute".
    3) Imports the package "plugins.<name>" (thanks to the plugins folder in the file system).
    4) Resolves the entry_point: imports the submodule (here "plugin") and retrieves the attribute (here "register").
    5) Calls the factory "register(window)" which must return the plugin instance.

- This file provides:
    - a `DisplayButtonPlugin` class (QObject) that receives the MainWindow and modifies the UI.
    - a `register(window)` function used as the entry-point by the manager.

WHERE THE BUTTON ENDS UP
-------------------------
- Your UI has:
        MainWindow.scrollDISPLAY  -> QScrollArea
            └─ widget()            -> DisplayPanel (custom QWidget with an internal layout)
- We DO NOT touch the QScrollArea directly; we go to its content (DisplayPanel)
    and add our `QGroupBox` with a button.

SIMPLE BEST PRACTICES FOR YOUR PLUGINS
--------------------------------------
- Do not block the UI thread: if you perform long tasks, use QThread/worker (as you normally would).
- Avoid assuming overly specific layout details; use `layout().addWidget(...)` at the end.
- Provide an `actions()` (or `get_actions()`) method to make commands appear in the Plugins menu.
- If you add elements to the scene or panel, consider a `teardown()` method to clean up (optional).
"""

from __future__ import annotations
from typing import Optional, Dict, Any

from PySide6 import QtCore, QtGui, QtWidgets


class DisplayButtonPlugin(QtCore.QObject):
    """
    Minimal plugin that:
      1) Inserts a panel into the DisplayPanel with a "Hello from Plugin" button.
      2) Exposes an action in the &Plugins menu ("Say Hello") that displays a message.

    The constructor receives the MainWindow (window) from the PluginManager.
    """

    def __init__(self, window: QtWidgets.QMainWindow):
        super().__init__(window)
        self.window = window  # reference to the host MainWindow
        self._panel_box: Optional[QtWidgets.QGroupBox] = None  # our group box in the panel

        # Try to add the panel immediately. If the DisplayPanel is not ready yet,
        # you can defer with a singleShot(0, ...) or connect to a "ready" signal.
        self._inject_panel_box()

        # (optional) Create a QAction and add it to the Tools or Plugins menu.
        # Alternatively, implement actions()/get_actions() (see below).
        try:
            mb = window.menuBar()
            m_plugins = None
            for a in mb.actions():
                if a.text().replace("&", "") == "Plugins":
                    m_plugins = a.menu()
                    break
            if m_plugins is None:
                m_plugins = mb.addMenu("&Plugins")
            act = QtGui.QAction("Say Hello (example_display_button)", self)
            act.triggered.connect(self.say_hello)
            m_plugins.addAction(act)
        except Exception:
            # if it fails (menu missing), it's not critical: the manager will use actions()
            pass

    # -----------------------
    # UI injection helpers
    # -----------------------
    def _display_panel(self) -> Optional[QtWidgets.QWidget]:
        """
        Returns the DisplayPanel widget (content of the QScrollArea scrollDISPLAY).
        """
        try:
            panel = getattr(self.window, "displayPanel", None)
            if isinstance(panel, QtWidgets.QWidget):
                return panel
        except Exception:
            pass
        # fallback: try to retrieve from the scroll area
        try:
            scroll = getattr(self.window, "scrollDISPLAY", None)
            if isinstance(scroll, QtWidgets.QScrollArea):
                return scroll.widget()
        except Exception:
            pass
        return None

    def _inject_panel_box(self) -> None:
        """
        Creates a QGroupBox with a button and adds it to the end of the DisplayPanel layout.
        """
        panel = self._display_panel()
        if panel is None:
            # DisplayPanel not ready? retry after the current event
            QtCore.QTimer.singleShot(0, self._inject_panel_box)
            return

        lay = panel.layout()
        if lay is None:
            # if the panel has no layout (unlikely), create a vertical one
            lay = QtWidgets.QVBoxLayout(panel)
            panel.setLayout(lay)

        # Build our group box
        box = QtWidgets.QGroupBox("Example plugin panel")
        v = QtWidgets.QVBoxLayout(box)

        # Descriptive label
        lbl = QtWidgets.QLabel(
            "This panel was added by a plugin.\n"
            "You can use it as a starting point to create your own controls."
        )
        lbl.setWordWrap(True)
        v.addWidget(lbl)

        # Example button
        btn = QtWidgets.QPushButton("Hello from Plugin")
        btn.setObjectName("btnHelloFromPlugin")
        btn.clicked.connect(self._on_click)
        v.addWidget(btn)

        # (Optional) additional UI: checkbox, slider, etc.
        chk = QtWidgets.QCheckBox("Also show a message in the console")
        chk.setObjectName("chkPluginConsoleEcho")
        chk.setChecked(True)
        v.addWidget(chk)

        # Store references if needed
        self._panel_box = box
        self._btn = btn
        self._chk_echo = chk

        # Add the group box to the end of the panel
        lay.addWidget(box)

    # -----------------------
    # Slots / Actions
    # -----------------------
    @QtCore.Slot()
    def _on_click(self) -> None:
        """
        Slot for the button in the panel: shows a message box and optionally writes to the console.
        """
        QtWidgets.QMessageBox.information(self.window, "Example Plugin", "Hello 👋 from example_display_button!")
        # Echo to the console (if present and allowed by the checkbox)
        try:
            if getattr(self, "_chk_echo", None) and self._chk_echo.isChecked():
                console = getattr(self.window, "console", None)
                if console is not None and hasattr(console, "appendPlainText"):
                    console.appendPlainText("# example_display_button: hello clicked")
        except Exception:
            pass

    @QtCore.Slot()
    def say_hello(self) -> None:
        """
        Menu action: identical to the button action, but callable from the Plugins menu.
        """
        self._on_click()

    # -----------------------
    # API that your MainWindow already supports
    # -----------------------
    def actions(self):
        """
        Returns a list of structured actions that the MainWindow attaches to the Plugins menu.
        See MainWindow._rebuild_plugins_menu/_invoke_plugin_action.
        """
        return [
            {
                "label": "Say Hello",
                "tooltip": "Displays a courtesy message",
                "slot": self.say_hello,  # direct callable
            },
            {
                "label": "Focus Hello Button",
                "tooltip": "Brings focus to the button in the DisplayPanel",
                "slot": self.focus_button,
            },
        ]

    def focus_button(self):
        """Brings focus to the button created by the plugin (if it exists)."""
        try:
            if getattr(self, "_btn", None) is not None:
                self._btn.setFocus()
                # visual feedback
                self._btn.animateClick(150)
        except Exception:
            pass

    # (Optional) Method that your systems can call as a "default run"
    def run(self, *_, **__):
        """
        Default entry-point if you launch the plugin from the combo without actions:
        focuses the button so it's immediately visible where it was added.
        """
        self.focus_button()

    # (Optional) if you want to clean up UI/actors when the plugin is destroyed
    def teardown(self):
        """
        Removes the box from the DisplayPanel (best-effort).
        You can call this manually if you foresee plugin unload/refresh.
        """
        try:
            if self._panel_box is not None:
                parent = self._panel_box.parent()
                self._panel_box.setParent(None)
                self._panel_box.deleteLater()
                # if needed, clean up other references
                self._panel_box = None
        except Exception:
            pass
