from __future__ import annotations

import sys
from PySide6 import QtCore, QtWidgets, QtGui

from .main_window import MainWindow

import os
# Optional: force software OpenGL if env var is set (last-resort stability)
if os.environ.get("C2F4DT_SOFTWARE_OPENGL", "0") == "1":
    try:
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseSoftwareOpenGL, True)
    except Exception:
        pass

def main() -> None:
    """Application entry-point.

    Initializes the Qt application, applies the platform theme, and shows the main window.
    """
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("C2F4DT")
    app.setOrganizationName("C2F4DT")
    app.setApplicationVersion("1.0")

    # Imposta l'icona dell'applicazione (Cmd+Tab su macOS, Alt+Tab su Windows)
    icon_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "icons", "C2F4DT.icns")
    if os.path.isfile(icon_path):
        app.setWindowIcon(QtGui.QIcon(icon_path))

    # High-DPI friendly defaults
    if hasattr(QtCore.Qt, "AA_UseHighDpiPixmaps"):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())