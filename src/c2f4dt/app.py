from __future__ import annotations

import sys
import os
from PySide6 import QtCore, QtWidgets, QtGui

from .main_window import MainWindow

# Optional: force software OpenGL if env var is set (last-resort stability)
if os.environ.get("C2F4DT_SOFTWARE_OPENGL", "0") == "1":
    try:
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseSoftwareOpenGL, True)
    except Exception:
        pass


def main() -> None:
    """Application entry-point.

    Initializes the Qt application, shows the splash as early as possible,
    enforces a minimum splash duration, then shows the main window.
    """
    app = QtWidgets.QApplication(sys.argv)

    # --- Splash screen (show ASAP) --------------------------------------
    splash = None
    splash_timer = None
    # Minimum splash duration (ms). Override with env C2F4DT_SPLASH_MIN_MS
    min_ms_default = 3200
    try:
        MIN_SPLASH_MS = int(os.environ.get("C2F4DT_SPLASH_MIN_MS", str(min_ms_default)))
    except Exception:
        MIN_SPLASH_MS = min_ms_default

    try:
        icons_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "c2f4dt/assets/icons")
        splash_path = os.path.join(icons_dir, "C2F4DT_512x512.png")
        if os.path.exists(splash_path):
            pixmap = QtGui.QPixmap(splash_path)
            if not pixmap.isNull():
                splash = QtWidgets.QSplashScreen(pixmap)
                splash.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, True)
                splash.show()
                splash.showMessage(
                    "Loading C2F4DT...",
                    QtCore.Qt.AlignBottom | QtCore.Qt.AlignHCenter,
                    QtCore.Qt.black,
                )
                app.processEvents()  # ensure splash paints immediately

                # Start elapsed timer to enforce minimum on-screen time
                splash_timer = QtCore.QElapsedTimer()
                splash_timer.start()
    except Exception:
        splash = None
        splash_timer = None

    # -------------------------------------------------------------------
    app.setApplicationName("C2F4DT")
    app.setOrganizationName("C2F4DT")
    app.setApplicationVersion("1.0")

    # Set app icon (Cmd+Tab on macOS, Alt+Tab on Windows)
    try:
        icon_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "c2f4dt/assets/icons", "C2F4DT.icns")
        if os.path.isfile(icon_path):
            app.setWindowIcon(QtGui.QIcon(icon_path))
    except Exception:
        pass

    # High-DPI friendly defaults
    if hasattr(QtCore.Qt, "AA_UseHighDpiPixmaps"):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    # Build and show the main window
    win = MainWindow()
    win.show()

    # --- Finish splash with minimum duration ----------------------------
    if splash is not None:
        try:
            elapsed = splash_timer.elapsed() if splash_timer is not None else MIN_SPLASH_MS
            remaining = max(0, MIN_SPLASH_MS - int(elapsed))
            if remaining == 0:
                splash.finish(win)
            else:
                QtCore.QTimer.singleShot(remaining, lambda: splash.finish(win))
        except Exception:
            # Best-effort fallback
            try:
                splash.finish(win)
            except Exception:
                pass

    # Enter event loop
    sys.exit(app.exec())