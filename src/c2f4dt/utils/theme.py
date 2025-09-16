from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

def apply_user_theme(window: QtWidgets.QWidget, mode: str = "auto") -> None:
    """Apply light/dark/auto theme.

    Args:
        window: Root widget to theme.
        mode: 'light', 'dark', or 'auto' (follows system).
    """
    palette = QtGui.QPalette()

    def set_dark(p: QtGui.QPalette) -> None:
        p.setColor(QtGui.QPalette.Window, QtGui.QColor(45, 45, 45))
        p.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
        p.setColor(QtGui.QPalette.Base, QtGui.QColor(30, 30, 30))
        p.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(45, 45, 45))
        p.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
        p.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
        p.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
        p.setColor(QtGui.QPalette.Button, QtGui.QColor(45, 45, 45))
        p.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
        p.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
        p.setColor(QtGui.QPalette.Highlight, QtGui.QColor(38, 79, 120))
        p.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.white)

    if mode == "auto":
        # Heuristic: use dark if system uses a dark base color
        # (Qt doesn't expose a universal dark-mode flag cross-platform)
        base_lightness = QtWidgets.QApplication.palette().window().color().lightness()
        if base_lightness < 128:
            set_dark(palette)
        else:
            palette = QtWidgets.QApplication.palette()
    elif mode == "dark":
        set_dark(palette)
    else:
        palette = QtWidgets.QApplication.palette()

    QtWidgets.QApplication.setPalette(palette)
