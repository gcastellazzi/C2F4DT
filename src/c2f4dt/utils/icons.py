from __future__ import annotations

import os
from PySide6 import QtGui

def icon_path(name: str) -> str:
    """Return absolute path to an icon in the project `icons/` directory.

    Args:
        name: Filename of the icon (e.g., '32x32_document-new.png').

    Returns:
        Absolute path string. If not found, returns the original name,
        allowing Qt to fallback gracefully (useful in dev).
    """
    # icons folder sits at project root beside c2f4dt/
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "c2f4dt/assets/icons"))
    path = os.path.join(base, name)
    return path if os.path.isfile(path) else name

def qicon(name: str) -> QtGui.QIcon:
    """Convenience to create a QIcon from the project icons folder."""
    return QtGui.QIcon(icon_path(name))
