from __future__ import annotations
from PySide6 import QtCore, QtGui, QtWidgets

class DisplayPanel(QtWidgets.QWidget):
    """Property panel for point cloud visualization.

    Signals:
        sigPointSizeChanged: Emitted when point size changes.
        sigPointBudgetChanged: Emitted when point budget (%) changes.
        sigColorModeChanged: Emitted when color mode changes.
        sigSolidColorChanged: Emitted when solid color changes.
        sigColormapChanged: Emitted when colormap changes.
    """

    # Signals
    sigPointSizeChanged = QtCore.Signal(int)
    sigPointBudgetChanged = QtCore.Signal(int)
    sigColorModeChanged = QtCore.Signal(str)
    sigSolidColorChanged = QtCore.Signal(QtGui.QColor)
    sigColormapChanged = QtCore.Signal(str)

    def __init__(self, parent=None) -> None:
        """Initialize the panel with controls."""
        super().__init__(parent)
        form = QtWidgets.QFormLayout(self)
        form.setLabelAlignment(QtCore.Qt.AlignRight)
        form.setFormAlignment(QtCore.Qt.AlignTop)

        # Point size
        self.sldSize = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sldSize.setRange(1, 20)
        self.sldSize.setValue(3)
        self.spinSize = QtWidgets.QSpinBox()
        self.spinSize.setRange(1, 20)
        self.spinSize.setValue(3)
        self.sldSize.valueChanged.connect(self.spinSize.setValue)
        self.spinSize.valueChanged.connect(self.sldSize.setValue)
        self.sldSize.valueChanged.connect(self.sigPointSizeChanged)
        sizeRow = QtWidgets.QHBoxLayout()
        sizeRow.addWidget(self.sldSize, 1)
        sizeRow.addWidget(self.spinSize)
        form.addRow("Point size", _wrap(sizeRow))

        # Point budget (%)
        self.sldBudget = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sldBudget.setRange(1, 100)
        self.sldBudget.setValue(100)
        self.spinBudget = QtWidgets.QSpinBox()
        self.spinBudget.setRange(1, 100)
        self.spinBudget.setValue(100)
        self.sldBudget.valueChanged.connect(self.spinBudget.setValue)
        self.spinBudget.valueChanged.connect(self.sldBudget.setValue)
        self.sldBudget.valueChanged.connect(self.sigPointBudgetChanged)
        budRow = QtWidgets.QHBoxLayout()
        budRow.addWidget(self.sldBudget, 1)
        budRow.addWidget(self.spinBudget)
        form.addRow("% points shown", _wrap(budRow))

        # Color mode
        self.cmbColorMode = QtWidgets.QComboBox()
        self.cmbColorMode.addItems(["Solid", "Normal RGB", "Normal Colormap"])
        self.cmbColorMode.currentTextChanged.connect(self._on_mode_changed)
        form.addRow("Color mode", self.cmbColorMode)

        # Solid color
        self.btnColor = QtWidgets.QPushButton("Choose…")
        self.btnColor.clicked.connect(self._pick_color)
        form.addRow("Solid color", self.btnColor)

        # Colormap
        self.cmbCmap = QtWidgets.QComboBox()
        self.cmbCmap.addItems(["viridis", "magma", "plasma", "cividis"])
        self.cmbCmap.currentTextChanged.connect(self.sigColormapChanged)
        form.addRow("Colormap", self.cmbCmap)

        self._update_visibility()

    def _on_mode_changed(self, text: str) -> None:
        """Handle color mode change."""
        self.sigColorModeChanged.emit(text)
        self._update_visibility()

    def _update_visibility(self) -> None:
        """Enable/disable controls based on color mode."""
        solid = self.cmbColorMode.currentText() == "Solid"
        cmap = self.cmbColorMode.currentText() == "Normal Colormap"
        self.btnColor.setEnabled(solid)
        self.cmbCmap.setEnabled(cmap)

    def _pick_color(self) -> None:
        """Open color dialog and emit chosen color."""
        col = QtWidgets.QColorDialog.getColor(parent=self)
        if col.isValid():
            pal = self.btnColor.palette()
            pal.setColor(QtGui.QPalette.Button, col)
            self.btnColor.setPalette(pal)
            self.btnColor.setAutoFillBackground(True)
            self.sigSolidColorChanged.emit(col)


def _wrap(layout: QtWidgets.QLayout) -> QtWidgets.QWidget:
    """Wrap a layout into a QWidget (utility for forms)."""
    w = QtWidgets.QWidget()
    w.setLayout(layout)
    return w