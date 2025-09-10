from __future__ import annotations
from PySide6 import QtCore, QtGui, QtWidgets

class DisplayPanel(QtWidgets.QWidget):
    """Property panel for dataset visualization.

    Signals:
        sigPointSizeChanged: Emitted when point size changes.
        sigPointBudgetChanged: Emitted when point budget (%) changes.
        sigColorModeChanged: Emitted when color mode changes.
        sigSolidColorChanged: Emitted when solid color changes.
        sigColormapChanged: Emitted when colormap changes.
        sigMeshRepresentationChanged: Emitted when mesh representation changes.
        sigMeshOpacityChanged: Emitted when mesh opacity changes.
    """

    # Signals
    sigPointSizeChanged = QtCore.Signal(int)
    sigPointBudgetChanged = QtCore.Signal(int)
    sigColorModeChanged = QtCore.Signal(str)
    sigSolidColorChanged = QtCore.Signal(QtGui.QColor)
    sigColormapChanged = QtCore.Signal(str)
    sigMeshRepresentationChanged = QtCore.Signal(str)
    sigMeshOpacityChanged = QtCore.Signal(int)

    def __init__(self, parent=None) -> None:
        """Initialize the panel with controls."""
        super().__init__(parent)
        form = QtWidgets.QFormLayout(self)
        form.setLabelAlignment(QtCore.Qt.AlignRight)
        form.setFormAlignment(QtCore.Qt.AlignTop)
        self.form = form

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
        self.rowSize = _wrap(sizeRow)
        form.addRow("Point size", self.rowSize)

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
        self.rowBudget = _wrap(budRow)
        form.addRow("% points shown", self.rowBudget)

        # Color mode
        self.cmbColorMode = QtWidgets.QComboBox()
        self.cmbColorMode.addItems(["Solid", "Normal RGB", "Normal Colormap"])
        self.cmbColorMode.currentTextChanged.connect(self._on_mode_changed)
        self.rowColorMode = self.cmbColorMode
        form.addRow("Color mode", self.rowColorMode)

        # Solid color
        self.btnColor = QtWidgets.QPushButton("Choose…")
        self.btnColor.clicked.connect(self._pick_color)
        self.rowSolid = self.btnColor
        form.addRow("Solid color", self.rowSolid)

        # Colormap
        self.cmbCmap = QtWidgets.QComboBox()
        self.cmbCmap.addItems(["viridis", "magma", "plasma", "cividis"])
        self.cmbCmap.currentTextChanged.connect(self.sigColormapChanged)
        self.rowCmap = self.cmbCmap
        form.addRow("Colormap", self.rowCmap)

        # Mesh representation
        self.cmbMeshRep = QtWidgets.QComboBox()
        self.cmbMeshRep.addItems(["Surface", "Wireframe"])
        self.cmbMeshRep.currentTextChanged.connect(self.sigMeshRepresentationChanged)
        self.rowMeshRep = self.cmbMeshRep
        form.addRow("Representation", self.rowMeshRep)

        # Mesh opacity
        self.sldOpacity = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sldOpacity.setRange(0, 100)
        self.sldOpacity.setValue(100)
        self.spinOpacity = QtWidgets.QSpinBox()
        self.spinOpacity.setRange(0, 100)
        self.spinOpacity.setValue(100)
        self.sldOpacity.valueChanged.connect(self.spinOpacity.setValue)
        self.spinOpacity.valueChanged.connect(self.sldOpacity.setValue)
        self.sldOpacity.valueChanged.connect(self.sigMeshOpacityChanged)
        opaRow = QtWidgets.QHBoxLayout()
        opaRow.addWidget(self.sldOpacity, 1)
        opaRow.addWidget(self.spinOpacity)
        self.rowOpacity = _wrap(opaRow)
        form.addRow("Opacity", self.rowOpacity)

        self._kind = "points"
        self._update_visibility()

    def _on_mode_changed(self, text: str) -> None:
        """Handle color mode change."""
        self.sigColorModeChanged.emit(text)
        self._update_visibility()

    def _update_visibility(self) -> None:
        """Enable/disable controls based on color mode."""
        solid = self.cmbColorMode.currentText() == "Solid"
        cmap = self.cmbColorMode.currentText() == "Normal Colormap"
        self.btnColor.setEnabled(solid and self._kind == "points")
        self.cmbCmap.setEnabled(cmap and self._kind == "points")
        self._set_row_visible(self.rowSolid, solid and self._kind == "points")
        self._set_row_visible(self.rowCmap, cmap and self._kind == "points")
        self._set_row_visible(self.rowSize, self._kind == "points")
        self._set_row_visible(self.rowBudget, self._kind == "points")
        self._set_row_visible(self.rowColorMode, self._kind == "points")
        self._set_row_visible(self.rowMeshRep, self._kind == "mesh")
        self._set_row_visible(self.rowOpacity, self._kind == "mesh")

    def _pick_color(self) -> None:
        """Open color dialog and emit chosen color."""
        col = QtWidgets.QColorDialog.getColor(parent=self)
        if col.isValid():
            pal = self.btnColor.palette()
            pal.setColor(QtGui.QPalette.Button, col)
            self.btnColor.setPalette(pal)
            self.btnColor.setAutoFillBackground(True)
            self.sigSolidColorChanged.emit(col)

    def _set_row_visible(self, widget: QtWidgets.QWidget, visible: bool) -> None:
        """Utility to show/hide a form row."""
        widget.setVisible(visible)
        lbl = self.form.labelForField(widget)
        if lbl is not None:
            lbl.setVisible(visible)

    def set_mode(self, kind: str) -> None:
        """Imposta il tipo di dataset (points|mesh).
        Set the dataset type (points|mesh)."""
        self._kind = kind
        self._update_visibility()

    def apply_properties(self, props: dict) -> None:
        """Carica le proprietà nel pannello.
        Load properties into the panel."""
        kind = props.get("kind", "points")
        if kind == "points":
            self.sldSize.setValue(int(props.get("point_size", 3)))
            self.sldBudget.setValue(int(props.get("point_budget", 100)))
            self.cmbColorMode.setCurrentText(props.get("color_mode", "Normal RGB"))
            self.cmbCmap.setCurrentText(props.get("colormap", "viridis"))
            col = props.get("solid_color")
            if col is not None:
                r, g, b = col
                if r <= 1 and g <= 1 and b <= 1:
                    r, g, b = int(r * 255), int(g * 255), int(b * 255)
                c = QtGui.QColor(int(r), int(g), int(b))
                pal = self.btnColor.palette()
                pal.setColor(QtGui.QPalette.Button, c)
                self.btnColor.setPalette(pal)
                self.btnColor.setAutoFillBackground(True)
        else:
            self.cmbMeshRep.setCurrentText(props.get("representation", "Surface").capitalize())
            self.sldOpacity.setValue(int(props.get("opacity", 100)))
        self.set_mode(kind)


def _wrap(layout: QtWidgets.QLayout) -> QtWidgets.QWidget:
    """Wrap a layout into a QWidget (utility for forms)."""
    w = QtWidgets.QWidget()
    w.setLayout(layout)
    return w