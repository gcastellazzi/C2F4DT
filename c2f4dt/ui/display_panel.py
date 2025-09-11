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

        # Normals (compute) signals
        sigComputeNormals: Emitted when compute normals is requested.
        sigFastNormalsChanged: Emitted when fast normals checkbox changes.
        sigNormalsStyleChanged: Emitted when normals display style changes.
        sigNormalsColorChanged: Emitted when normals uniform color changes.
        sigNormalsPercentChanged/sigNormalsScaleChanged: Emitted when fraction/scale of shown normals changes.
    """

    # Signals
    sigPointSizeChanged = QtCore.Signal(int)
    sigPointBudgetChanged = QtCore.Signal(int)
    sigColorModeChanged = QtCore.Signal(str)
    sigSolidColorChanged = QtCore.Signal(QtGui.QColor)
    sigColormapChanged = QtCore.Signal(str)
    sigMeshRepresentationChanged = QtCore.Signal(str)
    sigMeshOpacityChanged = QtCore.Signal(int)

    # Normals (display) signals
    sigNormalsStyleChanged = QtCore.Signal(str)       # 'Uniform' | 'Axis RGB' | 'RGB Components'
    sigNormalsColorChanged = QtCore.Signal(QtGui.QColor)
    sigNormalsPercentChanged = QtCore.Signal(int)     # 1..100
    sigNormalsScaleChanged = QtCore.Signal(int)       # 1..200

    # Normals (compute) signals
    sigComputeNormals = QtCore.Signal()
    sigFastNormalsChanged = QtCore.Signal(bool)

    def __init__(self, parent=None) -> None:
        """Initialize the panel with controls."""
        super().__init__(parent)
        form = QtWidgets.QFormLayout(self)
        form.setLabelAlignment(QtCore.Qt.AlignRight)
        form.setFormAlignment(QtCore.Qt.AlignTop)
        self.form = form

        # Points visualization group box
        pointsGroupBox = QtWidgets.QGroupBox("Points Visualization")
        pointsLayout = QtWidgets.QFormLayout(pointsGroupBox)

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
        pointsLayout.addRow("Point size", self.rowSize)

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
        pointsLayout.addRow("% points shown", self.rowBudget)

        # Color mode
        self.cmbColorMode = QtWidgets.QComboBox()
        self.cmbColorMode.addItems(["Solid", "Normal RGB", "Normal Colormap"])
        self.cmbColorMode.currentTextChanged.connect(self._on_mode_changed)
        self.rowColorMode = self.cmbColorMode
        pointsLayout.addRow("Color mode", self.rowColorMode)

        # Solid color
        self.btnColor = QtWidgets.QPushButton("Choose…")
        self.btnColor.clicked.connect(self._pick_color)

        # Color preview square
        self.colorPreview = QtWidgets.QLabel()
        self.colorPreview.setFixedSize(20, 20)
        self.colorPreview.setAutoFillBackground(True)
        self._update_color_preview(QtGui.QColor(255, 255, 255))  # Default to white

        # Layout for button and color preview
        colorRow = QtWidgets.QHBoxLayout()
        colorRow.addWidget(self.btnColor)
        colorRow.addWidget(self.colorPreview)

        self.rowSolid = _wrap(colorRow)
        pointsLayout.addRow("Solid color", self.rowSolid)

        # Colormap
        self.cmbCmap = QtWidgets.QComboBox()
        self.cmbCmap.addItems(["viridis", "magma", "plasma", "cividis"])
        self.cmbCmap.currentTextChanged.connect(self.sigColormapChanged)
        self.rowCmap = self.cmbCmap
        pointsLayout.addRow("Colormap", self.rowCmap)

        # Add the group box to the main form layout
        form.addRow(pointsGroupBox)

        # --- Normals visualization controls ---
        self._build_normals_section(form)

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
        self.colorPreview.setEnabled(solid and self._kind == "points")
        self.cmbCmap.setEnabled(cmap and self._kind == "points")
        self._set_row_visible(self.rowSolid, solid and self._kind == "points")
        self._set_row_visible(self.rowCmap, cmap and self._kind == "points")
        self._set_row_visible(self.rowSize, self._kind == "points")
        self._set_row_visible(self.rowBudget, self._kind == "points")
        self._set_row_visible(self.rowColorMode, self._kind == "points")
        self._set_row_visible(self.rowMeshRep, self._kind == "mesh")
        self._set_row_visible(self.rowOpacity, self._kind == "mesh")
        # Normals display (show for points by default; adjust if you also support mesh normals)
        self._set_row_visible(self.rowNormals, self._kind == "points")

    def _pick_color(self) -> None:
        """Open color dialog and emit chosen color."""
        col = QtWidgets.QColorDialog.getColor(parent=self)
        if col.isValid():
            self._update_color_preview(col)
            self.sigSolidColorChanged.emit(col)

    def _update_color_preview(self, col: QtGui.QColor) -> None:
        """Update the small color square beside the button.

        The square mirrors the currently selected solid color and follows
        the visibility/enabled state of the Solid controls. If ``col`` is not
        valid, fall back to a neutral gray.
        """
        if not isinstance(col, QtGui.QColor) or not col.isValid():
            col = QtGui.QColor(200, 200, 200)
        # Apply a simple stylesheet so it also works across themes
        try:
            hexcol = col.name(QtGui.QColor.HexRgb)
        except Exception:
            hexcol = col.name()
        self.colorPreview.setStyleSheet(
            f"QLabel {{ border:1px solid #666; border-radius:2px; background-color:{hexcol}; }}"
        )
        self.colorPreview.setToolTip(f"Solid color: {hexcol}")

    def _set_row_visible(self, widget: QtWidgets.QWidget, visible: bool) -> None:
        """Utility to show/hide a form row."""
        widget.setVisible(visible)
        lbl = self.form.labelForField(widget)
        if lbl is not None:
            lbl.setVisible(visible)

    def fast_normals_enabled(self) -> bool:
        """Return True if 'Fast normals' is currently enabled."""
        try:
            return bool(self.chkFastNormals.isChecked())
        except Exception:
            return True

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
                self._update_color_preview(c)
                pal = self.btnColor.palette()
                # pal.setColor(QtGui.QPalette.Button, c)
                # self.btnColor.setPalette(pal)
                self.btnColor.setAutoFillBackground(True)
            # Normals visualization (optional fields)
            style = props.get("normals_style")
            if style in ("Uniform", "Axis RGB", "RGB Components"):
                self.comboNormalsStyle.blockSignals(True)
                self.comboNormalsStyle.setCurrentText(style)
                self.comboNormalsStyle.blockSignals(False)
            if "normals_percent" in props:
                self.spinNormalsPercent.blockSignals(True)
                self.spinNormalsPercent.setValue(int(props["normals_percent"]))
                self.spinNormalsPercent.blockSignals(False)
            if "normals_scale" in props:
                self.sliderNormalsScale.blockSignals(True)
                self.sliderNormalsScale.setValue(int(props["normals_scale"]))
                self.sliderNormalsScale.blockSignals(False)
        else:
            self.cmbMeshRep.setCurrentText(props.get("representation", "Surface").capitalize())
            self.sldOpacity.setValue(int(props.get("opacity", 100)))
        self.set_mode(kind)

    # ------------------------------------------------------------------
    # Normals display UI
    # ------------------------------------------------------------------
    def _build_normals_section(self, form: QtWidgets.QFormLayout) -> None:
        """Build UI controls for normals visualization.
        
        Controls:
          * Style: Uniform | Axis RGB | RGB Components
          * Color button (enabled only for Uniform)
          * Percent shown: 1..100 %
          * Vector size (scale): 1..200
        """

        normalsGroupBox = QtWidgets.QGroupBox("Normals Visualization")
        normalsLayout = QtWidgets.QVBoxLayout(normalsGroupBox)

        bar_plugin = QtWidgets.QHBoxLayout()
        self.btnComputeNormals = QtWidgets.QPushButton("Compute normals…")
        self.btnComputeNormals.setObjectName("btnComputeNormals")
        self.chkFastNormals = QtWidgets.QCheckBox("Fast normals")
        self.chkFastNormals.setObjectName("chkFastNormals")
        self.chkFastNormals.setChecked(True)
        bar_plugin.addWidget(self.btnComputeNormals)
        bar_plugin.addWidget(self.chkFastNormals)
        normalsLayout.addLayout(bar_plugin)
        # Emit high-level signals for the host window
        self.btnComputeNormals.clicked.connect(self.sigComputeNormals.emit)
        self.chkFastNormals.toggled.connect(self.sigFastNormalsChanged)

        row1 = QtWidgets.QHBoxLayout()

        # Style combo
        self.comboNormalsStyle = QtWidgets.QComboBox()
        self.comboNormalsStyle.setObjectName("comboNormalsStyle")
        self.comboNormalsStyle.addItems(["Uniform", "Axis RGB", "RGB Components"])  # 3 modes
        self.comboNormalsStyle.currentTextChanged.connect(self.sigNormalsStyleChanged.emit)
        row1.addWidget(self.comboNormalsStyle, 1)

        # Color button (only for Uniform)
        self.btnNormalsColor = QtWidgets.QPushButton("Color…")
        self.btnNormalsColor.setObjectName("btnNormalsColor")
        self.btnNormalsColor.clicked.connect(self._on_pick_normals_color)
        row1.addWidget(self.btnNormalsColor)

        self.rowNormalsStyle = _wrap(row1)
        normalsLayout.addWidget(self.rowNormalsStyle)

        row2 = QtWidgets.QHBoxLayout()
        # Percent of normals to show
        self.spinNormalsPercent = QtWidgets.QSpinBox()
        self.spinNormalsPercent.setRange(1, 100)
        self.spinNormalsPercent.setValue(1)
        self.spinNormalsPercent.setSuffix(" %")
        self.spinNormalsPercent.valueChanged.connect(self.sigNormalsPercentChanged.emit)
        row2.addWidget(QtWidgets.QLabel("Shown:"))
        row2.addWidget(self.spinNormalsPercent)

        # Vector size (glyph scale)
        self.sliderNormalsScale = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sliderNormalsScale.setRange(1, 200)
        self.sliderNormalsScale.setValue(20)
        self.sliderNormalsScale.setObjectName("sliderNormalsScale")
        self.sliderNormalsScale.valueChanged.connect(self.sigNormalsScaleChanged.emit)
        row2.addWidget(QtWidgets.QLabel("Size:"))
        row2.addWidget(self.sliderNormalsScale, 2)

        self.rowNormals = _wrap(row2)
        normalsLayout.addWidget(self.rowNormals)

        # Enable state for color button (only Uniform)
        self._update_normals_color_enabled(self.comboNormalsStyle.currentText())
        self.comboNormalsStyle.currentTextChanged.connect(self._update_normals_color_enabled)

        form.addRow(normalsGroupBox)

    def _on_pick_normals_color(self) -> None:
        """Open a color dialog and emit chosen uniform color for normals."""
        col = QtWidgets.QColorDialog.getColor(parent=self, title="Normals color")
        if col.isValid():
            self.sigNormalsColorChanged.emit(col)

    def _update_normals_color_enabled(self, mode: str) -> None:
        """Enable/disable the normals color button based on the selected style."""
        enable = (mode == "Uniform")
        try:
            self.btnNormalsColor.setEnabled(enable)
        except Exception:
            pass


def _wrap(layout: QtWidgets.QLayout) -> QtWidgets.QWidget:
    """Wrap a layout into a QWidget (utility for forms)."""
    w = QtWidgets.QWidget()
    w.setLayout(layout)
    return w