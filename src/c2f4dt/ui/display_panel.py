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
        # Width constraints must be set on widgets, not layouts
        self.setMaximumWidth(300)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        # (optional) tighten paddings to fit narrow panel nicely
        form.setHorizontalSpacing(6)
        form.setVerticalSpacing(6)
        form.setContentsMargins(6, 6, 6, 6)
        self.form = form

        # ---- Capabilities (what is present in scene) ----
        self._has_points = True
        self._has_mesh = True
        self._kind = "points"  # kept for backward compat; no longer hides rows

        # Points visualization group box (ALWAYS visible)
        pointsGroupBox = QtWidgets.QGroupBox("Points Visualization")
        pointsGroupBox.setMaximumWidth(300)
        pointsGroupBox.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
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

        # Solid color (+ preview)
        self.btnColor = QtWidgets.QPushButton("Choose…")
        self.btnColor.clicked.connect(self._pick_color)
        self.colorPreview = QtWidgets.QLabel()
        self.colorPreview.setFixedSize(20, 20)
        self.colorPreview.setAutoFillBackground(True)
        self._update_color_preview(QtGui.QColor(255, 255, 255))  # Default to white
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

        form.addRow(pointsGroupBox)

        # --- Normals visualization controls (ALWAYS visible, but may be disabled) ---
        self._build_normals_section(form)

        # --- Mesh visualization group box (NEW & ALWAYS visible) ---
        meshGroupBox = QtWidgets.QGroupBox("Mesh Visualization")
        meshGroupBox.setMaximumWidth(300)
        meshGroupBox.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        meshLayout = QtWidgets.QFormLayout(meshGroupBox)

        # Representation
        self.cmbMeshRep = QtWidgets.QComboBox()
        self.cmbMeshRep.addItems(["Surface", "Wireframe", "Surface with Edges", "Points"])
        self.cmbMeshRep.currentTextChanged.connect(self.sigMeshRepresentationChanged)
        self.rowMeshRep = self.cmbMeshRep
        meshLayout.addRow("Representation", self.rowMeshRep)

        # Opacity
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
        meshLayout.addRow("Opacity", self.rowOpacity)

        form.addRow(meshGroupBox)

        # Initial enable/disable pass
        self._update_visibility()

    # NOTE: Changing color mode should not recreate the scalar bar.
    # The viewer manages bar placement via `set_colorbar_mode`.
    def _on_mode_changed(self, text: str) -> None:
        """Handle color mode change."""
        self.sigColorModeChanged.emit(text)
        self._update_visibility()

    def _update_visibility(self) -> None:
        """Enable/disable controls based on current capabilities.

        Rules:
        - Points controls are always visible; enabled only if _has_points is True.
        - Mesh controls are always visible; enabled only if _has_mesh is True.
        - Within points controls:
            * When color mode == 'Solid' -> enable solid color widgets, disable colormap.
            * When color mode == 'Normal Colormap' -> enable colormap, disable solid color.
            * 'Normal RGB' -> both solid and colormap selectors are disabled (RGB is derived).
        - Normals controls are enabled only if _has_points is True.
        """
        has_pts = bool(self._has_points)
        has_mesh = bool(self._has_mesh)

        # Points: base enables
        self.sldSize.setEnabled(has_pts)
        self.spinSize.setEnabled(has_pts)
        self.sldBudget.setEnabled(has_pts)
        self.spinBudget.setEnabled(has_pts)
        self.cmbColorMode.setEnabled(has_pts)

        # Points: color sub-modes
        mode = self.cmbColorMode.currentText()
        solid_mode = (mode == "Solid")
        cmap_mode = (mode == "Normal Colormap")

        self.btnColor.setEnabled(has_pts and solid_mode)
        self.colorPreview.setEnabled(has_pts and solid_mode)
        self.cmbCmap.setEnabled(has_pts and cmap_mode)

        # Normals
        self.comboNormalsStyle.setEnabled(has_pts)
        self.btnNormalsColor.setEnabled(has_pts and self.comboNormalsStyle.currentText() == "Uniform")
        self.spinNormalsPercent.setEnabled(has_pts)
        self.sliderNormalsScale.setEnabled(has_pts)

        # Mesh
        self.cmbMeshRep.setEnabled(has_mesh)
        self.sldOpacity.setEnabled(has_mesh)
        self.spinOpacity.setEnabled(has_mesh)

    def set_capabilities(self, has_points: bool, has_mesh: bool) -> None:
        """Declare what is currently available in the scene (points and/or mesh).

        Call this whenever the current tree selection changes, or after imports.
        The panel keeps every section visible and only toggles enabled state.
        """
        self._has_points = bool(has_points)
        self._has_mesh = bool(has_mesh)
        self._update_visibility()
        
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
        """Kept for backward compatibility. Does not hide sections anymore."""
        self._kind = kind
        self._update_visibility()

    def apply_properties(self, props: dict) -> None:
        """Load properties into the panel (non-destructive to visibility).

        This method must NOT emit any signals, as it is used to synchronize UI
        widgets with external state (e.g., loading). All widget updates are signal-safe.
        """
        # Optional: detect availability hints
        self._has_points = bool(props.get("has_points", self._has_points))
        self._has_mesh = bool(props.get("has_mesh", self._has_mesh))

        kind = props.get("kind", "points")

        # Point size (signal-safe)
        try:
            self.sldSize.blockSignals(True)
            self.spinSize.blockSignals(True)
            self.sldSize.setValue(int(props.get("point_size", self.sldSize.value())))
            self.spinSize.setValue(self.sldSize.value())
        finally:
            self.sldSize.blockSignals(False)
            self.spinSize.blockSignals(False)

        # Point budget (signal-safe)
        try:
            self.sldBudget.blockSignals(True)
            self.spinBudget.blockSignals(True)
            self.sldBudget.setValue(int(props.get("point_budget", self.sldBudget.value())))
            self.spinBudget.setValue(self.sldBudget.value())
        finally:
            self.sldBudget.blockSignals(False)
            self.spinBudget.blockSignals(False)

        # Color mode (signal-safe)
        try:
            self.cmbColorMode.blockSignals(True)
            self.cmbColorMode.setCurrentText(str(props.get("color_mode", self.cmbColorMode.currentText())))
        finally:
            self.cmbColorMode.blockSignals(False)

        # Colormap (signal-safe)
        try:
            self.cmbCmap.blockSignals(True)
            self.cmbCmap.setCurrentText(str(props.get("colormap", self.cmbCmap.currentText())))
        finally:
            self.cmbCmap.blockSignals(False)

        # Solid color preview (already signal-free, just update preview)
        col = props.get("solid_color")
        if col is not None:
            try:
                r, g, b = col
                if r <= 1 and g <= 1 and b <= 1:
                    r, g, b = int(r * 255), int(g * 255), int(b * 255)
                self._update_color_preview(QtGui.QColor(int(r), int(g), int(b)))
            except Exception:
                pass

        # Normals
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

        # Mesh
        if "representation" in props:
            self.cmbMeshRep.setCurrentText(str(props["representation"]))
        if "opacity" in props:
            self.sldOpacity.setValue(int(props["opacity"]))

        # Refresh enable/disable state only (do not emit any signals)
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
        normalsGroupBox.setMaximumWidth(300)
        normalsGroupBox.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
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
    w.setMaximumWidth(300)
    w.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
    return w