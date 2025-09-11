# /plugins/units/plugin.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
from PySide6 import QtCore, QtGui, QtWidgets

# -----------------------------
# Conversion tables (base SI)
# -----------------------------
LEN = {  # to meters
    "m": 1.0, "cm": 0.01, "mm": 0.001, "µm": 1e-6,
    "in": 0.0254, "ft": 0.3048,
}
FORCE = {  # to newtons
    "N": 1.0, "kN": 1e3, "MN": 1e6, "lbf": 4.4482216153,
}
MASS = {  # to kilograms
    "kg": 1.0, "g": 1e-3, "t": 1e3, "lb": 0.45359237,
}
TIME = {"s": 1.0, "ms": 1e-3, "min": 60.0, "h": 3600.0}
TEMP = {"°C": ("C",), "K": ("K",), "°F": ("F",)}
# pressure = force / length^2
PRESSURE = {
    "Pa": 1.0, "kPa": 1e3, "MPa": 1e6, "GPa": 1e9,
    "bar": 1e5, "psi": 6894.757293168,
}

def convert_linear(value: float, u_from: Dict[str, float], from_unit: str, to_unit: str) -> float:
    """Generic linear factor conversion (e.g., length, force, pressure)."""
    f = u_from[from_unit]
    t = u_from[to_unit]
    return value * (f / t)

def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """Handle °C/°F/K non-linear conversions."""
    if from_unit == to_unit:
        return float(value)
    # to Kelvin
    if from_unit == "K":
        k = float(value)
    elif from_unit == "°C":
        k = float(value) + 273.15
    elif from_unit == "°F":
        k = (float(value) - 32.0) * 5.0/9.0 + 273.15
    else:
        raise ValueError("Unknown temperature unit")
    # from Kelvin
    if to_unit == "K":
        return k
    if to_unit == "°C":
        return k - 273.15
    if to_unit == "°F":
        return (k - 273.15) * 9.0/5.0 + 32.0
    raise ValueError("Unknown temperature unit")

# -----------------------------------
# Units state + helpers
# -----------------------------------
@dataclass
class UnitsState:
    length: str = "m"
    mass: str = "kg"
    time: str = "s"
    force: str = "N"
    temperature: str = "°C"
    pressure: str = "Pa"   # derived, user-overridable
    density: str = "kg/m³" # derived, informative
    energy: str = "J"      # derived, informative

    def suggest_derived(self) -> None:
        """Set sensible defaults for derived quantities based on base units."""
        # pressure suggestion from force/length
        # keep user's explicit choice if already set to something consistent
        default_pressure = "Pa"
        if self.force in ("kN", "MN") and self.length in ("m",):
            default_pressure = "MPa"  # common in mech
        elif self.force in ("N",) and self.length in ("m",):
            default_pressure = "Pa"
        elif self.length in ("mm", "cm"):
            default_pressure = "MPa"  # still convenient
        self.pressure = self.pressure or default_pressure

# -----------------------------------
# Dialog UI
# -----------------------------------
class UnitsDialog(QtWidgets.QDialog):
    """Dialog to select units and perform quick conversions."""
    sigUnitsChanged = QtCore.Signal(object)  # emits UnitsState

    def __init__(self, parent=None, initial: Optional[UnitsState] = None):
        super().__init__(parent)
        self.setWindowTitle("Units & Scale")
        self._state = initial or UnitsState()
        self._build_ui()
        self._apply_state(self._state)

    def _build_ui(self) -> None:
        form = QtWidgets.QFormLayout(self)

        # Base units combos
        self.cmbLen = QtWidgets.QComboBox(); self.cmbLen.addItems(list(LEN.keys()))
        self.cmbMass = QtWidgets.QComboBox(); self.cmbMass.addItems(list(MASS.keys()))
        self.cmbTime = QtWidgets.QComboBox(); self.cmbTime.addItems(list(TIME.keys()))
        self.cmbForce = QtWidgets.QComboBox(); self.cmbForce.addItems(list(FORCE.keys()))
        self.cmbTemp = QtWidgets.QComboBox(); self.cmbTemp.addItems(list(TEMP.keys()))

        # Derived editable (pressure), the rest informative
        self.cmbPressure = QtWidgets.QComboBox(); self.cmbPressure.addItems(list(PRESSURE.keys()))
        self.lblDensity = QtWidgets.QLabel("kg/m³")
        self.lblEnergy = QtWidgets.QLabel("J")

        form.addRow("Length", self.cmbLen)
        form.addRow("Mass", self.cmbMass)
        form.addRow("Time", self.cmbTime)
        form.addRow("Force", self.cmbForce)
        form.addRow("Temperature", self.cmbTemp)
        form.addRow(QtWidgets.QLabel("<b>Derived</b>"))
        form.addRow("Pressure", self.cmbPressure)
        form.addRow("Density", self.lblDensity)
        form.addRow("Energy", self.lblEnergy)

        # Quick converter
        box = QtWidgets.QGroupBox("Quick convert")
        h = QtWidgets.QHBoxLayout(box)
        self.spinValue = QtWidgets.QDoubleSpinBox(); self.spinValue.setRange(-1e12, 1e12); self.spinValue.setDecimals(6); self.spinValue.setValue(1.0)
        self.cmbFrom = QtWidgets.QComboBox(); self.cmbTo = QtWidgets.QComboBox()
        self.lblResult = QtWidgets.QLineEdit(); self.lblResult.setReadOnly(True)
        # default to length
        self.cmbFrom.addItems(list(LEN.keys()))
        self.cmbTo.addItems(list(LEN.keys()))
        h.addWidget(self.spinValue); h.addWidget(self.cmbFrom); h.addWidget(QtWidgets.QLabel("→")); h.addWidget(self.cmbTo); h.addWidget(self.lblResult, 1)
        form.addRow(box)

        # Buttons
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        form.addRow(btns)

        # Signals
        for cmb in (self.cmbLen, self.cmbMass, self.cmbTime, self.cmbForce, self.cmbTemp, self.cmbPressure):
            cmb.currentTextChanged.connect(self._on_units_changed)
        for w in (self.spinValue, self.cmbFrom, self.cmbTo):
            if hasattr(w, "valueChanged"):
                w.valueChanged.connect(self._on_convert)
            else:
                w.currentTextChanged.connect(self._on_convert)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

    def _apply_state(self, st: UnitsState) -> None:
        self.cmbLen.setCurrentText(st.length)
        self.cmbMass.setCurrentText(st.mass)
        self.cmbTime.setCurrentText(st.time)
        self.cmbForce.setCurrentText(st.force)
        self.cmbTemp.setCurrentText(st.temperature)
        st.suggest_derived()
        self.cmbPressure.setCurrentText(st.pressure)
        self.lblDensity.setText("kg/m³")  # informative only
        self.lblEnergy.setText("J")
        self._on_convert()

    def _on_units_changed(self) -> None:
        st = UnitsState(
            length=self.cmbLen.currentText(),
            mass=self.cmbMass.currentText(),
            time=self.cmbTime.currentText(),
            force=self.cmbForce.currentText(),
            temperature=self.cmbTemp.currentText(),
            pressure=self.cmbPressure.currentText(),
        )
        st.suggest_derived()
        self._state = st
        self.sigUnitsChanged.emit(st)
        self._on_convert()

    def _on_convert(self) -> None:
        v = self.spinValue.value()
        u_from = self.cmbFrom.currentText()
        u_to = self.cmbTo.currentText()

        # try linear (length/force/pressure)
        try:
            dic = None
            for table in (LEN, FORCE, PRESSURE):
                if u_from in table and u_to in table:
                    dic = table; break
            if dic:
                out = convert_linear(v, dic, u_from, u_to)
                self.lblResult.setText(f"{out:.6g}")
                return
        except Exception:
            pass
        # temperature?
        if u_from in TEMP and u_to in TEMP:
            try:
                out = convert_temperature(v, u_from, u_to)
                self.lblResult.setText(f"{out:.6g}")
                return
            except Exception:
                pass
        self.lblResult.setText("—")

    def state(self) -> UnitsState:
        return self._state

# -----------------------------------
# Overlay / ruler helpers (best effort)
# -----------------------------------
class UnitsOverlay:
    """Manages a text overlay and a simple ruler actor in the Viewer3D."""
    def __init__(self, viewer3d):
        self.viewer = viewer3d
        self._text_id = None
        self._ruler_actor = None
        self._observer_tag = None
        # default readable length on screen as fraction of bbox diag
        self._fraction_of_diag = 0.25

    def _bbox_diag(self) -> float:
        try:
            import numpy as np
            # combine bounds of all visible datasets
            dd = getattr(self.viewer, "_datasets", [])
            mins, maxs = [], []
            for rec in dd:
                if not rec.get("visible", True): 
                    continue
                pd = rec.get("pdata") or rec.get("full_pdata")
                if pd is None or not hasattr(pd, "bounds"):
                    continue
                b = pd.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
                mins.append([b[0], b[2], b[4]]); maxs.append([b[1], b[3], b[5]])
            if not mins:
                return 1.0
            lo = np.min(np.array(mins), axis=0); hi = np.max(np.array(maxs), axis=0)
            return float(np.linalg.norm(hi - lo))
        except Exception:
            return 1.0

    def show_text(self, units: UnitsState) -> None:
        txt = f"Units: L={units.length}, F={units.force}, p={units.pressure}, T={units.temperature}"
        try:
            # try dedicated overlay API if present
            method = getattr(self.viewer, "add_overlay_text", None)
            if callable(method):
                if self._text_id is None:
                    self._text_id = method("units_overlay", txt, pos="lower_left")
                else:
                    getattr(self.viewer, "update_overlay_text", lambda *_: None)("units_overlay", txt)
            else:
                # fallback: plotter.add_text
                if self._text_id is None:
                    self._text_id = self.viewer.plotter.add_text(txt, position="lower_left", font_size=10, name="units_overlay_text")
                else:
                    self.viewer.plotter.remove_actor("units_overlay_text")
                    self._text_id = self.viewer.plotter.add_text(txt, position="lower_left", font_size=10, name="units_overlay_text")
            self.viewer.refresh()
        except Exception:
            pass

    def _build_ruler_poly(self, length_world: float) -> "pyvista.PolyData":
        import numpy as np
        import pyvista as pv
        # draw a horizontal line with small ticks (10 segments)
        nseg = 10
        x0 = 0.0
        x1 = length_world
        y = 0.0
        z = 0.0
        pts = [[x0, y, z], [x1, y, z]]
        lines = [2, 0, 1]  # single segment
        # small ticks every segment
        tick_h = 0.02 * length_world
        for i in range(nseg + 1):
            xi = x0 + (x1 - x0) * i / nseg
            pts.append([xi, y, z])
            pts.append([xi, y + tick_h, z])
            n = len(pts)
            lines += [2, n - 2, n - 1]
        poly = pv.PolyData(np.array(pts))
        poly.lines = np.array(lines)
        return poly

    def _place_ruler_origin(self) -> Tuple[float, float, float]:
        """Place the ruler near the min bounds on X/Y (world coords)."""
        try:
            dd = getattr(self.viewer, "_datasets", [])
            xmin = ymin = zmin = None
            for rec in dd:
                if not rec.get("visible", True): 
                    continue
                pd = rec.get("pdata") or rec.get("full_pdata")
                if pd is None or not hasattr(pd, "bounds"): 
                    continue
                b = pd.bounds
                xmin = b[0] if xmin is None else min(xmin, b[0])
                ymin = b[2] if ymin is None else min(ymin, b[2])
                zmin = b[4] if zmin is None else min(zmin, b[4])
            if xmin is None:
                return (0.0, 0.0, 0.0)
            return (float(xmin), float(ymin), float(zmin))
        except Exception:
            return (0.0, 0.0, 0.0)

    def update_ruler(self, units: UnitsState) -> None:
        """Create/refresh a world-space ruler roughly a quarter of the scene diagonal."""
        try:
            import pyvista as pv
            L = max(1e-9, self._bbox_diag() * self._fraction_of_diag)
            poly = self._build_ruler_poly(L)
            # translate to a corner of the scene
            ox, oy, oz = self._place_ruler_origin()
            poly = poly.translate((ox, oy, oz), inplace=False)

            if self._ruler_actor is not None:
                try:
                    self.viewer.plotter.remove_actor(self._ruler_actor)
                except Exception:
                    pass
                self._ruler_actor = None

            self._ruler_actor = self.viewer.plotter.add_mesh(
                poly, color=(1, 1, 1), line_width=2, name="units_ruler"
            )
            # add a small label with the world length and unit
            label = f"{L:.3g} {units.length}"
            # refresh text (independent overlay)
            self.show_text(units)
            # Optionally: label near ruler end
            try:
                self.viewer.plotter.add_point_labels(
                    [(ox + L, oy, oz)], [label], font_size=10, text_color="white", name="units_ruler_label"
                )
            except Exception:
                pass

            # observe camera interactions to keep ruler readable (optional)
            if self._observer_tag is None:
                try:
                    interactor = self.viewer.plotter.iren
                    self._observer_tag = interactor.AddObserver("EndInteractionEvent", lambda *_: self.update_ruler(units))
                except Exception:
                    pass

            self.viewer.refresh()
        except Exception:
            pass

# -----------------------------------
# Plugin entry
# -----------------------------------
# ... (intestazioni invariate)

class UnitsPlugin(QtCore.QObject):
    """Wires menu entry, dialog, and overlay into the host MainWindow."""
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.state = UnitsState()
        self.overlay = UnitsOverlay(window.viewer3d)

        # QAction in Tools menu (or Plugins)
        self.action = QtGui.QAction(QtGui.QIcon(), "Units & Scale…", self)
        # ⬇️ evita che Qt passi l'argomento 'checked' al metodo
        self.action.triggered.connect(lambda _checked=False: self.open_dialog())

        # try to add under Tools
        try:
            mb = window.menuBar()
            m_tools = None
            for a in mb.actions():
                if a.text().replace("&", "") == "Tools":
                    m_tools = a.menu(); break
            if m_tools is None:
                m_tools = mb.addMenu("&Tools")
            m_tools.addAction(self.action)
        except Exception:
            pass

        # initial overlay
        self.overlay.show_text(self.state)
        self.overlay.update_ruler(self.state)

    # Accetta anche il bool opzionale
    @QtCore.Slot(bool)
    def open_dialog(self, checked: bool = False):
        dlg = UnitsDialog(self.window, initial=self.state)
        dlg.sigUnitsChanged.connect(self._on_units_changed_live)  # live preview
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            self.state = dlg.state()
            # (eventuale persistenza)
        # refresh finale in entrambi i casi
        self.overlay.show_text(self.state)
        self.overlay.update_ruler(self.state)

    def _on_units_changed_live(self, st: UnitsState):
        self.state = st
        self.overlay.show_text(st)
        self.overlay.update_ruler(st)

    # ⬇️ così il PluginManager può “eseguire” il plugin
    def run(self, *args, **kwargs):
        return self.open_dialog(False)

# Public entry point (factory) compatibile col PluginManager
def load_plugin(parent):
    return UnitsPlugin(parent)

# Back-compat: allow entry_point "plugin:register"
def register(parent):
    return UnitsPlugin(parent)