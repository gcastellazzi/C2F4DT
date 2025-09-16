# -*- coding: utf-8 -*-
"""
VTK Import & Display plugin for C2F4DT.

COSA FA IL PLUGIN
-----------------
- Aggiunge una voce di menu: File ▸ Import VTK… (con scorciatoia)
- Importa un *singolo* file VTK/VTU/VTP/VTM/VTS/VTR/VTI/OBJ/STL
- Se MultiBlock: tiene *un unico attore* (come da preferenza)
- Se il reader espone `time_values`: aggiunge un semplice time-slider nel box
- Crea un box “VTK Display” nello scrollDISPLAY con controlli stile ParaView:
  * Representation: Points / Wireframe / Surface / Surface with Edges / Volume (se applicabile)
  * Color by: Solid Color / arrays (PointData/CellData), per vettori Mag/ X / Y / Z
  * LUT + invert, Rescale to Data, Scalar Bar On/Off
  * Opacity, Point Size, Line Width
  * Edge visibility + Edge color
  * Lighting base (toggle)
- Applica i cambi **live** al dataset “corrente” (selezione nel treeMCTS)
- Salva in mcts il `source_path` + scelte essenziali (estendibile verso uno “style file”)

COME ADATTARLO
--------------
- Vedi i metodi `_apply_*` per mappare i controlli UI alle API del tuo Viewer3D
- Se una API non è disponibile, i punti `TODO` indicano dove aggiungere il fallback
"""

from __future__ import annotations
from typing import Optional, Tuple, List, Dict
import os

from PySide6 import QtCore, QtGui, QtWidgets

# PyVista è una dipendenza dichiarata in plugin.yaml
import pyvista as pv


# ---------------------------------------------------------------------
# Helpers UI
# ---------------------------------------------------------------------
def _add_to_display_panel(window, title: str, widget: QtWidgets.QWidget) -> None:
    """
    Inserisce un *gruppo* nel pannello di destra (scrollDISPLAY).
    Preferisce un eventuale API `add_plugin_section`, altrimenti inserisce
    direttamente nel layout principale del DisplayPanel.
    """
    try:
        if hasattr(window.displayPanel, "add_plugin_section"):
            window.displayPanel.add_plugin_section(title, widget)
            return
    except Exception:
        pass

    # Fallback: cassa dentro un QGroupBox e appendi al layout verticale principale.
    box = QtWidgets.QGroupBox(title)
    box.setMaximumWidth(300)
    lay = QtWidgets.QVBoxLayout(box)
    lay.setContentsMargins(8, 8, 8, 8)
    lay.addWidget(widget)
    widget.setMaximumWidth(300)
    widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
    box.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
    try:
        # displayPanel è già inserito in una QScrollArea; usiamo il suo layout principale
        lp = window.displayPanel.layout()
        if lp is None:
            lp = QtWidgets.QVBoxLayout(window.displayPanel)
        lp.addWidget(box)
        lp.addStretch(0)
    except Exception:
        pass


def _solid_color_button() -> QtWidgets.QPushButton:
    btn = QtWidgets.QPushButton("Solid Color…")
    btn.setObjectName("btnVTKSolidColor")
    return btn


# ---------------------------------------------------------------------
# Plugin principale
# ---------------------------------------------------------------------
class VTKImportPlugin(QtCore.QObject):
    """
    Plugin “VTK Import & Display”.

    Struttura:
      - Azione menu + scorciatoia per import
      - Box controlli visualizzazione, applicati al dataset corrente
      - Rilevamento (best effort) del time-series via reader PyVista
    """
    def __init__(self, window):
        super().__init__(window)
        self.window = window

        # Stato UI (per dataset corrente – si aggiorna ad ogni selezione nel tree)
        self._current_ds: Optional[int] = None
        self._time_values: Optional[List[float]] = None
        self._time_idx: int = 0

        # ----- Menu & scorciatoie --------------------------------------
        self._action_import = QtGui.QAction(QtGui.QIcon(), "Import VTK…", self)
        # Shortcut: ⌘⇧I (mac) / Ctrl+Shift+I (others)
        self._action_import.setShortcut(QtGui.QKeySequence("Ctrl+Shift+I"))
        # Nota: Qt convertirà su macOS in Cmd in base al platform shortcut context
        self._action_import.triggered.connect(self.open_dialog)

        # Aggiungi in File
        try:
            mb = window.menuBar()
            for a in mb.actions():
                if a.text().replace("&", "") == "File":
                    a.menu().addAction(self._action_import)
                    break
        except Exception:
            pass

        # ----- Box UI in DisplayPanel ---------------------------------
        self._panel = self._build_display_box()
        _add_to_display_panel(window, "VTK Display", self._panel)

        # Tieni sincronizzati i controlli con il dataset selezionato
        try:
            window.treeMCTS.itemSelectionChanged.connect(self._on_tree_selection_changed)
        except Exception:
            pass

    # -----------------------------------------------------------------
    # DIALOG DI IMPORT
    # -----------------------------------------------------------------
    @QtCore.Slot()
    def open_dialog(self):
        """
        Dialog per scegliere UN file e importarlo con PyVista.
        Gestione MultiBlock → UNICO attore.
        Rileva (se presente) una time-series esponendo uno slider.
        """
        dlg = QtWidgets.QFileDialog(self.window, "Import VTK")
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        dlg.setNameFilters([
            "All supported (*.vtk *.vtp *.vtu *.vtr *.vts *.vtm *.vti *.obj *.stl)",
            "VTK legacy (*.vtk)",
            "VTK XML PolyData (*.vtp)",
            "VTK XML UnstructuredGrid (*.vtu)",
            "VTK XML RectilinearGrid (*.vtr)",
            "VTK XML StructuredGrid (*.vts)",
            "VTK XML MultiBlock (*.vtm)",
            "VTK ImageData (*.vti)",
            "Meshes (*.obj *.stl)",
            "All files (*)",
        ])
        if not dlg.exec():
            return
        paths = dlg.selectedFiles()
        if not paths:
            return
        path = paths[0]

        try:
            reader = pv.get_reader(path)
        except Exception as ex:
            self._msg(f"[VTK] Reader error: {ex}", error=True)
            return

        # Prova a leggere time steps, se il reader li espone
        self._time_values = None
        try:
            tvals = getattr(reader, "time_values", None)
            if tvals is not None and len(tvals) > 0:
                self._time_values = list(tvals)
                reader.set_active_time_value(self._time_values[0])
        except Exception:
            self._time_values = None

        try:
            data = reader.read()
        except Exception as ex:
            self._msg(f"[VTK] Read error: {ex}", error=True)
            return

        # MultiBlock → unico attore
        dataset_to_add = data
        try:
            if isinstance(data, pv.MultiBlock):
                # Se vuoi proprio *un unico attore*, puoi renderizzare il MultiBlock direttamente
                # (PyVista gestisce internamente i blocchi con un unico mapper/actor composito).
                dataset_to_add = data
        except Exception:
            pass

        # Aggiungi al viewer
        try:
            ds_index = self._add_dataset_to_viewer(dataset_to_add, path)
        except Exception as ex:
            self._msg(f"[VTK] Viewer add failed: {ex}", error=True)
            return

        # Fit camera
        try:
            self.window.viewer3d.view_fit()
        except Exception:
            pass

        # Seleziona il nuovo dataset nel tree
        try:
            self._select_tree_item_for_ds(ds_index)
        except Exception:
            pass

        # Mostra/aggiorna time slider se necessario
        self._sync_time_slider_visibility()

        # Aggiorna Inspector
        try:
            self.window._refresh_inspector_tree()
        except Exception:
            pass

        self._msg(f"[VTK] Imported: {os.path.basename(path)}")

    def _add_dataset_to_viewer(self, data, path: str) -> int:
        """
        Aggiunge il dataset PyVista al Viewer3D e registra in mcts.
        Ritorna l'indice di dataset (ds_index).
        """
        name = os.path.splitext(os.path.basename(path))[0]

        # Preferisci API dedicate se presenti
        ds_index = None
        try:
            if hasattr(self.window.viewer3d, "add_pyvista_mesh"):
                ds_index = self.window.viewer3d.add_pyvista_mesh(data)
            else:
                # TODO: fallback generico (non usato se hai add_pyvista_mesh)
                actor = self.window.viewer3d.plotter.add_mesh(data, name=name)
                # Registrazione manuale in viewer3d._datasets se necessario…
                # Qui assumiamo l'API ufficiale esistente.
                raise RuntimeError("Viewer3D.add_pyvista_mesh non disponibile")
        except Exception as ex:
            raise

        # Registra in mcts (nuova istanza sempre)
        entry = {
            "name": name,
            "kind": "mesh",            # PolyData / Grid / MultiBlock → lo teniamo come 'mesh'
            "ds_index": ds_index,
            "source_path": path,       # per reopen automatico
            # default stile iniziale
            "representation": "Surface",
            "opacity": 100,
            "color_mode": "Solid Color",
            "solid_color": (255, 255, 255),
            "colormap": "Viridis",
            "scalar_bar": False,
            "edge_visibility": False,
            "edge_color": (0, 0, 0),
            "point_size": 3,
            "line_width": 1,
            "lighting": True,
        }
        self.window.mcts[name] = entry
        self.window.mct = entry  # diventa “corrente”

        # Crea nodo nell’albero se serve (riusa pipeline del MainWindow se disponibile)
        try:
            # Se l’import ufficiale di MainWindow costruisce già il tree, potresti saltare questa parte.
            # Qui costruiamo un nodo minimale come esempio:
            self.window.treeMCTS.blockSignals(True)
            root = QtWidgets.QTreeWidgetItem([name])
            root.setFlags(root.flags() | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsAutoTristate)
            root.setCheckState(0, QtCore.Qt.Checked)
            # metadata sul nodo root
            root.setData(0, QtCore.Qt.UserRole, {"kind": "mesh", "ds": ds_index})
            # Aggiungi albero “Mesh”
            it_mesh = QtWidgets.QTreeWidgetItem(["Mesh"])
            it_mesh.setFlags(it_mesh.flags() | QtCore.Qt.ItemIsUserCheckable)
            it_mesh.setCheckState(0, QtCore.Qt.Checked)
            it_mesh.setData(0, QtCore.Qt.UserRole, {"kind": "mesh", "ds": ds_index})
            root.addChild(it_mesh)

            self.window.treeMCTS.addTopLevelItem(root)
            self.window.treeMCTS.blockSignals(False)
        except Exception:
            pass

        return ds_index

    # -----------------------------------------------------------------
    # BOX CONTROLLI (stile ParaView, applicazione LIVE)
    # -----------------------------------------------------------------
    def _build_display_box(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        w.setMaximumWidth(300)
        w.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        lay = QtWidgets.QFormLayout(w)
        lay.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        lay.setHorizontalSpacing(6)
        lay.setVerticalSpacing(6)
        lay.setContentsMargins(6, 6, 6, 6)

        # Representation
        self.cmbRep = QtWidgets.QComboBox()
        self.cmbRep.addItems(["Points", "Wireframe", "Surface", "Surface with Edges", "Volume"])
        self.cmbRep.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.cmbRep.currentTextChanged.connect(self._on_rep_changed)
        lay.addRow("Representation", self.cmbRep)

        # Color By
        self.cmbColorBy = QtWidgets.QComboBox()
        self.cmbColorBy.setMinimumWidth(120)
        self.cmbColorBy.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.cmbColorBy.currentTextChanged.connect(self._on_color_by_changed)
        lay.addRow("Color by", self.cmbColorBy)

        # Vector component
        self.cmbVectorMode = QtWidgets.QComboBox()
        self.cmbVectorMode.addItems(["Magnitude", "X", "Y", "Z"])
        self.cmbVectorMode.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.cmbVectorMode.currentTextChanged.connect(self._on_color_by_changed)
        lay.addRow("Vector component", self.cmbVectorMode)

        # LUT
        self.cmbLUT = QtWidgets.QComboBox()
        self.cmbLUT.addItems(["Viridis", "Plasma", "CoolWarm", "Gray", "Jet"])
        self.cmbLUT.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.cmbLUT.currentTextChanged.connect(self._on_color_by_changed)
        lay.addRow("LUT", self.cmbLUT)

        # Invert LUT
        self.chkInvertLUT = QtWidgets.QCheckBox("Invert")
        self.chkInvertLUT.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.chkInvertLUT.toggled.connect(self._on_color_by_changed)
        lay.addRow("", self.chkInvertLUT)

        # Solid Color button
        self.btnSolid = _solid_color_button()
        self.btnSolid.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.btnSolid.clicked.connect(self._on_pick_solid_color)
        lay.addRow(self.btnSolid)

        # Scalar range with min/max and buttons
        rngw = QtWidgets.QWidget()
        rngw.setMaximumWidth(260)
        rngw.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        rngLay = QtWidgets.QVBoxLayout(rngw)
        rngLay.setContentsMargins(0,0,0,0)
        rngLay.setSpacing(4)
        frm = QtWidgets.QFormLayout()
        frm.setContentsMargins(0,0,0,0)
        frm.setSpacing(4)
        self.editMin = QtWidgets.QLineEdit()
        self.editMin.setPlaceholderText("min")
        self.editMin.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.editMax = QtWidgets.QLineEdit()
        self.editMax.setPlaceholderText("max")
        self.editMax.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        frm.addRow("Min", self.editMin)
        frm.addRow("Max", self.editMax)
        rngLay.addLayout(frm)
        btnRow = QtWidgets.QHBoxLayout()
        btnRow.setSpacing(4)
        self.btnAuto = QtWidgets.QPushButton("Auto")
        self.btnAuto.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.btnAuto.clicked.connect(self._on_range_auto)
        self.btnRescale = QtWidgets.QPushButton("Rescale")
        self.btnRescale.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.btnRescale.clicked.connect(self._on_rescale_to_data)
        btnRow.addWidget(self.btnAuto)
        btnRow.addWidget(self.btnRescale)
        rngLay.addLayout(btnRow)
        lay.addRow("Scalar range", rngw)

        # Scalar bar
        self.chkScalarBar = QtWidgets.QCheckBox("Show scalar bar")
        self.chkScalarBar.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.chkScalarBar.toggled.connect(self._on_scalar_bar_toggle)
        lay.addRow(self.chkScalarBar)

        # Opacity slider
        self.sldOpacity = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sldOpacity.setRange(0, 100)
        self.sldOpacity.setValue(100)
        self.sldOpacity.setMaximumWidth(260)
        self.sldOpacity.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.sldOpacity.valueChanged.connect(self._on_opacity_changed)
        lay.addRow("Opacity", self.sldOpacity)

        # Point size slider
        self.sldPointSize = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sldPointSize.setRange(1, 15)
        self.sldPointSize.setValue(3)
        self.sldPointSize.setMaximumWidth(260)
        self.sldPointSize.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.sldPointSize.valueChanged.connect(self._on_point_size_changed)
        lay.addRow("Point size", self.sldPointSize)

        # Line width slider
        self.sldLineWidth = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sldLineWidth.setRange(1, 10)
        self.sldLineWidth.setValue(1)
        self.sldLineWidth.setMaximumWidth(260)
        self.sldLineWidth.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.sldLineWidth.valueChanged.connect(self._on_line_width_changed)
        lay.addRow("Line width", self.sldLineWidth)

        # Edges visible
        self.chkEdges = QtWidgets.QCheckBox("Edges visible")
        self.chkEdges.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.chkEdges.toggled.connect(self._on_edges_toggle)
        lay.addRow(self.chkEdges)

        # Edge color button
        self.btnEdgeColor = QtWidgets.QPushButton("Edge color…")
        self.btnEdgeColor.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.btnEdgeColor.clicked.connect(self._on_pick_edge_color)
        lay.addRow(self.btnEdgeColor)

        # Lighting
        self.chkLighting = QtWidgets.QCheckBox("Lighting")
        self.chkLighting.setChecked(True)
        self.chkLighting.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.chkLighting.toggled.connect(self._on_lighting_toggle)
        lay.addRow(self.chkLighting)

        # Time-series group
        self.grpTime = QtWidgets.QGroupBox("Time")
        self.grpTime.setMaximumWidth(300)
        time_lay = QtWidgets.QVBoxLayout(self.grpTime)
        self.sldTime = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sldTime.setRange(0, 0)
        self.sldTime.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.sldTime.valueChanged.connect(self._on_time_changed)
        self.lblTime = QtWidgets.QLabel("—")
        self.lblTime.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.lblTime.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        time_lay.addWidget(self.sldTime)
        time_lay.addWidget(self.lblTime)
        self.grpTime.setVisible(False)
        lay.addRow(self.grpTime)

        # Reset button
        self.btnReset = QtWidgets.QPushButton("Reset defaults")
        self.btnReset.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.btnReset.clicked.connect(self._on_reset_defaults)
        lay.addRow(self.btnReset)

        return w

    # -----------------------------------------------------------------
    # SYNC UI CON DATASET CORRENTE
    # -----------------------------------------------------------------
    def _on_tree_selection_changed(self):
        ds = self._current_dataset_index()
        self._current_ds = ds
        self._rebuild_colorby_combo()
        # carica stato dal mct (se presente)
        entry = self._current_mct()
        if entry:
            self._load_ui_from_entry(entry)

    def _current_dataset_index(self) -> Optional[int]:
        try:
            return self.window._current_dataset_index()
        except Exception:
            return None

    def _current_mct(self) -> Optional[dict]:
        try:
            ds = self._current_dataset_index()
            for e in self.window.mcts.values():
                if e.get("ds_index") == ds:
                    return e
            return self.window.mct if self.window.mct.get("ds_index") == ds else None
        except Exception:
            return None

    def _select_tree_item_for_ds(self, ds_index: int) -> None:
        """Prova a selezionare nel tree il root item con ds=ds_index."""
        t = self.window.treeMCTS
        for i in range(t.topLevelItemCount()):
            root = t.topLevelItem(i)
            data = root.data(0, QtCore.Qt.UserRole)
            if isinstance(data, dict) and data.get("ds") == ds_index:
                t.setCurrentItem(root)
                break

    def _sync_time_slider_visibility(self) -> None:
        has_time = bool(self._time_values) and len(self._time_values) > 1
        self.grpTime.setVisible(has_time)
        if has_time:
            self.sldTime.blockSignals(True)
            self.sldTime.setRange(0, len(self._time_values) - 1)
            self.sldTime.setValue(0)
            self.sldTime.blockSignals(False)
            self.lblTime.setText(f"t = {self._time_values[0]:.6g}")

    # -----------------------------------------------------------------
    # UI EVENTS → APPLY
    # -----------------------------------------------------------------
    def _on_rep_changed(self, mode: str):
        ds = self._current_dataset_index()
        if ds is None: return
        # Mappatura semplice
        self._apply_representation(ds, mode)
        self._save_to_mct("representation", mode)

    def _on_color_by_changed(self):
        ds = self._current_dataset_index()
        if ds is None: return
        label = self.cmbColorBy.currentText()
        vec_mode = self.cmbVectorMode.currentText()
        lut = self.cmbLUT.currentText()
        invert = self.chkInvertLUT.isChecked()
        self._apply_coloring(ds, label, vec_mode, lut, invert)

        self._save_to_mct("color_mode", label)
        self._save_to_mct("colormap", lut)

    def _on_pick_solid_color(self):
        col = QtWidgets.QColorDialog.getColor(parent=self.window, title="Solid Color")
        if not col.isValid(): return
        ds = self._current_dataset_index()
        if ds is None: return
        self._apply_solid_color(ds, (col.red(), col.green(), col.blue()))
        self._save_to_mct("color_mode", "Solid Color")
        self._save_to_mct("solid_color", (col.red(), col.green(), col.blue()))
        # Forza combo su “Solid Color”
        self.cmbColorBy.blockSignals(True)
        self.cmbColorBy.setCurrentText("Solid Color")
        self.cmbColorBy.blockSignals(False)

    def _on_range_auto(self):
        self.editMin.clear(); self.editMax.clear()
        self._on_color_by_changed()  # ri-applica con auto-range

    def _on_rescale_to_data(self):
        # Re-apply colormap chiedendo al viewer di usare data range
        self._on_color_by_changed()

    def _on_scalar_bar_toggle(self, on: bool):
        ds = self._current_dataset_index()
        if ds is None: return
        self._apply_scalar_bar(ds, on)
        self._save_to_mct("scalar_bar", bool(on))

    def _on_opacity_changed(self, val: int):
        ds = self._current_dataset_index()
        if ds is None: return
        self._apply_opacity(ds, val)
        self._save_to_mct("opacity", int(val))

    def _on_point_size_changed(self, val: int):
        ds = self._current_dataset_index()
        if ds is None: return
        self._apply_point_size(ds, val)
        self._save_to_mct("point_size", int(val))

    def _on_line_width_changed(self, val: int):
        ds = self._current_dataset_index()
        if ds is None: return
        self._apply_line_width(ds, val)
        self._save_to_mct("line_width", int(val))

    def _on_edges_toggle(self, on: bool):
        ds = self._current_dataset_index()
        if ds is None: return
        self._apply_edges(ds, bool(on))
        self._save_to_mct("edge_visibility", bool(on))

    def _on_pick_edge_color(self):
        col = QtWidgets.QColorDialog.getColor(parent=self.window, title="Edge Color")
        if not col.isValid(): return
        ds = self._current_dataset_index()
        if ds is None: return
        self._apply_edge_color(ds, (col.red(), col.green(), col.blue()))
        self._save_to_mct("edge_color", (col.red(), col.green(), col.blue()))

    def _on_lighting_toggle(self, on: bool):
        ds = self._current_dataset_index()
        if ds is None: return
        self._apply_lighting(ds, bool(on))
        self._save_to_mct("lighting", bool(on))

    def _on_time_changed(self, idx: int):
        if not self._time_values: return
        self._time_idx = int(idx)
        t = self._time_values[self._time_idx]
        self.lblTime.setText(f"t = {t:.6g}")
        # Rileggi il dataset al tempo selezionato
        # NOTE: servirebbe conservare il reader in self; per semplicità omesso.
        # TODO: estendere per ricaricare dal reader e aggiornare l'attore.

    def _on_reset_defaults(self):
        # Reset UI
        self.cmbRep.setCurrentText("Surface")
        self.cmbColorBy.setCurrentText("Solid Color")
        self.cmbVectorMode.setCurrentText("Magnitude")
        self.cmbLUT.setCurrentText("Viridis")
        self.chkInvertLUT.setChecked(False)
        self.chkScalarBar.setChecked(False)
        self.sldOpacity.setValue(100)
        self.sldPointSize.setValue(3)
        self.sldLineWidth.setValue(1)
        self.chkEdges.setChecked(False)
        self.chkLighting.setChecked(True)
        # Applica allo stato corrente
        self._on_rep_changed("Surface")
        self._on_color_by_changed()
        self._on_scalar_bar_toggle(False)
        self._on_opacity_changed(100)
        self._on_point_size_changed(3)
        self._on_line_width_changed(1)
        self._on_edges_toggle(False)
        self._on_lighting_toggle(True)

    # -----------------------------------------------------------------
    # APPLY (adapter verso le API del viewer)
    # -----------------------------------------------------------------
    def _apply_representation(self, ds: int, mode: str):
        """
        Mappa delle rappresentazioni:
        - Points, Wireframe, Surface, Surface with Edges, Volume
        """
        # Se il tuo Viewer3D espone un metodo diretto:
        fn = getattr(self.window.viewer3d, "set_mesh_representation", None)
        if callable(fn):
            fn(ds, mode)
            return
        # TODO: fallback diretto su actor (non necessario se hai l’API)

    def _apply_solid_color(self, ds: int, rgb: Tuple[int, int, int]):
        fn = getattr(self.window.viewer3d, "set_dataset_color", None)
        if callable(fn):
            fn(ds, *rgb)
            return
        # TODO: fallback

    def _apply_coloring(self, ds: int, label: str, vec_mode: str, lut: str, invert: bool):
        """
        label = "Solid Color" oppure "PointData/<array>" o "CellData/<array>"
        vec_mode = Magnitude / X / Y / Z
        """
        # Caso Solid Color → forza colore uniforme
        if label == "Solid Color":
            self._apply_solid_color(ds, self._current_mct().get("solid_color", (255, 255, 255)))
            # Se il viewer ha un “color mode”, impostalo
            try:
                self.window.viewer3d.set_color_mode("Solid Color", ds)
            except Exception:
                pass
            return

        # Parsing “PointData/NAME” o “CellData/NAME”
        assoc = "POINT"
        array_name = label
        if label.startswith("PointData/"):
            assoc = "POINT"
            array_name = label.split("/", 1)[1]
        elif label.startswith("CellData/"):
            assoc = "CELL"
            array_name = label.split("/", 1)[1]

        # Viewer API personalizzata (se esiste):
        # Immaginiamo una API del tipo: set_scalar_coloring(ds, array_name, assoc, component, lut, invert, range)
        fn = getattr(self.window.viewer3d, "set_scalar_coloring", None)
        rng = self._manual_range_or_none()
        component = {"Magnitude": None, "X": 0, "Y": 1, "Z": 2}.get(vec_mode, None)
        if callable(fn):
            fn(ds, array_name, assoc, component, lut, bool(invert), rng)
            return

        # TODO: fallback generico su actor

    def _manual_range_or_none(self) -> Optional[Tuple[float, float]]:
        try:
            smin = self.editMin.text().strip()
            smax = self.editMax.text().strip()
            if not smin or not smax:
                return None
            return (float(smin), float(smax))
        except Exception:
            return None

    def _apply_scalar_bar(self, ds: int, show: bool):
        fn = getattr(self.window.viewer3d, "set_scalar_bar_visible", None)
        if callable(fn):
            fn(ds, bool(show))
            return
        # TODO: fallback

    def _apply_opacity(self, ds: int, val: int):
        fn = getattr(self.window.viewer3d, "set_mesh_opacity", None)
        if callable(fn):
            fn(ds, int(val))
            return
        # TODO: fallback

    def _apply_point_size(self, ds: int, val: int):
        fn = getattr(self.window.viewer3d, "set_point_size", None)
        if callable(fn):
            fn(int(val), ds)
            return
        # TODO: fallback

    def _apply_line_width(self, ds: int, val: int):
        fn = getattr(self.window.viewer3d, "set_line_width", None)
        if callable(fn):
            fn(ds, int(val))
            return
        # TODO: fallback

    def _apply_edges(self, ds: int, on: bool):
        fn = getattr(self.window.viewer3d, "set_edge_visibility", None)
        if callable(fn):
            fn(ds, bool(on))
            return
        # TODO: fallback

    def _apply_edge_color(self, ds: int, rgb: Tuple[int, int, int]):
        fn = getattr(self.window.viewer3d, "set_edge_color", None)
        if callable(fn):
            fn(ds, *rgb)
            return
        # TODO: fallback

    def _apply_lighting(self, ds: int, on: bool):
        fn = getattr(self.window.viewer3d, "set_lighting_enabled", None)
        if callable(fn):
            fn(ds, bool(on))
            return
        # TODO: fallback

    # -----------------------------------------------------------------
    # POPOLAMENTO COMBO E SYNC UI
    # -----------------------------------------------------------------
    def _rebuild_colorby_combo(self):
        """Legge le arrays dal PolyData/mesh corrente e popola la combo “Color by”."""
        self.cmbColorBy.blockSignals(True)
        self.cmbColorBy.clear()
        self.cmbColorBy.addItem("Solid Color")
        ds = self._current_dataset_index()
        if ds is None:
            self.cmbColorBy.blockSignals(False)
            return

        # Recupera pdata dal viewer
        arrays_pt, arrays_cell = [], []
        try:
            recs = getattr(self.window.viewer3d, "_datasets", [])
            rec = recs[ds]
            pdata = rec.get("pdata") or rec.get("full_pdata") or rec.get("mesh")
            if pdata is not None:
                # PointData
                try:
                    for name in list(pdata.point_data.keys()):
                        if str(name).startswith("vtkOriginal"):  # nascondi array tecniche
                            continue
                        arrays_pt.append(str(name))
                except Exception:
                    pass
                # CellData
                try:
                    for name in list(pdata.cell_data.keys()):
                        if str(name).startswith("vtkOriginal"):
                            continue
                        arrays_cell.append(str(name))
                except Exception:
                    pass
        except Exception:
            pass

        if arrays_pt:
            for n in arrays_pt:
                self.cmbColorBy.addItem(f"PointData/{n}")
        if arrays_cell:
            for n in arrays_cell:
                self.cmbColorBy.addItem(f"CellData/{n}")

        self.cmbColorBy.blockSignals(False)

    def _load_ui_from_entry(self, e: dict) -> None:
        """Carica (best effort) i controlli dai valori persistiti nel mct entry."""
        try: self.cmbRep.setCurrentText(e.get("representation", "Surface"))
        except Exception: pass
        try: self.cmbColorBy.setCurrentText(e.get("color_mode", "Solid Color"))
        except Exception: pass
        try: self.cmbLUT.setCurrentText(e.get("colormap", "Viridis"))
        except Exception: pass
        try: self.chkScalarBar.setChecked(bool(e.get("scalar_bar", False)))
        except Exception: pass
        try: self.sldOpacity.setValue(int(e.get("opacity", 100)))
        except Exception: pass
        try: self.sldPointSize.setValue(int(e.get("point_size", 3)))
        except Exception: pass
        try: self.sldLineWidth.setValue(int(e.get("line_width", 1)))
        except Exception: pass
        try: self.chkEdges.setChecked(bool(e.get("edge_visibility", False)))
        except Exception: pass
        try: self.chkLighting.setChecked(bool(e.get("lighting", True)))
        except Exception: pass

    def _save_to_mct(self, key: str, val):
        e = self._current_mct()
        if e is not None:
            e[key] = val

    # -----------------------------------------------------------------
    # LOGGING/UTILITY
    # -----------------------------------------------------------------
    def _msg(self, text: str, error: bool = False):
        # Scrive nella statusbar + pannello messaggi (se disponibile)
        try:
            self.window.statusBar().showMessage(text, 5000)
        except Exception:
            pass
        try:
            self.window.txtMessages.appendPlainText(text)
        except Exception:
            pass
        if error:
            print(text)

# ---------------------------------------------------------------------
# ENTRY POINT per PluginManager
# ---------------------------------------------------------------------
def register(window) -> object:
    """
    Factory entry-point richiesto da `entry_point: "plugin:register"`.
    Ritorna l'istanza del plugin.
    """
    return VTKImportPlugin(window)