
import typing
from PySide6.QtWidgets import QApplication, QButtonGroup, QCheckBox, QDataWidgetMapper, QDialogButtonBox, QDockWidget, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QListView, QMainWindow, QPushButton, QSlider, QSpinBox, QSplitter, QTableView, QVBoxLayout, QWidget, QHeaderView
from PySide6.QtCore import QAbstractListModel, QAbstractTableModel, QModelIndex, QPersistentModelIndex, Qt, Signal
from vispy.scene import widgets
from superqt import QLabeledSlider
import sys
import numpy as np
from ecgsyn_wr import utils
import vispy as vp
from vispy import scene
import os 
import colorednoise as cn

inputs = np.zeros((2, 9), dtype=np.float64)
params = np.zeros((4, 3), dtype=np.float64)
fiducials = np.zeros(13, dtype=np.float32)
RED = np.array([1., 0, 0, .5], dtype=np.float32)
RR_MAX = int(300.)
Y_RANGE = (-1000, 1000)
N_MAX = 10000

class FiducialsModel(QAbstractTableModel):
    drawMarkers = Signal(np.ndarray, np.ndarray)

    def __init__(self, data: np.ndarray, parent=None) -> None:
        super().__init__(parent=parent)
        self._data = data
        self._names = ['Pon', 'P', 'Pend', 'QRSon', 'Q', 'R', 'S', 'QRSend','res','Ton', 'T', 'Tend', 'res']
        _, self.T_end = utils.build_transforms()

    def rowCount(self, parent=None) -> int:
        return 1
    
    def columnCount(self, parent = None) -> int:
        return self._data.size

    def headerData(self, section: int, orientation: Qt.Orientation, role: int) -> typing.Any:
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self._names[section]
        return super().headerData(section, orientation, role=role)

    def data(self, index: QModelIndex, role: int) -> typing.Any:
        if role == Qt.DisplayRole:
            j = index.column()
            value = np.round(self._data[j], decimals=2)
            return float(value)
        
        if role == Qt.TextAlignmentRole:
            return Qt.AlignCenter
    
    def updateFiducials(self, params):
        self._data[:] = utils.params2fiducials(params, fun=self.T_end)
        self.layoutChanged.emit()

        subset_fiducials = np.delete(self._data, [4, 8, 9, 12])
        self.drawMarkers.emit(subset_fiducials, params)

class InputsModel(QAbstractTableModel):
    computeParams = Signal(np.ndarray)

    def __init__(self, data: np.ndarray, parent=None) -> None:
        super().__init__(parent=parent)
        self._data = data
        self._hheader = ['RR interval', 'P duration', 'PR interval', 'QRS duration', 'QT inrterval', 'P peak', 'R peak', 'S peak', 'T peak']
        self._vheader = ['mu', 'sigma']

    def rowCount(self, parent=None) -> int:
        return 2
    
    def columnCount(self, parent = None) -> int:
        return self._data.shape[1]


    def headerData(self, section: int, orientation: Qt.Orientation, role: int) -> typing.Any:
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self._hheader[section]
            else:
                return self._vheader[section]

        return super().headerData(section, orientation, role=role)

    def data(self, index: QModelIndex, role: int) -> typing.Any:
        if role == Qt.DisplayRole:
            i = index.row()
            j = index.column()
            value = self._data[i, j]
            return float(value)

        if role == Qt.TextAlignmentRole:
            return Qt.AlignCenter

    # def flags(self, index: typing.Union[QModelIndex, QPersistentModelIndex]) -> Qt.ItemFlags:
    #     return Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable
          
    def updateInputs(self, i, j, v):
        self._data[i, j] = v
        index = self.index(i, j)
        self.dataChanged.emit(index, index)
        if i == 0:
            self.computeParams.emit(self._data[0,:])

    def setDefaultInputs(self, values):
        self._data[0,:] = np.array(values)
        self.layoutChanged.emit()
        self.computeParams.emit(self._data[0,:])

class ParamsModel(QAbstractTableModel):
    computeSignal = Signal(np.ndarray)

    def __init__(self, data: np.ndarray, parent=None) -> None:
        super().__init__(parent=parent)
        self._data = data
        self._flatten_data = np.ravel(self._data, order='C')        

        self._hheader = ['a', 'mu', 'sigma']
        self._vheader = ['P', 'R', 'S', 'T']

        self.T_gauss = utils.transform_matrix()
        self.T_gumbel, _ = utils.build_transforms()

    def rowCount(self, parent=None) -> int:
        return self._data.shape[0]

    def columnCount(self, parent=None) -> int:
        return self._data.shape[1]

    def data(self, index: QModelIndex, role: int) -> typing.Any:
        if role == Qt.DisplayRole:
            i, j = index.row(), index.column()
            value = np.round(self._data[i,j], decimals=2)
            return float(value)

        if role == Qt.TextAlignmentRole:
            return Qt.AlignCenter

    def headerData(self, section: int, orientation: Qt.Orientation, role: int) -> typing.Any:
        if role == Qt.DisplayRole:
            return self._vheader[section] if orientation == Qt.Vertical else self._hheader[section]
        return super().headerData(section, orientation, role=role)

    def updateParams(self, inputs):
        self._data[:] = utils.inputs2params(inputs, transforms=[self.T_gauss, self.T_gumbel])
        self.layoutChanged.emit()
        self.computeSignal.emit(self._flatten_data)

class TablePanel(QWidget):
    def __init__(self, model, parent=None) -> None:
        super().__init__(parent=parent)
        self.setMinimumWidth(500)
        table = QTableView()
        table.setModel(model)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout = QVBoxLayout()
        layout.addWidget(table)
        self.setLayout(layout)

class RunPanel(QWidget):
    drawEcg = Signal(np.ndarray)

    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)

        self.model = parent.inputs_model
        self.setup_ui()
        self.buttons.button(QDialogButtonBox.Apply).clicked.connect(self.run)

    def setup_ui(self):


        self.n_beats = QSpinBox()
        self.n_beats.setRange(10,1000)
        self.n_beats.setValue(10)

        form_layout = QFormLayout()
        form_layout.addRow('# beats', self.n_beats)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Apply) #  | QDialogButtonBox.Cancel)
        
        layout = QVBoxLayout()
        layout.addLayout(form_layout)
        layout.addWidget(self.buttons)

        layout.addStretch(1)
        self.setLayout(layout)

    def run(self, e):
        
        mean_inputs = self.model._data[0,:]
        sd_inputs = self.model._data[1,:].copy()
        sd_inputs *= np.abs(mean_inputs)
        sd_inputs /= 100

        t, v = utils.ecgsyn_wr(
            self.n_beats.value(),
            mean_inputs,
            sd_inputs,
            remove_drift=True
        )

        ecg = np.empty((t.size, 2), dtype=np.float32)
        ecg[:, 0] = t
        ecg[:, 1] = v[2,:]

        self.drawEcg.emit(ecg)

class NoisePanel(QGroupBox):
    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
    
        self.setup_ui()

    def setup_ui(self):

        self.setTitle('Additive noise')
        self.setCheckable(True)
        self.setChecked(False)

        self.beta = QLabeledSlider(Qt.Horizontal)
        self.beta.setRange(0, 5)
        self.beta.setValue(0)

        self.snr = QLabeledSlider(Qt.Horizontal)
        self.snr.setRange(-5, 20)
        self.snr.setValue(1)

        layout = QFormLayout()
        layout.addRow('β', self.beta)
        layout.addRow('SNR [db]', self.snr)

        self.setLayout(layout)

    def add_noise(self, signal):
        if self.isChecked():
            beta = self.beta.value()
            noise = cn.powerlaw_psd_gaussian(beta, signal.size)

            power_s = np.mean(signal*signal)
            snr = self.snr.value()
            power_n = power_s * np.power(10, -snr/10)

            noise *= np.sqrt(power_n)
            signal += noise

class InputsPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self.setup_ui()

    def setup_ui(self):
        self.setMinimumWidth(500)
        model = self.parent().inputs_model
        
        steps = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        median_limits = [
            (200., RR_MAX), 
            (20., 30.), 
            (40., 70.),
            (10., 40.),
            (60., 120.),
            (-250., 250.), 
            (300., 1000.),
            (-600., -100.),
            (80., 300.)
        ]
        self.default_median_values = [250., 22., 50., 16., 80., 70., 600., -300., 120.,]

        self.mapper = QDataWidgetMapper()
        self.mapper.setSubmitPolicy(QDataWidgetMapper.SubmitPolicy.ManualSubmit)
        self.mapper.setModel(model)
        
        grid = QGridLayout()
        for j, (name, step, limit, value) in enumerate(zip(model._hheader, steps, median_limits, self.default_median_values)):
            label = QLabel(name)
            label.setAlignment(Qt.AlignRight | Qt.AlignCenter)
            grid.addWidget(label, j, 0)

            slider = QLabeledSlider(Qt.Horizontal)
            slider.setRange(*limit)
            slider.setValue(value)
            grid.addWidget(slider, j, 1)
            self.mapper.addMapping(slider, j)
            slider.valueChanged.connect(lambda v, i=0, j=j: model.updateInputs(i, j, v))

            slider = QLabeledSlider(Qt.Horizontal)
            slider.setRange(0, 20)
            grid.addWidget(slider, j, 2)
            self.mapper.addMapping(slider, j)
            slider.valueChanged.connect(lambda v, i=1, j=j: model.updateInputs(i, j, v))

        self.mapper.toFirst()

        layout = QVBoxLayout()
        layout.addLayout(grid)

        self.setLayout(layout)

class BeatViewer(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        
        self._data = np.empty((2*RR_MAX, 2), dtype = np.float32)
        self._data[:,0] = np.arange(0, 2*RR_MAX)

        self._markers = np.zeros((9, 2), dtype=np.float32)

        self.setup_ui()

    def setup_ui(self):
        self.canvas = vp.scene.SceneCanvas(show=True, bgcolor='white', parent=self)
        self.grid = self.canvas.central_widget.add_grid()

        self.vb1 = self.grid.add_view(row=0, col=0, camera='panzoom')
        self.vb1.camera.interactive = False

        self.line = vp.scene.Line(
            pos = self._data,
            color = 'black',
            parent=self.vb1.scene
        )

        # self.vb2 = self.grid.add_view(row=1, col=0, camera='panzoom')
        # self.vb2.camera.interactive = False

        self.markers = vp.scene.Markers(
            pos = self._markers,
            parent=self.vb1.scene
        )

        layout = QVBoxLayout()
        layout.addWidget(self.canvas.native)
        self.setLayout(layout)

    def set_line(self, params):
        self.vb1.camera.set_range(
            x=(0, 2*params[4]),
            y = Y_RANGE
        )
        self._data[:,1] = utils.model(self._data[:,0], *params.tolist())
        self.line.set_data(pos=self._data)

    def set_markers(self, fiducials, params):
        self._markers[:, 0] = fiducials
        self._markers[:, 1] = utils.model(fiducials, *params.tolist())

        self.markers.set_data(
            self._markers,
            size = 10,
            edge_width = 0,
            edge_color = RED,
            face_color = RED,
        )
        self.markers.update()

class EcgViewer(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        
        self.setup_ui()

    def setup_ui(self):
        self.canvas = vp.scene.SceneCanvas(show=True, bgcolor='white', parent=self)
        self.grid = self.canvas.central_widget.add_grid()

        self.vb1 = self.grid.add_view(row=0, col=0, camera='panzoom')
        self.vb1.camera.interactive = True

        self.line = vp.scene.Line(
            color = 'black',
            parent=self.vb1.scene
        )

        layout = QVBoxLayout()
        layout.addWidget(self.canvas.native)
        self.setLayout(layout)

    def set_line(self, data):
        self.x_max = data.shape[0]
        self.y_min, self.y_max = data[:,1].min(), data[:,1].max()
        self.y_min *= 1.2
        self.y_max *= 1.2

        self.line.set_data(pos=data)

        self.autospan()

    def autospan(self):
        self.vb1.camera.set_range(
            x = (0, self.x_max),
            y = (self.y_min , self.y_max),
            margin = 0.
        )

class ecgsyn(QMainWindow):
    def __init__(self, app):
        super().__init__()
        self.app = app

        self.params_model = ParamsModel(params)
        self.inputs_model = InputsModel(inputs)
        self.fiducials_model = FiducialsModel(fiducials)

        self.setup_ui()

        self.inputs_model.computeParams.connect(self.params_model.updateParams)
        self.params_model.computeSignal.connect(self.update)
        self.fiducials_model.drawMarkers.connect(self.beatViewer.set_markers)
        self.runPanel.drawEcg.connect(self.plot)

        self.inputs_model.setDefaultInputs(self.inputsPanel.default_median_values)

    def update(self, params):
        self.beatViewer.set_line(params)
        self.fiducials_model.updateFiducials(params)

    def plot(self, ecg):
        self.noisePanel.add_noise(ecg[:,1])
        self.ecgViewer.set_line(ecg)

    def setup_ui(self):
        self.setWindowTitle('ecgsyn')

        self.inputsPanel = InputsPanel(parent=self)
        self.noisePanel = NoisePanel(parent=self)
        self.runPanel = RunPanel(parent=self)
        # inputsView = TablePanel(self.inputs_model, parent=self)
        # paramsTable = TablePanel(self.params_model, parent=self)
        # fiducialsTable = TablePanel(self.fiducials_model, parent=self)
        self.beatViewer = BeatViewer(parent=self)
        self.ecgViewer = EcgViewer(parent=self)

        # vertical splitter
        vsplitter = QSplitter(Qt.Vertical)

        vsplitter.addWidget(self.inputsPanel)
        vsplitter.setStretchFactor(0, 3)

        # vsplitter.addWidget(inputsView)
        # vsplitter.setStretchFactor(1, 1)

        # vsplitter.addWidget(fiducialsTable)
        # vsplitter.setStretchFactor(2, 1)

        # vsplitter.addWidget(paramsTable)
        # vsplitter.setStretchFactor(3, 1)

        vsplitter.addWidget(self.noisePanel)
        vsplitter.setStretchFactor(1, 1)

        vsplitter.addWidget(self.runPanel)
        vsplitter.setStretchFactor(2, 1)

        # horizontal
        hsplitter = QSplitter(Qt.Horizontal)

        hsplitter.addWidget(vsplitter)
        hsplitter.addWidget(self.beatViewer)
        
        msplitter = QSplitter(Qt.Vertical)
        msplitter.addWidget(hsplitter)
        msplitter.addWidget(self.ecgViewer)

        self.setCentralWidget(msplitter)

        self.show()

def run():
    app = QApplication(sys.argv)
    viewer = ecgsyn(app)
    sys.exit(app.exec())