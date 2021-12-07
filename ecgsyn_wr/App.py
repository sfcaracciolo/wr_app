
import typing
from PySide6.QtWidgets import QApplication, QDataWidgetMapper, QDockWidget, QGridLayout, QHBoxLayout, QLabel, QListView, QMainWindow, QSlider, QSplitter, QTableView, QVBoxLayout, QWidget
from PySide6.QtCore import QAbstractListModel, QAbstractTableModel, QModelIndex, QPersistentModelIndex, Qt, Signal
from superqt import QLabeledSlider
import sys
import numpy as np
from ecgsyn_wr import utils
import dill
import vispy as vp
from vispy import scene

dill.settings['recurse'] = True
inputs = np.zeros(9, dtype=np.float64)
params = np.zeros((4, 3), dtype=np.float64)
N_MAX = 1000
class InputsModel(QAbstractTableModel):
    computeParams = Signal(np.ndarray)

    def __init__(self, data: np.ndarray, parent=None) -> None:
        super().__init__(parent=parent)
        self._data = data
        self._names = ['RR', 'P duration', 'PR', 'QRS', 'QT', 'P peak', 'R peak', 'S peak', 'T peak']
        
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
            value = self._data[j]
            return float(value)

    # def flags(self, index: typing.Union[QModelIndex, QPersistentModelIndex]) -> Qt.ItemFlags:
    #     return Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable
          
    def updateInputs(self, i, v):
        self._data[i] = v
        index = self.index(1, i)
        self.dataChanged.emit(index, index)
        self.computeParams.emit(self._data)

    def setDefaultInputs(self, values):
        self._data[:] = np.array(values)
        self.layoutChanged.emit()
        self.computeParams.emit(self._data)

class ParamsModel(QAbstractTableModel):
    computeModel = Signal(np.ndarray)

    def __init__(self, data: np.ndarray, parent=None) -> None:
        super().__init__(parent=parent)
        self._data = data
        self._hheader = ['a', 'mu', 'sigma']
        self._vheader = ['P', 'R', 'S', 'T']

        self.T_gauss = utils.transform_matrix()

        try:
            self.T_gumbel = dill.load(open("coeffs", "rb"))
        except FileNotFoundError:
            self.T_gumbel = utils.poly_coeffs()
            dill.dump(self.T_gumbel, open("coeffs", "wb"))

    def rowCount(self, parent=None) -> int:
        return self._data.shape[0]

    def columnCount(self, parent=None) -> int:
        return self._data.shape[1]

    def data(self, index: QModelIndex, role: int) -> typing.Any:
        if role == Qt.DisplayRole:
            i, j = index.row(), index.column()
            value = self._data[i, j]
            return float(value)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int) -> typing.Any:
        if role == Qt.DisplayRole:
            return self._vheader[section] if orientation == Qt.Vertical else self._hheader[section]
        return super().headerData(section, orientation, role=role)

    def updateParams(self, inputs):
        
        RR = inputs[0]
        P = inputs[1]
        PR = inputs[2]
        QRS = inputs[3]
        QT = inputs[4]
        Ppeak = inputs[5]
        Rpeak = inputs[6]
        Speak = inputs[7]
        Tpeak = inputs[8]

        # PQRS part
        p = utils.temporal_gaussian_params(RR, PR, P, QRS, fun=self.T_gauss)
        self._data[:3, 1] = p[:3]
        self._data[:3, 2] = p[3:6]

        # T part
        QRS_on = self._data[1, 1] - 3*self._data[1, 2]
        p = utils.temporal_gumbel_params(QT, QRS, QRS_on, fun=self.T_gumbel)
        self._data[3, 1:] = p

        # amplitudes
        params = np.ravel(self._data, order='C')
        self._data[:, 0] = np.array([Ppeak, Rpeak, Speak, Tpeak])
        F = lambda x: utils.nonlinear_system(x, params=params)
        p = utils.amplitude_params(Ppeak, Rpeak, Speak, Tpeak, fun=F)
        self._data[:, 0] = p

        self.layoutChanged.emit()
        self.computeModel.emit(params)

class ParamsPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self.setup_ui()

    def setup_ui(self):
        self.setMinimumWidth(500)
        table = QTableView()
        table.setModel(self.parent().params_model)
        layout = QVBoxLayout()
        layout.addWidget(table)
        self.setLayout(layout)
        
class InputsPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self.setup_ui()

    def setup_ui(self):
        self.setMinimumWidth(500)
        model = self.parent().inputs_model
        
        steps = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        limits = [
            (200., 300.), 
            (20., 30.), 
            (40., 70.),
            (10., 40.),
            (60., 120.),
            (-250., 250.), 
            (300., 1000.),
            (-600., -100.),
            (80., 150.)
        ]
        default_values = [250., 22., 50., 16., 80., 70., 600., -300., 120.,]

        self.mapper = QDataWidgetMapper()
        self.mapper.setSubmitPolicy(QDataWidgetMapper.SubmitPolicy.ManualSubmit)
        self.mapper.setModel(model)
        
        grid = QGridLayout()
        for i, (name, step, limit, value) in enumerate(zip(model._names, steps, limits, default_values)):
            grid.addWidget(QLabel(name), i, 0)
            slider = QLabeledSlider(Qt.Horizontal)
            slider.setRange(*limit)
            slider.setTickPosition(QSlider.TickPosition.TicksAbove)
            slider.setSingleStep(step)
            slider.setValue(value)
            grid.addWidget(slider, i, 1)
            self.mapper.addMapping(slider, i)
            slider.valueChanged.connect(lambda v, i=i: model.updateInputs(i, v))

        self.mapper.toFirst()
        inputsView = QTableView()
        inputsView.setModel(model)
        inputsView.resizeColumnsToContents()
        model.setDefaultInputs(default_values)

        layout = QVBoxLayout()
        layout.addLayout(grid)
        layout.addWidget(inputsView)

        self.setLayout(layout)

class BeatViewer(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        
        self._data = np.empty((N_MAX, 2), dtype = np.float64)
        self._data[:,0] = np.arange(N_MAX)

        self.setup_ui()

    def setup_ui(self):
        self.canvas = vp.scene.SceneCanvas(show=True, bgcolor='white', parent=self)
        view = self.canvas.central_widget.add_view(camera='panzoom')
        view.camera.interactive = True

        self.line = vp.scene.Line(
            pos = self._data,
            color = 'black',
            parent=view.scene
        )

        view.camera.set_range(
            x = (0, N_MAX),
            y = (-1000, 1000)
        )

        layout = QVBoxLayout()
        layout.addWidget(self.canvas.native)
        self.setLayout(layout)

    def set_data(self, params):
        t = self._data[:,0]
        self._data[:,1] = utils.model(t, *params.tolist())
        self.line.set_data(pos=self._data)

class ecgsyn(QMainWindow):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.params_model = ParamsModel(params)
        self.inputs_model = InputsModel(inputs)
        self.setup_ui()

        self.inputs_model.computeParams.connect(self.params_model.updateParams)
        self.params_model.computeModel.connect(self.beatViewer.set_data)


    def setup_ui(self):
        self.setWindowTitle('ecgsyn')

        inputsPanel = InputsPanel(parent=self)
        paramsPanel = ParamsPanel(parent=self)
        self.beatViewer = BeatViewer(parent=self)

        vsplitter = QSplitter(Qt.Vertical)
        vsplitter.addWidget(inputsPanel)
        vsplitter.addWidget(paramsPanel)

        hsplitter = QSplitter(Qt.Horizontal)
        hsplitter.addWidget(vsplitter)
        hsplitter.addWidget(self.beatViewer)

        self.setCentralWidget(hsplitter)

        self.show()

def run():
    app = QApplication(sys.argv)
    viewer = ecgsyn(app)
    sys.exit(app.exec())