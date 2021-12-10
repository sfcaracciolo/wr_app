
import typing
from PySide6.QtWidgets import QApplication, QDataWidgetMapper, QDockWidget, QGridLayout, QHBoxLayout, QLabel, QListView, QMainWindow, QSlider, QSplitter, QTableView, QVBoxLayout, QWidget, QHeaderView
from PySide6.QtCore import QAbstractListModel, QAbstractTableModel, QModelIndex, QPersistentModelIndex, Qt, Signal
from superqt import QLabeledSlider
import sys
import numpy as np
from ecgsyn_wr import utils
import dill
import vispy as vp
from vispy import scene
import os 

dill.settings['recurse'] = True
inputs = np.zeros(9, dtype=np.float64)
params = np.zeros((4, 3), dtype=np.float64)
fiducials = np.zeros(13, dtype=np.float32)
RED = np.array([1., 0, 0, .5], dtype=np.float32)
RR_MAX = int(300.)
Y_RANGE = (-1000, 1000)

class FiducialsModel(QAbstractTableModel):
    drawMarkers = Signal(np.ndarray, np.ndarray)

    def __init__(self, data: np.ndarray, parent=None) -> None:
        super().__init__(parent=parent)
        self._data = data
        self._names = ['Pon', 'P', 'Pend', 'QRSon', 'Q', 'R', 'S', 'QRSend','res','Ton', 'T', 'Tend', 'res']
        self.T_end = dill.load(open("tend", "rb"))

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

        if role == Qt.TextAlignmentRole:
            return Qt.AlignCenter

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
    computeSignal = Signal(np.ndarray)

    def __init__(self, data: np.ndarray, parent=None) -> None:
        super().__init__(parent=parent)
        self._data = data
        self._flatten_data = np.ravel(self._data, order='C')        

        self._hheader = ['a', 'mu', 'sigma']
        self._vheader = ['P', 'R', 'S', 'T']

        self.T_gauss = utils.transform_matrix()
        self.T_gumbel = dill.load(open("coeffs", "rb"))

    def rowCount(self, parent=None) -> int:
        return self._data.shape[0]

    def columnCount(self, parent=None) -> int:
        return self._data.shape[1]

    def data(self, index: QModelIndex, role: int) -> typing.Any:
        if role == Qt.DisplayRole:
            i, j = index.row(), index.column()
            value = self._data[i, j]
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
        
class InputsPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self.setup_ui()

    def setup_ui(self):
        self.setMinimumWidth(500)
        model = self.parent().inputs_model
        
        steps = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        limits = [
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
        self.default_values = [250., 22., 50., 16., 80., 70., 600., -300., 120.,]

        self.mapper = QDataWidgetMapper()
        self.mapper.setSubmitPolicy(QDataWidgetMapper.SubmitPolicy.ManualSubmit)
        self.mapper.setModel(model)
        
        grid = QGridLayout()
        for i, (name, step, limit, value) in enumerate(zip(model._names, steps, limits, self.default_values)):
            grid.addWidget(QLabel(name), i, 0)
            slider = QLabeledSlider(Qt.Horizontal)
            slider.setRange(*limit)
            # slider.setTickPosition(QSlider.TickPosition.TicksAbove)
            # slider.setSingleStep(step)
            slider.setValue(value)
            grid.addWidget(slider, i, 1)
            self.mapper.addMapping(slider, i)
            slider.valueChanged.connect(lambda v, i=i: model.updateInputs(i, v))

        self.mapper.toFirst()

        # inputsView = TablePanel(model)
        layout = QVBoxLayout()
        layout.addLayout(grid)
        # layout.addWidget(inputsView)

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

class ecgsyn(QMainWindow):
    def __init__(self, app):
        super().__init__()
        self.app = app

        self.load_files()
        self.params_model = ParamsModel(params)
        self.inputs_model = InputsModel(inputs)
        self.fiducials_model = FiducialsModel(fiducials)

        self.setup_ui()

        self.inputs_model.computeParams.connect(self.params_model.updateParams)
        self.params_model.computeSignal.connect(self.update)
        self.fiducials_model.drawMarkers.connect(self.beatViewer.set_markers)

        self.inputs_model.setDefaultInputs(self.inputsPanel.default_values)

    def update(self, params):
        self.beatViewer.set_line(params)
        self.fiducials_model.updateFiducials(params)

    def load_files(self):
        if not( os.path.exists('coeffs') and os.path.exists('tend')):
            print('wait 100 seconds aprox ... only for the first opening.')
            T_gumbel, T_end = utils.poly_coeffs()
            dill.dump(T_gumbel, open("coeffs", "wb"))
            dill.dump(T_end, open("tend", "wb"))

    def setup_ui(self):
        self.setWindowTitle('ecgsyn')

        self.inputsPanel = InputsPanel(parent=self)
        paramsTable = TablePanel(self.params_model, parent=self)
        fiducialsTable = TablePanel(self.fiducials_model, parent=self)
        self.beatViewer = BeatViewer(parent=self)

        vsplitter = QSplitter(Qt.Vertical)
        vsplitter.addWidget(self.inputsPanel)
        vsplitter.setStretchFactor(0, 3)
        vsplitter.addWidget(fiducialsTable)
        vsplitter.setStretchFactor(1, 1)
        vsplitter.addWidget(paramsTable)
        vsplitter.setStretchFactor(2, 1)

        hsplitter = QSplitter(Qt.Horizontal)
        hsplitter.addWidget(vsplitter)
        hsplitter.addWidget(self.beatViewer)

        self.setCentralWidget(hsplitter)

        self.show()

def run():
    app = QApplication(sys.argv)
    viewer = ecgsyn(app)
    sys.exit(app.exec())