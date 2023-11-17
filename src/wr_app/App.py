from PySide6.QtWidgets import *
from PySide6.QtCore import *
from typing import Tuple, Any
import sys
from vispy import scene
import numpy as np 
import scipy as sp 
import superqt 
import wr_transform
import ecg_models 
import ecg_simulator

RED = np.array([1., 0, 0, .5], dtype=np.float32)

class BaseViewer(QWidget):

    def __init__(self, n_markers: int = None, interactive: bool = False, **kwargs ) -> None:
        
        super().__init__(**kwargs) 

        axis_kwargs = dict(
            axis_color='black',
            tick_color='black',
            text_color='black',
            tick_font_size=6,
            tick_label_margin=18,
            axis_width=1,
            tick_width=1
        )

        if n_markers is not None:
            self._markers = np.zeros((n_markers, 2), dtype=np.float32)

        self.canvas = scene.SceneCanvas(show=True, bgcolor='white', parent=self)
        self.grid = self.canvas.central_widget.add_grid()

        top_padding = self.grid.add_widget(row=0, col=1, row_span=1)
        top_padding.height_max = 20

        right_padding = self.grid.add_widget(row=1, col=2, col_span=1)
        right_padding.width_max = 20

        self.vb1 = self.grid.add_view(row=1, col=1, camera='panzoom')
        self.vb1.camera.interactive = interactive

        self.line = scene.Line(
            color = 'black',
            parent=self.vb1.scene
        )

        if n_markers is not None:
            self.markers = scene.Markers(
                pos = self._markers,
                parent=self.vb1.scene
            )

        yaxis = scene.AxisWidget(
            orientation='left',
            **axis_kwargs
        )
        yaxis.width_max = 50
        self.grid.add_widget(yaxis, row=1, col=0)
        yaxis.link_view(self.vb1)

        xaxis = scene.AxisWidget(
            orientation='bottom',
            **axis_kwargs
        )
        xaxis.height_max = 50
        self.grid.add_widget(xaxis, row=2, col=1)
        xaxis.link_view(self.vb1)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas.native)
        self.setLayout(layout)

    def set_range(self, x_lims: Tuple = None, y_lims: Tuple = None):
        self.vb1.camera.set_range(
            x=x_lims,
            y=y_lims
        )

    def set_line(self, x: np.ndarray = None, y: np.ndarray = None):
        if x is not None: self.set_x(x)
        if y is not None: self.set_y(y)
        self.line.set_data(pos=self._data)

    def set_markers(self, x: np.ndarray = None, y: np.ndarray = None):
        if x is not None: self._markers[:,0] = x
        if y is not None: self._markers[:,1] = y
        self.markers.set_data(
            self._markers,
            size = 10,
            edge_width = 0,
            edge_color = RED,
            face_color = RED,
        )
        self.markers.update()

    def set_x(self, values: np.ndarray):
        self._data[:,0] = values

    def get_x(self):
        return self._data[:,0]

    def get_y(self):
        return self._data[:,1]
    
    def set_y(self, values: np.ndarray):
        self._data[:,1] = values

    def reshape(self, size: int):
        self._data = np.empty((size,2), dtype = np.float32)
class ArrayModel(QAbstractListModel):
    emmiter = Signal(np.ndarray)

    def __init__(self, names: Tuple[str] = None, dtype=np.float64, parent: QObject | None = ...) -> None:
        super().__init__(parent)

        if names is not None:
            size = len(names)
        else:
            size = 0
        self._data = np.empty(size, dtype=dtype)
        self._names = names 

    def rowCount(self, parent: QModelIndex | QPersistentModelIndex = ...) -> int:
        return self._data.size 

    def data(self, index: QModelIndex, role: int) -> Any:
        if role == Qt.ItemDataRole.DisplayRole:
            return self._data[index.row()]

        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignCenter
        
        return None
    
    def flags(self, index: QModelIndex | QPersistentModelIndex) -> Qt.ItemFlag:
        return Qt.ItemFlag.ItemIsEditable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
    
    def setData(self, index: QModelIndex | QPersistentModelIndex, value: Any, role: int = ...) -> bool:
        if role == Qt.ItemDataRole.EditRole:
            self._data[index.row()] = value
            return True
        return super().setData(index, value, role)
    
    def updateItem(self, row: int, value: float):
        index = self.createIndex(row, 0)
        self.setData(index, value, role=Qt.ItemDataRole.EditRole)
        self.emmiter.emit(self._data)

    def updateData(self, values: np.ndarray):
        self._data[:] = values
        self.layoutChanged.emit()
        self.emmiter.emit(self._data)
class SliderForm(QGroupBox):
    def __init__(self, model: ArrayModel, limits: Tuple[float], steps: Tuple[float, int], default_values: Tuple[float], title: str, parent=None) -> None:
        super().__init__(parent=parent)
        self.setTitle(title)
        self.setMinimumWidth(300)
        self.mapper = QDataWidgetMapper()
        self.mapper.setSubmitPolicy(QDataWidgetMapper.SubmitPolicy.ManualSubmit)
        self.mapper.setOrientation(Qt.Orientation.Horizontal)
        self.mapper.setModel(model)
        
        grid = QGridLayout()
        for i, (name, step, limit, value) in enumerate(zip(model._names, steps, limits, default_values)):
            label = QLabel(name)
            label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(label, i, 0)

            if isinstance(step, int):
                slider = superqt.QLabeledSlider(Qt.Orientation.Horizontal) 
            
            if isinstance(step, float):
                slider = superqt.QLabeledDoubleSlider(Qt.Orientation.Horizontal) 

            slider.setRange(*limit)
            slider.setValue(value)
            slider.setSingleStep(step)

            grid.addWidget(slider, i, 1)
            self.mapper.addMapping(slider, i)
            slider.valueChanged.connect(lambda v, i=i: model.updateItem(i, v))

        model._data[:] = default_values 
        self.mapper.toFirst()
        layout = QVBoxLayout()
        layout.addLayout(grid)
        self.setLayout(layout)
class WorkerSignals(QObject):
    started = Signal()
    finished = Signal()
    result = Signal(np.ndarray, np.ndarray)
class Worker(QRunnable):

    def __init__(self, *args, **kwargs):
        super(Worker, self).__init__()
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self):
        self.signals.started.emit()

        sim, fes = self.args
        t, θ, ρ, z = sim.solve(
            features=fes,
        )
        self.signals.result.emit(t, z)
        self.signals.finished.emit()
class Simulator(ecg_simulator.AbstractSimulator):
    def dfdt(self, p, ω):
        return ecg_models.Rat.dfdt(p, ω, self.fe)
    
K = wr_transform.TransformParameters(
    P=wr_transform.TransformParameters.kP(.9, .1),
    R=3.0,
    S=2.5,
    T=wr_transform.TransformParameters.kT(.8, .4),
    W=1.0,
    D=2.0,
    J=wr_transform.TransformParameters.kJ()
)
class ECGSYN(QMainWindow):
    def __init__(self, app):
        super().__init__()
        self.app = app

        self.fmodel = lambda x, fea: ecg_models.Rat.f(x, ecg_models.utils.modelize([0]+fea.tolist(), ecg_models.Rat.Waves))
        self.transform = wr_transform.TransformModel(K, self.fmodel)

        self.fs = 1024
        self.RR = None

        self.sparamsmodel = ArrayModel(names=['ζ', 'amplitude', 'freq'], parent=self)
        self.fparamsmodel = ArrayModel(names=['μ1', 'σ1', 'μ2', 'σ2', 'LF/HF'], parent=self)

        self.tachmodel = ArrayModel(names=['# Beats', 'μ', 'σ'], parent=self)
        self.noisemodel = ArrayModel(names=['β', 'SNR [db]'], parent=self)

        self.meamodel = ArrayModel(names=['RR', 'P duration', 'PR interval', 'QRS duration', 'QT interval', 'P', 'R', 'S', 'T'], parent=self)
        self.fidmodel = ArrayModel(names=['Pon', 'Ppeak', 'Poff', 'QRSon', 'Rpeak', 'Speak', 'J', 'Tpeak', 'Toff'], dtype=np.int64, parent=self)
        self.feamodel = ArrayModel(names=['a_P', 'mu_P', 's_P','a_R', 'mu_R', 's_R','a_S', 'mu_S', 's_S','a_T', 'mu_T', 's_T'], parent=self)

        self.meamodel.emmiter.connect(self.compute_features)
        self.feamodel.emmiter.connect(self.compute_fiducials)
        self.fidmodel.emmiter.connect(self.draw_beat)
        self.fparamsmodel.emmiter.connect(self.draw_psd)
        self.tachmodel.emmiter.connect(self.draw_tachogram)
        self.noisemodel.emmiter.connect(self.draw_noise)

        self.setup_main_ui()
        self.setup_ecg_ui()

        # psd data
        self.bimodal = ecg_simulator.BiModal()
        PSD_SIZE = 1000
        x = np.linspace(0, 1, num=PSD_SIZE)
        self.psdViewer.reshape(PSD_SIZE)
        self.psdViewer.set_x(x)

        # tachogram data
        self.Nb = None
        self.mean_RR = None
        
        # init
        self.compute_features(self.meamodel._data)
        self.draw_psd(self.fparamsmodel._data)

    def compute_features(self, values: np.ndarray):
        new_RR, measurements = values[0], values[1:]
        if new_RR != self.RR:
            self.beat_size = int(new_RR * self.fs)
            self.RR = new_RR
            self.beatViewer.reshape(self.beat_size)
            self.beatViewer.set_x(np.linspace(0, 2*np.pi, num=self.beat_size))

        measurements = np.insert(measurements, 5, 0) # insert null Q
        measurements[:4] *=  2*np.pi / self.RR # time 2 rad conversion
        features = self.transform.inverse(measurements)
        self.feamodel.updateData(features)

    def compute_fiducials(self, values: np.ndarray):
        _, fea_t = self.transform.split(values, 'features')
        fiducials = self.transform.F @ fea_t
        fiducials *= self.beat_size / (2*np.pi)# rad 2 sample conversion
        self.fidmodel.updateData(fiducials)

    def draw_beat(self, values: np.ndarray):
        x = self.beatViewer.get_x()
        beat = self.fmodel(x, self.feamodel._data)
        self.beatViewer.set_line(y=beat)

        values = np.clip(values, 0, self.beat_size-1)
        xfid = x[values]
        yfid = beat[values]
        self.beatViewer.set_markers(xfid, yfid)
        self.beatViewer.set_range(y_lims=(beat.min(), beat.max()))

    def draw_psd(self, values: np.ndarray):
        x = self.psdViewer.get_x()
        psd = self.bimodal.pdf(x, *values.tolist())
        self.psdViewer.set_line(y=psd) 
        self.psdViewer.set_range(y_lims=(psd.min(), psd.max()))
        self.draw_tachogram(self.tachmodel._data)

    def draw_tachogram(self, values: np.ndarray):
        Nb, tparams = int(values[0]), values[1:].tolist()

        (_, rr), _ = ecg_simulator.tachogram(
            self.fparamsmodel._data.tolist(),
            tparams,
            Nb,
            self.fs,
            return_Nb = True
        )

        self.tachViewer.reshape(Nb)
        x = np.arange(Nb)
        self.tachViewer.set_line(x=x, y=rr) 
        self.tachViewer.set_range(x_lims=(0, Nb), y_lims=(rr.min(), rr.max()))

    def draw_noise(self, values: np.ndarray = None):
        if values is None: values = self.noisemodel._data
        if self.noiseForm.isChecked():
            y = self.sim.add_noise(self.noiseless_y, beta=values[0], snr=values[1], seed=0, in_place=False)
            self.ecgViewer.set_line(y=y)
        else:
            self.ecgViewer.set_line(y=self.noiseless_y)


    def show_progress(self):
        self.syn_button.hide()
        self.bar.show()
        # self.bar.setValue(50)

    def show_button(self):
        self.bar.hide()
        self.syn_button.show()
        # self.bar.reset()

    def synthetize(self, e: QEvent):
        ζ, resp = self.sparamsmodel._data[0], self.sparamsmodel._data[1:].tolist()
        self.sim = Simulator(
            fs=self.fs, # sampling frequency
            ζ=ζ, # damping factor
            resp=resp # respiration baseline (amplitud, frequency)
        ) 

        fe = ecg_models.utils.modelize([0]+self.feamodel._data.tolist(), ecg_models.Rat.Waves)
        fes = ecg_simulator.tachogram_features(fe, self.tachViewer.get_y())

        worker = Worker(self.sim,fes)
        worker.signals.started.connect(self.show_progress)
        worker.signals.finished.connect(self.show_button)
        worker.signals.result.connect(self.draw_ecg)
        QThreadPool.globalInstance().start(worker)

    def draw_ecg(self, x: np.ndarray, y: np.ndarray):
        
        self.noiseless_y = y.copy()
        self.ecgViewer.reshape(x.size)
        self.ecgViewer.set_line(x=x, y=y)
        self.ecgViewer.set_range(
            x_lims=(0, x.max()),
            y_lims=(y.min(), y.max())
        )

        self.draw_noise()
        self.ecg_window.show()

    def setup_main_ui(self):

        self.setWindowTitle('ECGSYN')

        self.beatViewer = BaseViewer(
            n_markers=self.fidmodel.rowCount(),
            parent=self
        )

        self.psdViewer = BaseViewer(
            parent=self
        )

        self.tachViewer = BaseViewer(
            parent=self
        )

        self.ecgViewer = BaseViewer(
            parent=self
        )
        
        self.meaForm = SliderForm(
            self.meamodel,
            limits=[(15, 300), (20, 30), (40, 70),(10, 40),(60, 120),(-.25, .25), (.3, 1.),(-.6, -.1),(.1, .4)],
            steps=[1, 1, 1, 1, 1, .01, .01, .01, .01],
            default_values=[250, 22, 50, 16, 80, .07, .6, -.3, .12],
            title='Measurements',
            parent=self
        )

        self.fparamsForm = SliderForm(
            self.fparamsmodel,
            limits=[(.01, 1.), (.01, 1.), (.01, 1.), (.01, 1.), (.1, 1.)],
            steps=[.01,.01,.01,.01,.1,],
            default_values=[.1, .01, .25, .01, .5],
            title='PSD',
            parent=self
        )

        self.tachForm = SliderForm(
            self.tachmodel,
            limits=[(1, 1000), (.1, 2.), (.01, 1.)],
            steps=[1, .01,.01,],
            default_values=[5, 1., .1,],
            title='Tachogram',
            parent=self
        )

        self.simForm = SliderForm(
            self.sparamsmodel,
            limits=[(.01, 1.), (0, 2.), (.01, 2.)],
            steps=[.01, .01,.01,],
            default_values=[.1, .1, .75,],
            title='Simulator',
            parent=self
        )
        
        # vertical splitter
        v_left_splitter = QSplitter(Qt.Orientation.Vertical)

        v_left_splitter.addWidget(self.meaForm)
        v_left_splitter.setStretchFactor(0, 5)

        v_left_splitter.addWidget(self.fparamsForm)
        v_left_splitter.setStretchFactor(1, 5)

        v_left_splitter.addWidget(self.tachForm)
        v_left_splitter.setStretchFactor(2, 5)

        v_left_splitter.addWidget(self.simForm)
        v_left_splitter.setStretchFactor(2, 5)

        # self.progressDialog = QProgressDialog('Simulating ECG ...', 'Abort', 0, 100, self)
        self.bar = QProgressBar()
        self.bar.setMinimum(0)
        self.bar.setMaximum(0)
        self.bar.setValue(0)
        self.bar.hide()
        # self.progressDialog.setBar(bar)
        # self.progressDialog.setWindowModality(Qt.Windowmod)

        self.syn_button = QPushButton('Synthetize')
        self.syn_button.clicked.connect(self.synthetize)

        button_bar_splitter = QSplitter(Qt.Orientation.Horizontal)
        button_bar_splitter.addWidget(self.bar)
        button_bar_splitter.addWidget(self.syn_button)


        v_left_splitter.addWidget(button_bar_splitter)
        v_left_splitter.setStretchFactor(4, 5)

        v_right_splitter = QSplitter(Qt.Orientation.Vertical)
        v_right_splitter.addWidget(self.beatViewer)
        v_right_splitter.setStretchFactor(0, 3)

        v_right_splitter.addWidget(self.psdViewer)
        v_right_splitter.setStretchFactor(1, 3)

        v_right_splitter.addWidget(self.tachViewer)
        v_right_splitter.setStretchFactor(2, 3)

        # horizontal
        h_splitter = QSplitter(Qt.Orientation.Horizontal)
        h_splitter.addWidget(v_left_splitter)
        h_splitter.addWidget(v_right_splitter)
        self.setCentralWidget(h_splitter)

    def setup_ecg_ui(self):

        self.ecg_window = QSplitter(Qt.Orientation.Horizontal, windowFlags = Qt.WindowType.Window)
        self.ecg_window.setMaximumHeight(300)

        self.noiseForm = SliderForm(
            self.noisemodel,
            limits=[(0, 5), (-5, 20)],
            steps=[1, 1],
            default_values=[2, 5],
            title='Noise',
            parent=self.ecg_window
        )
        self.noiseForm.setCheckable(True)
        self.noiseForm.setChecked(False)
        self.noiseForm.clicked.connect(lambda e: self.draw_noise())
        
        save_button = QPushButton('Save')
        save_button.clicked.connect(self.save)

        self.ecgViewer = BaseViewer(
            interactive=True,
            parent=self.ecg_window
        )

        v_left_splitter = QSplitter(Qt.Orientation.Vertical)

        v_left_splitter.addWidget(self.noiseForm)
        v_left_splitter.setStretchFactor(0, 2)

        v_left_splitter.addWidget(save_button)
        v_left_splitter.setStretchFactor(1, 2)


        self.ecg_window.addWidget(v_left_splitter)
        self.ecg_window.setStretchFactor(0, 2)

        self.ecg_window.addWidget(self.ecgViewer)
        self.ecg_window.setStretchFactor(1, 2)

        self.show()

    def save(self, e):
        path, _ = QFileDialog.getSaveFileName(self, caption='Save File', filter='MATLAB file (*.mat)')
        if path:
            sp.io.savemat(path, {'fs':self.fs, 'x': self.ecgViewer.get_x(), 'y': self.ecgViewer.get_y()})




if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = ECGSYN(app)
    sys.exit(app.exec())