# nuitka-project: --standalone
# nuitka-project: --msvc=14.3
# nuitka-project: --lto=yes
# nuitka-project: --enable-plugin=pyside6
# nuitka-project: --enable-console
# nuitka-project: --windows-uac-admin
# nuitka-project: --windows-product-name=wr_app
# nuitka-project: --windows-product-version=0.0.0.1

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
            axis_width=1,
            tick_width=1
        )
        xaxis_kwargs = axis_kwargs | {'tick_label_margin':18}
        yaxis_kwargs = axis_kwargs | {'tick_label_margin':5}

        if n_markers is not None:
            self._markers = np.zeros((n_markers, 2), dtype=np.float32)

        self.canvas = scene.SceneCanvas(show=True, bgcolor='white', parent=self)
        if interactive: self.canvas.connect(self.on_key_press)

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
            **yaxis_kwargs,
        )
        yaxis.width_max = 50
        self.grid.add_widget(yaxis, row=1, col=0)
        yaxis.link_view(self.vb1)

        xaxis = scene.AxisWidget(
            orientation='bottom',
            **xaxis_kwargs
        )
        xaxis.height_max = 50
        self.grid.add_widget(xaxis, row=2, col=1)
        xaxis.link_view(self.vb1)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas.native)
        self.setLayout(layout)

    def set_range(self, x_lims: Tuple = None, y_lims: Tuple = None):
        if x_lims is None and y_lims is None:
            x_lims = (self._data[:,0].min(), self._data[:,0].max())
            y_lims = (self._data[:,1].min(), self._data[:,1].max())
        if y_lims is not None and np.allclose(y_lims[0], y_lims[1]): 
            eps = 1e-4
            y_lims[0] -= eps
            y_lims[1] += eps
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

    def on_key_press(self, e):
        key = e.key.name
        if key == 'R':
            self.set_range()
class ArrayModel(QAbstractListModel):
    emmiter = Signal(np.ndarray)

    def __init__(self, names: Tuple[str] = None, dtype=np.float64, parent: QObject | None = ...) -> None:
        super().__init__(parent=None)

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
        self.setAutoDelete(True)

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
        self.beat_size = None
        self.Nb = None
        self.rr_mean = None

        self.fmodel = lambda x, fea: ecg_models.Rat.f(x, ecg_models.utils.modelize([0]+fea.tolist(), ecg_models.Rat.Waves))
        self.transform = wr_transform.TransformModel(K, self.fmodel)


        self.simmodel = ArrayModel(names=['damping'])
        self.respmodel = ArrayModel(names=['A [mV]', 'f [Hz]'])
        self.tachmodel = ArrayModel(names=['fs [Hz]', 'Beats [#]', 'μRR [ms]', 'σRR [ms]', 'μLF [Hz]', 'σLF [Hz]', 'μHF [Hz]', 'σHF [Hz]', 'LF/HF'])
        self.noisemodel = ArrayModel(names=['β', 'SNR [db]'])
        self.meamodel = ArrayModel(names=['P duration [ms]', 'PR interval [ms]', 'QRS duration [ms]', 'QT interval [ms]', 'P [mV]', 'R [mV]', 'S [mV]', 'T [mV]'], )
        self.fidmodel = ArrayModel(names=['Pon', 'Ppeak', 'Poff', 'QRSon', 'Rpeak', 'Speak', 'J', 'Tpeak', 'Toff'], dtype=np.int64)
        self.feamodel = ArrayModel(names=['a_P', 'mu_P', 's_P','a_R', 'mu_R', 's_R','a_S', 'mu_S', 's_S','a_T', 'mu_T', 's_T'])

        self.meamodel.emmiter.connect(self.compute_features)
        self.feamodel.emmiter.connect(self.compute_fiducials)
        self.fidmodel.emmiter.connect(self.draw_beat)
        self.tachmodel.emmiter.connect(self.compute_tachogram)
        self.noisemodel.emmiter.connect(self.add_additives)
        self.respmodel.emmiter.connect(self.add_additives)

        self.setup_main_ui()
        self.setup_ecg_ui()

        # init
        self.compute_tachogram()

        self.show()

    def compute_features(self, values: np.ndarray = None):
        if values is None: values = self.meamodel._data
        measurements = np.insert(values, 5, 0) # insert null Q
        measurements[:4] *=  2*np.pi / self.rr_mean # time 2 rad conversion
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
        self.beatViewer.set_range(y_lims=[beat.min(), beat.max()])

    def compute_tachogram(self, values: np.ndarray = None):
        if values is None: values = self.tachmodel._data 

        new_fs, new_Nb, tparams, fparams = values[0], int(values[1]), values[2:4]/1000., values[4:] # ms to s

        new_beat_size, (_, rr), (f, psd) = ecg_simulator.tachogram(
            fparams.tolist(),
            tparams.tolist(), 
            new_Nb,
            new_fs,
        )

        rr = rr[::new_beat_size]
        rr_mean, rr_std = tparams.tolist()
        if new_beat_size != self.beat_size:
            self.rr_mean = rr_mean*1000. # s to ms
            self.beatViewer.reshape(new_beat_size)
            self.beatViewer.set_x(np.linspace(0, 2*np.pi, num=new_beat_size))
            self.beat_size = new_beat_size
            self.fs = new_fs
            self.compute_features()
        
        # draw tachogram
        if new_Nb != self.Nb:
            self.tachViewer.reshape(new_Nb)
            x = np.arange(new_Nb)
            self.tachViewer.set_line(x=x, y=rr) 
            self.Nb = new_Nb
        else:
            self.tachViewer.set_line(y=rr) 
        
        exc = 4*rr_std
        self.tachViewer.set_range(
            x_lims=(0, self.Nb-1),
            y_lims=[rr_mean-exc, rr_mean+exc])
        
        # draw psd 
        self.psdViewer.reshape(f.size)
        self.psdViewer.set_line(x=f, y=psd) 
        self.psdViewer.set_range(
            x_lims=(0, 2),
            y_lims=[psd.min(), psd.max()]
        )

    def add_additives(self, _ = None):

        y=self.raw_y.copy()
        if self.noiseForm.isChecked():
            noise = self.noisemodel._data
            self.sim.add_noise(y, beta=noise[0], snr=noise[1], seed=0, in_place=True)
        
        resp = self.respmodel._data
        if np.all(resp != 0.):
            y += resp[0] * np.sin(2*np.pi*resp[1]*self.raw_x)
        
        self.ecgViewer.set_line(x=self.raw_x, y=y)
        
    def set_loading(self):
        self.syn_button.hide()
        self.bar.show()
        self.v_left_splitter.setEnabled(False)

    def reset_loading(self):
        self.bar.hide()
        self.syn_button.show()
        self.v_left_splitter.setEnabled(True)

    def synthetize(self, e: QEvent):
        ζ = self.simmodel._data[0]
        self.sim = Simulator(
            fs=self.fs, # sampling frequency
            ζ=ζ, # damping factor
        ) 

        fe = ecg_models.utils.modelize([0]+self.feamodel._data.tolist(), ecg_models.Rat.Waves)
        fes = ecg_simulator.tachogram_features(fe, self.tachViewer.get_y())
        worker = Worker(self.sim,fes)
        worker.signals.started.connect(self.set_loading)
        worker.signals.finished.connect(self.reset_loading)
        worker.signals.result.connect(self.draw_ecg)
        QThreadPool.globalInstance().start(worker)

    def draw_ecg(self, x: np.ndarray, y: np.ndarray):
        
        self.ecg_window.close()

        self.raw_x = x.copy()
        self.raw_y = y.copy()

        self.ecgViewer.reshape(x.size)
        self.add_additives()
        self.ecgViewer.set_range()

        self.ecg_window.setFixedHeight(300)
        self.ecg_window.show()

    def setup_main_ui(self):

        self.setWindowTitle('ECGSYN')

        self.beatViewer = BaseViewer(
            n_markers=self.fidmodel.rowCount(),
        )

        self.psdViewer = BaseViewer()

        self.tachViewer = BaseViewer()

        self.ecgViewer = BaseViewer()
        
        self.meaForm = SliderForm(
            self.meamodel,
            limits=[(20, 30), (40, 70),(10, 40),(60, 120),(-.25, .25), (.3, 1.),(-.6, -.1),(.1, .4)],
            steps=[1, 1, 1, 1, .01, .01, .01, .01],
            default_values=[22, 50, 16, 80, .07, .6, -.3, .12],
            title='Measurements',
        )

        self.tachForm = SliderForm(
            self.tachmodel,
            limits=[(512, 2048), (5, 1000), (150, 1000), (0, 100), (.01, 1.), (.01, 1.), (.01, 1.), (.01, 1.), (.1, 1.)],
            steps=[1, 1, 1, 1, .01,.01,.01,.01,.1,],
            default_values=[1024, 5, 250, 10, .1, .01, .25, .01, .5],
            title='Tachogram',
        )

        self.simForm = SliderForm(
            self.simmodel,
            limits=[(.01, 1.)],
            steps=[.01],
            default_values=[.1],
            title='Simulator',
        )

        # vertical splitter
        self.v_left_splitter = QSplitter(
            Qt.Orientation.Vertical,
        )

        self.v_left_splitter.addWidget(self.meaForm)
        self.v_left_splitter.addWidget(self.tachForm)
        self.v_left_splitter.addWidget(self.simForm)

        h_widget = QWidget()
        h_layout = QHBoxLayout(h_widget)

        self.bar = QProgressBar()
        self.bar.setMinimum(0)
        self.bar.setMaximum(0)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.hide()

        self.syn_button = QPushButton('Synthetize')
        self.syn_button.setMaximumWidth(150)
        self.syn_button.clicked.connect(self.synthetize)
        h_layout.addWidget(self.syn_button)
        h_layout.addWidget(self.bar)
        
        self.v_left_splitter.addWidget(h_widget)

        v_right_splitter = QSplitter(Qt.Orientation.Vertical)
        v_right_splitter.addWidget(self.beatViewer)
        v_right_splitter.addWidget(self.psdViewer)
        v_right_splitter.addWidget(self.tachViewer)

        # horizontal
        h_splitter = QSplitter(Qt.Orientation.Horizontal)
        h_splitter.addWidget(self.v_left_splitter)
        h_splitter.addWidget(v_right_splitter)
        self.setCentralWidget(h_splitter)

    def setup_ecg_ui(self):

        self.ecg_window = QSplitter(
            Qt.Orientation.Horizontal,
            windowFlags = Qt.WindowType.Window,
        )
        self.ecg_window.setFixedHeight(300)
        self.ecg_window.setWindowTitle('ECG Viewer')

        self.noiseForm = SliderForm(
            self.noisemodel,
            limits=[(0, 5), (-100, 20)],
            steps=[1, 1],
            default_values=[2, 5],
            title='Noise',
        )
        self.noiseForm.setCheckable(True)
        self.noiseForm.setChecked(False)
        self.noiseForm.clicked.connect(self.add_additives)

        self.respForm = SliderForm(
            self.respmodel,
            limits=[(0, 2.), (0, 2.)],
            steps=[.01,.01,],
            default_values=[.1, .75,],
            title='Respiration',
        )

        save_button = QPushButton('Save')
        save_button.setMaximumWidth(150)
        save_button.clicked.connect(self.save)
        h_widget = QWidget()
        h_layout = QHBoxLayout(h_widget)
        h_layout.addWidget(save_button)

        self.ecgViewer = BaseViewer(
            interactive=True,
        )

        v_left_splitter = QSplitter(Qt.Orientation.Vertical)

        v_left_splitter.addWidget(self.noiseForm)
        v_left_splitter.addWidget(self.respForm)
        v_left_splitter.addWidget(h_widget)
        self.ecg_window.addWidget(v_left_splitter)
        self.ecg_window.addWidget(self.ecgViewer)

    def save(self, e):
        path, _ = QFileDialog.getSaveFileName(self, caption='Save File', filter='MATLAB file (*.mat)')
        if path:
            sp.io.savemat(path, {'fs':self.fs, 'y': self.ecgViewer.get_y()})

    def close(self) -> bool:
        self.ecg_window.close()
        return super().close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = ECGSYN(app)
    sys.exit(app.exec())