from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
import numpy as np
from superqt import QLabeledSlider
from wr_app import Constants
from wr_core import utils
import colorednoise as cn
from scipy import io
class WorkerSignals(QObject):
    started = Signal()
    finished = Signal()
    result = Signal(object)

class Worker(QRunnable):

    def __init__(self, *args, **kwargs):
        super(Worker, self).__init__()
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self):
        self.signals.started.emit()

        mean_measurements = self.args[0] # self.model._data[0,:]
        sd_measurements = self.args[1].copy() # self.model._data[1,:]
        sd_measurements *= np.abs(mean_measurements)
        sd_measurements /= 100

        t, v = utils.wr_ecgsyn(
            self.args[2], # self.n_beats.value()
            mean_measurements,
            sd_measurements,
            Constants.state.Tc,
            remove_drift=True
        )

        ecg = np.empty((t.size, 2), dtype=np.float32)
        ecg[:, 0] = t
        ecg[:, 1] = v[2,:]

        self.signals.result.emit(ecg)
        self.signals.finished.emit()

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

    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)

        self.parent = parent
        self.model = parent.measurements_model
        self.setup_ui()
        self.buttons.button(QDialogButtonBox.Apply).clicked.connect(self.run)
        self.buttons.button(QDialogButtonBox.Save).clicked.connect(self.save)
        self.buttons.button(QDialogButtonBox.Save).setEnabled(False)

    def setup_ui(self):


        self.n_beats = QSpinBox()
        self.n_beats.setRange(10,1000)
        self.n_beats.setValue(10)

        form_layout = QFormLayout()
        form_layout.addRow('# beats', self.n_beats)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Save)
        
        layout = QVBoxLayout()
        layout.addLayout(form_layout)
        layout.addWidget(self.buttons)

        layout.addStretch(1)
        self.setLayout(layout)

    def run(self, e):
        
        worker = Worker(
            self.model._data[0,:],
            self.model._data[1,:],
            self.n_beats.value(),
        )

        worker.signals.started.connect(self.parent.open_progress)
        worker.signals.finished.connect(self.return_to_app)
        worker.signals.result.connect(self.parent.plot)
        QThreadPool.globalInstance().start(worker)

    @Slot()
    def return_to_app(self):
        self.parent.close_progress()
        self.buttons.button(QDialogButtonBox.Save).setEnabled(True)


    def save(self, e):
        path, _ = QFileDialog.getSaveFileName(self, caption='Save File', filter='MATLAB file (*.mat)')
        io.savemat(path, {'data': self.parent.ecgViewer.line.pos})

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
        self.beta.setValue(2)

        self.snr = QLabeledSlider(Qt.Horizontal)
        self.snr.setRange(-5, 20)
        self.snr.setValue(5)

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

class MeasurementsPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self.setup_ui()

    def setup_ui(self):
        self.setMinimumWidth(500)
        model = self.parent().measurements_model
        
        steps = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        median_limits = [
            (200., Constants.RR_MAX), 
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
        self.mapper.setOrientation(Qt.Horizontal)
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
            slider.valueChanged.connect(lambda v, i=0, j=j: model.updateMeasurements(i, j, v))

            slider = QLabeledSlider(Qt.Horizontal)
            slider.setRange(0, 20)
            grid.addWidget(slider, j, 2)
            self.mapper.addMapping(slider, j)
            slider.valueChanged.connect(lambda v, i=1, j=j: model.updateMeasurements(i, j, v))

        self.mapper.toFirst()

        layout = QVBoxLayout()
        layout.addLayout(grid)

        self.setLayout(layout)
