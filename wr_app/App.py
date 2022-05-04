
from PySide6.QtWidgets import *
from PySide6.QtCore import *
import sys
import numpy as np
from wr_core import utils
import vispy as vp
from vispy import scene
from wr_app import Models, Panels, Constants

class BeatViewer(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        
        self._data = np.empty((2*Constants.RR_MAX, 2), dtype = np.float32)
        self._data[:,0] = np.arange(0, 2*Constants.RR_MAX)

        self._markers = np.zeros((9, 2), dtype=np.float32)

        self.setup_ui()

    def setup_ui(self):
        self.canvas = scene.SceneCanvas(show=True, bgcolor='white', parent=self)
        self.grid = self.canvas.central_widget.add_grid()

        self.vb1 = self.grid.add_view(row=0, col=0, camera='panzoom')
        self.vb1.camera.interactive = False

        self.line = scene.Line(
            pos = self._data,
            color = 'black',
            parent=self.vb1.scene
        )

        # self.vb2 = self.grid.add_view(row=1, col=0, camera='panzoom')
        # self.vb2.camera.interactive = False

        self.markers = scene.Markers(
            pos = self._markers,
            parent=self.vb1.scene
        )

        layout = QVBoxLayout()
        layout.addWidget(self.canvas.native)
        self.setLayout(layout)

    def set_line(self, features):
        self.vb1.camera.set_range(
            x=(0, 2*features[6]),
            y = Constants.Y_RANGE
        )
        self._data[:,1] = utils.model(self._data[:,0], *features.tolist())
        self.line.set_data(pos=self._data)

    def set_markers(self, fiducials, features):
        self._markers[:, 0] = fiducials
        self._markers[:, 1] = utils.model(fiducials, *features.tolist())

        self.markers.set_data(
            self._markers,
            size = 10,
            edge_width = 0,
            edge_color = Constants.RED,
            face_color = Constants.RED,
        )
        self.markers.update()

class EcgViewer(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        
        self.setup_ui()

    def setup_ui(self):
        self.canvas = scene.SceneCanvas(show=True, bgcolor='white', parent=self)
        self.grid = self.canvas.central_widget.add_grid()

        self.vb1 = self.grid.add_view(row=0, col=0, camera='panzoom')
        self.vb1.camera.interactive = True

        self.line = scene.Line(
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

        self.measurements_model = Models.MeasurementsModel(Constants.measurements)
        self.features_model = Models.FeaturesModel(Constants.features)
        self.fiducials_model = Models.FiducialsModel(Constants.fiducials)

        self.setup_ui()

        # self.measurements_model.computeFeatures.connect(self.features_model.updateFeatures)
        self.features_model.computeSignal.connect(self.update)
        self.fiducials_model.drawMarkers.connect(self.beatViewer.set_markers)

        self.measurements_model.setDefaultMeasurements(self.measurementsPanel.default_median_values)

    def open_progress(self):
        self.progressDialog.setValue(50)

    def close_progress(self):
        self.progressDialog.reset()

    def update(self, features):
        self.beatViewer.set_line(features)
        # self.fiducials_model.updateFiducials(features)

    def plot(self, ecg):
        self.noisePanel.add_noise(ecg[:,1])
        self.ecgViewer.set_line(ecg)

    def setup_ui(self):
        self.setWindowTitle('ecgsyn')

        self.measurementsPanel = Panels.MeasurementsPanel(parent=self)
        self.noisePanel = Panels.NoisePanel(parent=self)
        self.runPanel = Panels.RunPanel(parent=self)
        # measurementsView = TablePanel(self.measurements_model, parent=self)
        # featuresTable = TablePanel(self.features_model, parent=self)
        # fiducialsTable = TablePanel(self.fiducials_model, parent=self)
        self.beatViewer = BeatViewer(parent=self)
        self.ecgViewer = EcgViewer(parent=self)

        # vertical splitter
        vsplitter = QSplitter(Qt.Vertical)

        vsplitter.addWidget(self.measurementsPanel)
        vsplitter.setStretchFactor(0, 3)

        # vsplitter.addWidget(measurementsView)
        # vsplitter.setStretchFactor(1, 1)

        # vsplitter.addWidget(fiducialsTable)
        # vsplitter.setStretchFactor(2, 1)

        # vsplitter.addWidget(featuresTable)
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

        self.progressDialog = QProgressDialog('Simulating ECG ...', 'Abort', 0, 100, self)
        bar = QProgressBar()
        bar.setMinimum(0)
        bar.setMaximum(0)
        bar.setValue(0)
        self.progressDialog.setBar(bar)
        self.close_progress()
        # self.progressDialog.setWindowModality(Qt.Windowmod)

        self.show()

def run():
    app = QApplication(sys.argv)
    viewer = ecgsyn(app)
    sys.exit(app.exec())