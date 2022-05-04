import typing
from PySide6.QtCore import *
import numpy as np
from wr_core import utils
from wr_app import Constants
class FiducialsModel(QAbstractTableModel):
    drawBeat = Signal(np.ndarray, np.ndarray)

    def __init__(self, data: np.ndarray, parent=None) -> None:
        super().__init__(parent=parent)
        self._data = data
        # self._names = ['Pon', 'P', 'Pend', 'QRSon', 'Q', 'R', 'S', 'QRSend','res','Ton', 'T', 'Tend', 'res']
        self._names = ['Pon', 'Ppeak', 'Poff', 'QRSon', 'Rpeak', 'Speak', 'J', 'Tpeak', 'Toff']

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
    
    def updateFiducials(self, features):
        theta_t = features[4:]
        fiducials = Constants.state.Ta @ theta_t

        self._data[:] = fiducials
        self.layoutChanged.emit()
        self.drawBeat.emit(fiducials, features)
class MeasurementsModel(QAbstractTableModel):
    computeFeatures = Signal(np.ndarray)

    def __init__(self, data: np.ndarray, parent=None) -> None:
        super().__init__(parent=parent)
        self._data = data
        self._hheader = ['RR interval', 'P duration', 'PR interval', 'QRS duration', 'QT interval', 'P peak', 'R peak', 'S peak', 'T peak']
        self._vheader = ['mu', 'sigma']

    def rowCount(self, parent=None) -> int:
        return 2
    
    def columnCount(self, parent = None) -> int:
        return self._data.shape[1]

    # def headerData(self, section: int, orientation: Qt.Orientation, role: int) -> typing.Any:
    #     if role == Qt.DisplayRole:
    #         if orientation == Qt.Horizontal:
    #             return self._hheader[section]
    #         else:
    #             return self._vheader[section]

    #     return super().headerData(section, orientation, role=role)

    def data(self, index: QModelIndex, role: int) -> typing.Any:
        if role == Qt.DisplayRole:
            i = index.row()
            j = index.column()
            value = self._data[i, j]
            return float(value)

        if role == Qt.TextAlignmentRole:
            return Qt.AlignCenter

    def flags(self, index: typing.Union[QModelIndex, QPersistentModelIndex]) -> Qt.ItemFlags:
        return Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def updateMeasurements(self, i, j, v):
        self._data[i, j] = v
        index = self.index(i, j)
        self.dataChanged.emit(index, index)
        if i == 0:
            self.computeFeatures.emit(self._data[0,:])

    def setDefaultMeasurements(self, values):
        self._data[0,:] = np.array(values)
        self.layoutChanged.emit()
        self.computeFeatures.emit(self._data[0,:])

class FeaturesModel(QAbstractTableModel):
    computeFiducials = Signal(np.ndarray)

    def __init__(self, data: np.ndarray, parent=None) -> None:
        super().__init__(parent=parent)
        self._data = data
        self._flatten_data = np.ravel(self._data, order='C')        

        self._hheader = ['a', 'mu', 'sigma']
        self._vheader = ['P', 'R', 'S', 'T']

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

    # def headerData(self, section: int, orientation: Qt.Orientation, role: int) -> typing.Any:
    #     if role == Qt.DisplayRole:
    #         return self._vheader[section] if orientation == Qt.Vertical else self._hheader[section]
    #     return super().headerData(section, orientation, role=role)

    def updateFeatures(self, measurements):
        features = utils.inverse_transform(measurements, Constants.state.Tc)
        self._flatten_data[:] = features
        self.layoutChanged.emit()
        self.computeFiducials.emit(features)
        # self.computeSignal.emit(self._flatten_data)