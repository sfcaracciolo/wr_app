import typing
from PySide6.QtCore import *
import numpy as np
import Constants

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
        _, fea_t = Constants.tr_model.split(features, 'features')
        fiducials = Constants.tr_model.F @ fea_t
        self._data[:] = fiducials
        self.layoutChanged.emit()
        self.drawBeat.emit(fiducials, features)

class MeasurementsModel(QAbstractTableModel):
    computeFeatures = Signal(np.ndarray)

    def __init__(self, data: np.ndarray, parent=None) -> None:
        super().__init__(parent=parent)
        self._data = data
        self._header = ['RR', 'P duration', 'PR interval', 'QRS duration', 'QT interval', 'P', 'R', 'S', 'T']

    def rowCount(self, parent=None) -> int:
        return 1
    
    def columnCount(self, parent = None) -> int:
        return self._data.size

    # def headerData(self, section: int, orientation: Qt.Orientation, role: int) -> typing.Any:
    #     if role == Qt.DisplayRole:
    #         if orientation == Qt.Horizontal:
    #             return self._hheader[section]
    #         else:
    #             return self._vheader[section]

    #     return super().headerData(section, orientation, role=role)

    def data(self, index: QModelIndex, role: int) -> typing.Any:
        if role == Qt.DisplayRole:
            j = index.column()
            value = self._data[j]
            return float(value)

        if role == Qt.TextAlignmentRole:
            return Qt.AlignCenter

    def flags(self, index: typing.Union[QModelIndex, QPersistentModelIndex]) -> Qt.ItemFlags:
        return Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def updateMeasurements(self, j, v):
        self._data[j] = v
        index = self.index(0, j)
        self.dataChanged.emit(index, index)
        self.computeFeatures.emit(self._data)

    def setDefaultMeasurements(self, values):
        self._data[:] = np.array(values)
        self.layoutChanged.emit()
        self.computeFeatures.emit(self._data)

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
        RR, mea = measurements[0], measurements[1:] # 
        mea = np.insert(mea, 5, 0) # insert null Q
        mea[:4] *=  2*np.pi / RR # time 2 rad conversion
        features = Constants.tr_model.inverse(mea)
        self._flatten_data[:] = features
        self.layoutChanged.emit()
        self.computeFiducials.emit(features)
        # self.computeSignal.emit(self._flatten_data)