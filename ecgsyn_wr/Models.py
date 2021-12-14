import typing
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
import numpy as np
from ecgsyn_wr import utils

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