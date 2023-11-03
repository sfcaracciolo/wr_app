import numpy as np
from wr_transform import TransformModel, TransformParameters
from ecg_models.Rat import Waves, f
from ecg_models.utils import modelize

K = TransformParameters(
    P=TransformParameters.kP(.9, .1),
    R=3.0,
    S=2.5,
    T=TransformParameters.kT(.8, .4),
    W=1.0,
    D=2.0,
    J=TransformParameters.kJ()
)

model = lambda x, fea: f(x, modelize([0]+fea.tolist(), Waves))
tr_model = TransformModel(K, model)

measurements = np.zeros(9, dtype=np.float64)
features = np.zeros((4, 3), dtype=np.float64)
fiducials = np.zeros(9, dtype=np.float32)

RED = np.array([1., 0, 0, .5], dtype=np.float32)
RR = 250.
Y_RANGE = (-1000, 1000)
WINDOW = 1000
