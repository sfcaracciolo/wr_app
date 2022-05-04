import numpy as np
from wr_core import utils 
import pickle

measurements = np.zeros((2, 9), dtype=np.float64)
features = np.zeros((4, 3), dtype=np.float64)
fiducials = np.zeros(9, dtype=np.float32)

RED = np.array([1., 0, 0, .5], dtype=np.float32)
RR_MAX = int(300.)
Y_RANGE = (-1000, 1000)
N_MAX = 10000

class Status:
    def __init__(self) -> None:
        z1, z2 = utils.z(.8), utils.z(.4)
        Kp, Kr, Ks, Kt, Kw, Kd = 3., 3., 3., utils.omega(z1, z2), 1., 3.
        self.Tc = utils.build_Tc(Kp, Kr, Ks, Kt, Kw, Kd)
        self.Ta = utils.build_Ta(Kp, Kr, Ks, Kt)

try:
    with open('state.pkl', 'rb') as f:
        state = pickle.load(f)
except FileNotFoundError:
    print('State creation ...')
    state = Status()
    with open('state.pkl', 'wb') as f:
        pickle.dump(state, f, pickle.HIGHEST_PROTOCOL)
    print('File saved.')  
else:
    print('File loaded.')
