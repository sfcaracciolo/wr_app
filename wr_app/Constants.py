import numpy as np

inputs = np.zeros((2, 9), dtype=np.float64)
params = np.zeros((4, 3), dtype=np.float64)
fiducials = np.zeros(13, dtype=np.float32)

RED = np.array([1., 0, 0, .5], dtype=np.float32)
RR_MAX = int(300.)
Y_RANGE = (-1000, 1000)
N_MAX = 10000