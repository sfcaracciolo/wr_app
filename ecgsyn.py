from ecgsyn_wr import utils
import matplotlib.pyplot as plt
import numpy as np

RR = 250.
params = [ 50, 80, 4.4, 641, 125, 2.3, -420, 130, 2.3 ]

t, v = utils.ecgsyn_wr(10, RR, params)
x = v[0,:]
y = v[1,:]
z = v[2,:]

plt.plot(t, z)

plt.show()
