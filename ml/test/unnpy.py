import os

import numpy as np

import GlobalVariable

path = os.path.join(GlobalVariable.datadir, "data.npy")
f = np.load(path)
print f
