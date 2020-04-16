import sys
import numpy as np
w_file = sys.argv[1]
h_file = sys.argv[2]
m = int(sys.argv[3])
n = int(sys.argv[4])
r = int(sys.argv[5])
np.random.seed(42069)
w = np.random.rand(m, r)
h = np.random.rand(r, n)
np.save(w_file, w)
np.save(h_file, h)

