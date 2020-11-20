import numpy as np
import sys
#CLA: start and stop (excluded) for ROIs to be generated

num_ROI_beg = int(sys.argv[1])
num_ROI_end = int(sys.argv[2])
num_delta = num_ROI_end - num_ROI_beg

sim = np.linspace(num_ROI_beg, num_ROI_end, num=num_delta, endpoint=False, dtype=int)
grid = np.linspace(0, 900, num=900, endpoint=False, dtype=int)
index = np.zeros((sim.shape[0], grid.shape[0], 2), dtype=int)
for i in range(sim.shape[0]):
    for j in range(grid.shape[0]):
        index[i, j, :] = sim[i], grid[j]
index = index.reshape((sim.shape[0] * grid.shape[0], 2))
print(index)
print(index.shape)
np.savetxt('submit_this.txt', index, fmt='%1.1i')
