import numpy as np
import sys
#CLA: start and stop (excluded) for ROIs to be generated

# num_ROI_beg = int(sys.argv[1])
# num_ROI_end = int(sys.argv[2])
# num_delta = num_ROI_end - num_ROI_beg
'''
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
'''
grid = np.linspace(0, 900, num=900, endpoint=False)
r_0, r_1 = 0, 4
roi_0, roi_1 = 435, 436
r = np.linspace(r_0, r_1, num=r_1-r_0, endpoint=False, dtype=int)
roi = np.linspace(roi_0, roi_1, num=roi_1-roi_0, endpoint=False, dtype=int)
args = np.zeros((r.shape[0], roi.shape[0], grid.shape[0], 3), dtype=int)
for l in range(roi.shape[0]):
    for j in range(r.shape[0]):
        for i in range(grid.shape[0]):
            args[j, l, i, :] = grid[i], roi[l], r[j]
args = args.reshape((grid.shape[0] * r.shape[0] * roi.shape[0], 3))
np.savetxt('submit_this.txt', args, fmt='%1.1i')
