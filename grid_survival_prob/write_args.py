import numpy as np

grid = np.linspace(0, 900, num=900, endpoint=False)
roi = 3
start = 110
r = np.linspace(start, start+roi, num=roi, endpoint=False, dtype=int)
args = np.zeros((r.shape[0], grid.shape[0], 2), dtype=int)
for j in range(0, r.shape[0]):
    #print(j)
    for i in range(grid.shape[0]):
        #print(i)
        args[j, i, :] = grid[i], r[j]
args = args.reshape((grid.shape[0] * r.shape[0], 2))

np.savetxt('submit_this.txt', args, fmt="%1.1i")
