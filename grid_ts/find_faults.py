import numpy as np
import sys
import os


files = os.listdir('outdata/roi_0')
missing_file_arg = []
for c, v in enumerate(files):
    if 'out' in v:
        num = v.lstrip('out')
        num = num.rstrip('.dat')
        missing_file_arg.append(int(num))

print(missing_file_arg)
print(len(missing_file_arg))

arr = np.zeros((len(missing_file_arg)))
for c, v in enumerate(missing_file_arg):
    arr[c] = v

np.savetxt('missing_args.txt', arr, fmt='%1.1i')

