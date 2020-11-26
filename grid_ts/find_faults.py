import numpy as np
import sys
import os


files = os.listdir('outdata/roi_3/ts_95')
missing_file_arg = np.ones((900))
missing_counter = 900
needed = np.linspace(0, 899, num=900)
for c, v in enumerate(files):
    num = v.rsplit('_', maxsplit=1)[-1]
    num = num.rstrip('.dat')
    num = int(num)
    # print(num)
    for i in needed:
        if num == i:
            missing_file_arg[num] = 0
            missing_counter -= 1
            break
        else:
            continue
print(missing_file_arg)
to_be_submitted = np.zeros((missing_counter, 2))
counter = 0
for c, v in enumerate(missing_file_arg):
    if v == 1:
        to_be_submitted[counter, :] = np.array([3, c])
        counter += 1
    else:
        continue

np.savetxt('to_be_submitted.txt', to_be_submitted, fmt='%1.1i')
        

#np.savetxt('missing_args.txt', arr, fmt='%1.1i')

