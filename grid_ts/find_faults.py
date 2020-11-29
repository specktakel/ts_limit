import numpy as np
import sys
import os

incomplete_roi = np.loadtxt('to_be_submitted.txt', dtype=int)
counter = 0
total_num = 0
for roi in incomplete_roi:
    print('incomplete cluster:', roi)
    files = os.listdir(f'outdata/roi_{roi}/ts_95')
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
    # print(i, missing_file_arg)
    submit_this = np.zeros((missing_counter, 2))
    counter = 0
    for c, v in enumerate(missing_file_arg):
        if v == 1:
            submit_this[counter, :] = np.array([roi, c])
            counter += 1
        else:
            continue
    try:
        to_be_submitted = np.append(to_be_submitted, submit_this, axis=0)
        print('appending to existing out-array')
    except:
        to_be_submitted = submit_this.copy()
        print('creating new array')
    print(submit_this)
print(to_be_submitted)
np.savetxt('submit_to_cluster.txt', to_be_submitted, fmt='%1.1i')
        

#np.savetxt('missing_args.txt', arr, fmt='%1.1i')

