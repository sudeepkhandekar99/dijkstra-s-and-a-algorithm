#creating testing sets

import numpy as np
import random as rd
rd.seed(69)

itr = 1
for test in range(6):
    itr += 1
    up_lim = itr*itr

    for loop in range(20):
        y = np.array(rd.sample(range(up_lim),up_lim))
        z = np.reshape(y, (itr,itr))
        print(z)
        print(end = "\n")
    print(end = "\n")
