# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-04-25 19:03:52
# @Last Modified by:   yulidong
# @Last Modified time: 2018-09-13 11:38:48
import scipy.io
import numpy as np
import os
file=r'/home/lidong/Documents/datasets/Driving/train_data_clean_pass'
left=os.path.join(file,'left_re')
match=os.path.join(file,'match_re')
files=os.listdir(left)
files.sort()
num=np.random.randint(low=0,high=len(files),size=900)
test=[]
for i in range(len(num)):
    if num[i] not in test:
        test.append(num[i])
for i in range(len(test)):
    os.rename(os.path.join(left,files[test[i]]),os.path.join(file,'test/left',files[test[i]]))
for i in range(len(test)):
    os.rename(os.path.join(match,files[test[i]]),os.path.join(file,'test/match',files[test[i]]))

