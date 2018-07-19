# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-07-18 18:49:15
# @Last Modified by:   yulidong
# @Last Modified time: 2018-07-19 23:06:30
import numpy as np
import os
import time
import matplotlib.pyplot as plt
def crop(object):
    ground=np.array((object==1).nonzero())
    x1=np.min(ground[0,:])
    x2=np.max(ground[0,:])
    y1=np.min(ground[1,:])
    y2=np.max(ground[1,:])
    size=np.sum(object)
    return x1,y1,x2+1,y2+1,size
left_dir=r'/home/lidong/Documents/datasets/Driving/train_data_clean_pass/left/'
right_dir=r'/home/lidong/Documents/datasets/Driving/train_data_clean_pass/right/'
match_dir=r'/home/lidong/Documents/datasets/Driving/train_data_clean_pass/match/'
left_files=os.listdir(left_dir)
left_files.sort()
right_files=os.listdir(right_dir)
right_files.sort()
for i in range(int(len(left_files)/len(left_files))):
    left=np.load(os.path.join(left_dir,left_files[i]))[...,8]
    right=np.load(os.path.join(right_dir,right_files[i]))[...,8]
    print(i)
    l_box=[]
    r_box=[]
    match=[]
    start=time.time()
    for m in range(int(np.max(left)+1)):
        object=np.where(left==m,1,0)
        l_box.append(crop(object))
    l_box=np.array(l_box)
    for m in range(int(np.max(right)+1)):
        object=np.where(right==m,1,0)
        r_box.append(crop(object))
    r_box=np.array(r_box)
    print(time.time()-start)
    for m in range(int(np.max(left)+1)):
        x1,y1,x2,y2,size=l_box[m]
        left_object_m=np.where(left==m,1,-1)[x1:x2,y1:y2]
        #matching=np.abs(r_box[:,0]-x1)+np.abs(r_box[:,2]-x2)
        #object=np.where(left==m,1,0)[x1:x2,y1:y2]
        #overlap=np.abs(r_box[:,4]-size)/np.max([r_box[:,4],np.ones_like(r_box[:,4])*size],0)
        x_matching=(np.min([np.ones_like(r_box[:,2])*x2,r_box[:,2]],0)-np.max([np.ones_like(r_box[:,0])*x1,r_box[:,0]],0))/(x2-x1)
        x_matching=np.where(x_matching>0.8,1,0)
        y_matching=np.min([(r_box[:,3]-r_box[:,1]),np.ones_like(r_box[:,0])*(y2-y1)],0)/np.max([(r_box[:,3]-r_box[:,1]), \
                   np.ones_like(r_box[:,0])*(y2-y1)],0)
        y_matching=np.where(y_matching>0.5,1,0)
        matching=x_matching*y_matching
        overlap=[]
        for n in range(matching.shape[0]):
            if matching[n]==1:
                r_x1,r_y1,r_x2,r_y2,r_size=r_box[n]
                right_object_n=np.where(right==n,1,0)[x1:x2,r_y1:r_y2]
                if (r_y2-r_y1)>(y2-y1):
                    shift=[]
                    for k in range((r_y2-r_y1)-(y2-y1)+1):
                        tmp_overelap=left_object_m-right_object_n[:,k:k+y2-y1]
                        shift.append(np.sum(np.where(tmp_overelap==0,1,0))/np.max([size,r_size]))
                else:
                    shift=[]
                    for k in range((y2-y1)-(r_y2-r_y1)+1):
                        tmp_overelap=right_object_n-left_object_m[:,k:k+r_y2-r_y1]
                        shift.append(np.sum(np.where(tmp_overelap==0,1,0))/np.max([size,r_size]))                   
                overlap.append(np.max(shift))
            else:
                overlap.append(0)
        match.append(np.argmax(overlap))
#for object in range(int(np.max(left)+1)):
object=23
fig, ax = plt.subplots(nrows=1,ncols=2, sharex=True, sharey=True,figsize=(16, 32))
ax[0].imshow(np.where(left==object,1,0), cmap=plt.cm.gray)
ax[1].imshow(np.where(right==match[object],1,0), cmap=plt.cm.gray)


