# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-07-18 18:49:15
# @Last Modified by:   yulidong
# @Last Modified time: 2018-08-27 22:19:08
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from multiprocessing import Process,Lock
from multiprocessing import Pool
thread_num=10
def crop(object):
    ground=np.array((object==1).nonzero())
    x1=np.min(ground[0,:])
    x2=np.max(ground[0,:])
    y1=np.min(ground[1,:])
    y2=np.max(ground[1,:])
    size=np.sum(object)
    return x1,y1,x2+1,y2+1,size
def pre_matching(start,end):
    left_dir=r'/home/lidong/Documents/datasets/Driving/train_data_clean_pass/left/'
    right_dir=r'/home/lidong/Documents/datasets/Driving/train_data_clean_pass/right/'
    match_dir=r'/home/lidong/Documents/datasets/Driving/train_data_clean_pass/match/'
    left_files=os.listdir(left_dir)
    left_files.sort()
    right_files=os.listdir(right_dir)
    right_files.sort()
    # s_index=int(np.floor(length/thread_num*index))-2
    # e_index=int(np.floor(length/thread_num*(index+1)))+2
    # if e_index>length:
    #     e_index=length
    # if s_index<0:
    #     s_index=0
    for i in range(int(start),int(end)):
        left=np.load(os.path.join(left_dir,left_files[i]))[...,8]
        right=np.load(os.path.join(right_dir,right_files[i]))[...,8]
        pre=[]
        pre2=[]
        l_box=[]
        r_box=[]
        match=[]
        start_time=time.time()
        for m in range(int(np.max(left)+1)):
            object=np.where(left==m,1,0)
            if np.sum(object)>0:
                l_box.append(crop(object))
            else:
                l_box.append((0,0,1,1,1))
        l_box=np.array(l_box)
        for m in range(int(np.max(right)+1)):
            object=np.where(right==m,1,0)
            if np.sum(object)>0:
                r_box.append(crop(object))
            else:
                r_box.append((0,0,1,1,1))
        r_box=np.array(r_box)
        for m in range(int(np.max(left)+1)):
            x1,y1,x2,y2,size=l_box[m]
            if size==1:
                match.append(-1)
                continue
            left_object_m=np.where(left==m,1,-1)[x1:x2,y1:y2]
            x_matching=(np.min([np.ones_like(r_box[:,2])*x2,r_box[:,2]],0)-np.max([np.ones_like(r_box[:,0])*x1,r_box[:,0]],0))/(x2-x1)
            x_matching=np.where(x_matching>0.8,1,0)
            y_matching=np.min([(r_box[:,3]-r_box[:,1]),np.ones_like(r_box[:,0])*(y2-y1)],0)/np.max([(r_box[:,3]-r_box[:,1]), \
                       np.ones_like(r_box[:,0])*(y2-y1)],0)
            y_matching=np.where(y_matching>0.5,1,0)
            y_check=np.where(r_box[:,1]<=y2,1,0)
            y_matching=y_matching*y_check
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
                    overlap.append(-1)
            if np.max(overlap)>0:
                match.append(np.argmax(overlap))
            else:
                match.append(-1)
        match=np.array(match)
        pre.append([l_box,r_box,match])
        #min_d=np.array(np.max([np.where(match==-1,0,r_box[match,1]+l_box[:,1]-l_box[:,3]),np.zeros_like(match)],0))
        #max_d=np.array(np.min([np.where(match==-1,l_box[:,3],r_box[match,3]+l_box[:,3]-l_box[:,1]),min_d+300],0))
        variance_d=np.floor((l_box[:,3]-l_box[:,1])/10).astype(np.int)
        a=l_box[:,1]-r_box[match,1]
        b=l_box[:,3]-r_box[match,3]
        min_d=np.where(match==-1,0,np.max([np.min([a,b],0)-variance_d,np.zeros_like(match)],0))
        max_d=np.where(match==-1,192,np.min([np.max([a,b],0)+variance_d,np.ones_like(match)*192+min_d],0))
        min_d=np.where(l_box[:,3]-l_box[:,1]>900,0,min_d)
        max_d=np.where(l_box[:,3]-l_box[:,1]>900,l_box[:,2]-l_box[:,0],max_d)
        min_d=np.where(min_d>l_box[:,3],0,np.max([min_d,np.zeros_like(match)],0))
        max_d=np.where(max_d<=min_d,min_d+192,max_d)
        max_d=np.min([max_d,l_box[:,3]],0)
        pre2.append(np.array([min_d,max_d]))
        pre_match=np.array([pre,pre2])
        np.save(os.path.join(match_dir,left_files[i]),pre_match)
        print('thread:%d,doing:%d,time:%.3f' % (end/440,i,time.time()-start_time))


process = []
left_dir=r'/home/lidong/Documents/datasets/Driving/train_data_clean_pass/left/'
left_files=os.listdir(left_dir)
left_files.sort()
length=len(left_files)
start=[]
end=[]
p = Pool(thread_num)
for z in range(thread_num):
    start.append(z*length/10)
    end.append((z+1)*length/10)
for z in range(thread_num):
    p.apply_async(pre_matching, args=(start[z],end[z]))

p.close()
p.join()
#pre_matching(0,1)
print('end')
