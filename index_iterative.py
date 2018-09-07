# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-08-31 17:25:26
# @Last Modified by:   yulidong
# @Last Modified time: 2018-09-07 20:35:13
import torch
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
    #print(start)
    left_dir=r'/home/lidong/Documents/datasets/Driving/train_data_clean_pass/left/'
    box_dir=r'/home/lidong/Documents/datasets/Driving/train_data_clean_pass/match/'
    index_dir=r'/home/lidong/Documents/datasets/Driving/train_data_clean_pass/matching/'
    aggregation_dir=r'/home/lidong/Documents/datasets/Driving/train_data_clean_pass/aggregation/'
    left_files=os.listdir(left_dir)
    left_files.sort()
    
    for i in range(int(start),int(end)):
        start_time=time.time()
        
        P=np.load(os.path.join(left_dir,left_files[i]))[...,7:]
        P1=P[...,0]
        P2=np.where(P[...,3]-P1>0,1,0)
        P3=P[...,1]
        P4=P[...,2]

        match=np.load(os.path.join(box_dir,left_files[i]))
        l_box=match[0,0][0]
        l_d=match[1,0]
        s_match_x=[]
        s_match_y=[]
        s_shfit=[]
        s_repeat=[]
        l_match_x=[]
        l_match_y=[]
        l_shfit=[]
        l_repeat=[]
        match_x=[]
        match_y=[]


        plane=[]
        plane_num=[]
        s_plane=[]
        s_plane_num=[]
        l_plane=[]
        l_plane_num=[]



        #start_time=time.time()
        for m in range(int(np.max(P3)+1)):
            x1,y1,x2,y2,size=l_box[m]
            min_d=l_d[0,m]
            if min_d==0:
                min_d=1
            max_d=l_d[1,m]
            if min_d>300:
                min_d=1
                max_d=192
            min_d=1
            max_d=192
            object=P3[x1:x2,y1:y2]
            object=np.where(object==m,1,0)
            s_pixel=object*P1[x1:x2,y1:y2]
            l_pixel=object*P2[x1:x2,y1:y2]

            shift=np.arange(min_d,max_d+1)
            s_match_x_t=s_pixel.nonzero()[0]
            s_match_y_t=s_pixel.nonzero()[1]
            if s_match_x_t.shape[0]>0:
                s_shfit.append(np.array(shift))
                s_match_x.append(np.array(s_match_x_t))
                s_match_y.append(np.array(s_match_y_t))
                s_repeat.append(np.array([shift.shape[0],s_match_x_t.shape[0]]))
            else:
                s_shfit.append(np.array([0]))
                s_match_x.append(np.array([-1]))
                s_match_y.append(np.array([-1]))
                s_repeat.append(np.array([1,1]))                

            shift=np.arange(min_d,max_d+1)
            l_match_x_t=l_pixel.nonzero()[0]
            l_match_y_t=l_pixel.nonzero()[1]
            if l_match_x_t.shape[0]>0:
                l_shfit.append(np.array(shift))
                l_match_x.append(np.array(l_match_x_t))
                l_match_y.append(np.array(l_match_y_t))
                l_repeat.append(np.array([shift.shape[0],l_match_x_t.shape[0]]))
            else:
                l_shfit.append(np.array([0]))
                l_match_x.append(np.array([-1]))
                l_match_y.append(np.array([-1]))
                l_repeat.append(np.array([1,1]))
            object_r=object
            object_r=np.where(object_r>0,P4[x1:x2,y1:y2],0)
            #print(np.max(object_r))
            plane_0=[]
            plane_num_0=[]

            s_plane_0=[]
            s_plane_num_0=[]

            l_plane_0=[]
            l_plane_num_0=[]
            for n in range(1,int(np.max(object_r)+1)):
                plane_t=np.where(object_r==n,1,0).nonzero()
                plane_num_0.append(np.array(plane_t[0].shape[0]))
                plane_0.append(np.array([plane_t[0],plane_t[1]]))
                s_plane_t=np.where(object_r==n,1,0)*P1[x1:x2,y1:y2]
                s_plane_t=s_plane_t.nonzero()
                if s_plane_t[0].shape[0]>0:
                    s_plane_num_0.append(np.array(s_plane_t[0].shape[0]))
                    s_plane_0.append(np.array([s_plane_t[0],s_plane_t[1]]))
                else:
                    s_plane_num_0.append(np.array(0))
                    s_plane_0.append(np.array([-1,-1]))
                l_plane_t=np.where(object_r==n,1,0)*P2[x1:x2,y1:y2]
                l_plane_t=l_plane_t.nonzero()
                if l_plane_t[0].shape[0]>0:
                    l_plane_num_0.append(np.array(l_plane_t[0].shape[0]))
                    l_plane_0.append(np.array([l_plane_t[0],l_plane_t[1]]))
                else:
                    l_plane_num_0.append(np.array(0))
                    l_plane_0.append(np.array([-1,-1]))
            plane.append(plane_0)
            plane_num.append(np.array(plane_num_0))
            s_plane.append(s_plane_0)
            s_plane_num.append(np.array(s_plane_num_0))
            l_plane.append(l_plane_0)
            l_plane_num.append(np.array(l_plane_num_0))                 

        match=[s_match_x,
                s_match_y,
                s_shfit,
                s_repeat,
                l_match_x,
                l_match_y,
                l_shfit,
                l_repeat]

        aggregation=[plane,plane_num,s_plane,s_plane_num,l_plane,l_plane_num]

        #print(time.time()-start_time)
        np.save(os.path.join(index_dir,left_files[i]),match)
        np.save(os.path.join(aggregation_dir,left_files[i]),aggregation)
        print('thread:%d,doing:%d,time:%.3f' % (end/440,i,time.time()-start_time))
        # fig, ax = plt.subplots(nrows=5,ncols=1, sharex=True, sharey=True,figsize=(32, 64))
        # ax[0].imshow(P1, cmap=plt.cm.gray)
        # ax[1].imshow(P2, cmap=plt.cm.gray)
        # ax[2].imshow(P3, cmap=plt.cm.gray)
        # ax[3].imshow(P4, cmap=plt.cm.gray)



process = []
left_dir=r'/home/lidong/Documents/datasets/Driving/train_data_clean_pass/left/'
left_files=os.listdir(left_dir)
left_files.sort()
length=len(left_files)
# start=[]
# end=[]
# p = Pool(thread_num)
# for z in range(thread_num):
#     start.append(z*length/10)
#     end.append((z+1)*length/10)
# for z in range(thread_num):
#     p.apply_async(pre_matching, args=(start[z],end[z]))

# p.close()
# p.join()
pre_matching(0,1)
print('end')
