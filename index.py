# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-08-31 17:25:26
# @Last Modified by:   yulidong
# @Last Modified time: 2018-08-31 23:34:11
import torch
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from multiprocessing import Process,Lock
from multiprocessing import Pool
thread_num=1
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
    box_dir=r'/home/lidong/Documents/datasets/Driving/train_data_clean_pass/match/'
    index_dir=r'/home/lidong/Documents/datasets/Driving/train_data_clean_pass/index/'
    left_files=os.listdir(left_dir)
    left_files.sort()

    for i in range(int(start),int(end)):
        P=np.load(os.path.join(left_dir,left_files[i]))[...,7:]
        P1=P[...,0]
        P2=P[...,3]-P1
        P3=P[...,1]
        P4=P[...,2]
        match=np.load(os.path.join(box_dir,left_files[i]))
        l_box=match[0,0][0]
        l_d=match[1,0]
        s_match_x=[]
        s_match_y=[]
        l_match_x=[]
        l_match_y=[]
        match_x=[]
        match_y=[]
        plane=[]
        plane_num=[]
        s_plane=[]
        s_plane_num=[]
        l_plane=[]
        l_plane_num=[]
        start_time=time.time()
        for m in range(int(np.max(P2)+1)):
            x1,y1,x2,y2,size=l_box[0,m]
            min_d=l_d[0,0,m]
            max_d=l_d[0,1,m]
            ojbect=np.where(P3==i,1,0)
            s_pxiel=ojbect*P1
            l_pixel=ojbect*P2
            shift=np.arange(min_d,max_d+1)
            shift=np.tile(shift,max_d-min_d)
            s_match_x_t=s_pixel.nonzero()[0].repeat(max_d-min_d)
            s_match_y_t=s_pixel.nonzero()[1].repeat(max_d-min_d)-shift
            l_match_x_t=l_pixel.nonzero()[0].repeat(max_d-min_d)
            l_match_y_t=l_pixel.nonzero()[1].repeat(max_d-min_d)-shift
            s_match_x=np.concatenate([s_match_x,s_match_x_t])
            s_match_y=np.concatenate([s_match_y,s_match_y_t])
            l_match_x=np.concatenate([l_match_x,l_match_x_t])
            l_match_y=np.concatenate([l_match_y,l_match_y_t])
            ojbect_r=np.where(P3==i,P4,0)
            for n in range(1,np.max(ojbect_r)):
                plane_t=np.where(ojbect_r==n,1,0).nonzero()
                plane_num.append(plane_t[0].shape[0])
                plane.append(plane_t)
                s_plane_t=np.where(ojbect_r==n,1,0)*P1
                s_plane_t=s_plane_t.nonzero()
                s_plane_num.append(s_plane_t[0].shape[0])
                s_plane.append(s_plane_t)
                l_plane_t=np.where(ojbect_r==n,1,0)*P2
                l_plane_t=l_plane_t.nonzero()
                l_plane_num.append(l_plane_t[0].shape[0])
                l_plane.append(l_plane_t)




        np.save(os.path.join(match_dir,left_files[i]),pre_match)
        print('thread:%d,doing:%d,time:%.3f' % (end/440,i,time.time()-start_time))


# process = []
# left_dir=r'/home/lidong/Documents/datasets/Driving/train_data_clean_pass/left/'
# left_files=os.listdir(left_dir)
# left_files.sort()
# length=len(left_files)
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
