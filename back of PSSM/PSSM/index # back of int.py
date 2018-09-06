# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-08-31 17:25:26
# @Last Modified by:   yulidong
# @Last Modified time: 2018-09-03 22:45:32
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
    print(start)
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
        l_match_x=[]
        l_match_y=[]
        l_shfit=[]
        match_x=[]
        match_y=[]


        plane=[]
        plane_num=[]
        s_plane=[]
        s_plane_num=[]
        l_plane=[]
        l_plane_num=[]
        plane_0=[]
        plane_num_0=[]
        plane_1=[]
        plane_num_1=[]
        plane_2=[]
        plane_num_2=[]
        plane_3=[]
        plane_num_3=[]
        s_plane_0=[]
        s_plane_num_0=[]
        s_plane_1=[]
        s_plane_num_1=[]
        s_plane_2=[]
        s_plane_num_2=[]
        s_plane_3=[]
        s_plane_num_3=[]
        l_plane_0=[]
        l_plane_num_0=[]
        l_plane_1=[]
        l_plane_num_1=[]
        l_plane_2=[]
        l_plane_num_2=[]
        l_plane_3=[]
        l_plane_num_3=[]

        #start_time=time.time()
        for m in range(int(np.max(P3)+1)):
            x1,y1,x2,y2,size=l_box[m]
            min_d=l_d[0,m]
            max_d=l_d[1,m]
            object=np.where(P3==m,1,0)
            s_pixel=object*P1
            l_pixel=object*P2
            shift=np.arange(min_d,max_d+1)
            shift=np.tile(shift,s_pixel.nonzero()[0].shape[0])
            s_match_x_t=s_pixel.nonzero()[0].repeat(max_d-min_d+1)
            s_match_y_t=s_pixel.nonzero()[1].repeat(max_d-min_d+1)
            s_shfit=np.concatenate([s_shfit,shift])
            s_match_x=np.concatenate([s_match_x,s_match_x_t])
            s_match_y=np.concatenate([s_match_y,s_match_y_t])
            shift=np.arange(min_d,max_d+1)
            shift=np.tile(shift,l_pixel.nonzero()[0].shape[0])
            l_match_x_t=l_pixel.nonzero()[0].repeat(max_d-min_d+1)
            l_match_y_t=l_pixel.nonzero()[1].repeat(max_d-min_d+1)
            l_shfit=np.concatenate([l_shfit,shift])
            l_match_x=np.concatenate([l_match_x,l_match_x_t])
            l_match_y=np.concatenate([l_match_y,l_match_y_t])
            object_r=object[x1:x2,y1:y2]
            object_r=np.where(object_r>0,P4[x1:x2,y1:y2],0)
            #print(np.max(object_r))

            for n in range(1,int(np.max(object_r)+1)):

                plane_t=np.where(object_r==n,1,0).nonzero()
                if plane_t[0].shape[0]<1600:
                    plane_num_0.append(plane_t[0].shape[0])
                    plane_0.append([plane_t[0]+x1,plane_t[1]+y1])
                else:
                    if plane_t[0].shape[0]<6400:
                        plane_num_1.append(plane_t[0].shape[0])
                        plane_1.append([plane_t[0]+x1,plane_t[1]+y1])
                    else:
                        if plane_t[0].shape[0]<25600:
                            plane_num_2.append(plane_t[0].shape[0])
                            plane_2.append([plane_t[0]+x1,plane_t[1]+y1])
                        else:
                            plane_num_3.append(plane_t[0].shape[0])
                            plane_3.append([plane_t[0]+x1,plane_t[1]+y1])                            

                s_plane_t=np.where(object_r==n,1,0)*P1[x1:x2,y1:y2]
                s_plane_t=s_plane_t.nonzero()
                if s_plane_t[0].shape[0]<1600:
                    s_plane_num_0.append(s_plane_t[0].shape[0])
                    s_plane_0.append([s_plane_t[0]+x1,s_plane_t[1]+y1])
                else:
                    if s_plane_t[0].shape[0]<6400:
                        s_plane_num_1.append(s_plane_t[0].shape[0])
                        s_plane_1.append([s_plane_t[0]+x1,s_plane_t[1]+y1])
                    else:
                        if s_plane_t[0].shape[0]<25600:
                            s_plane_num_2.append(s_plane_t[0].shape[0])
                            s_plane_2.append([s_plane_t[0]+x1,s_plane_t[1]+y1])
                        else:
                            s_plane_num_3.append(s_plane_t[0].shape[0])
                            s_plane_3.append([s_plane_t[0]+x1,s_plane_t[1]+y1])   
                l_plane_t=np.where(object_r==n,1,0)*P2[x1:x2,y1:y2]
                l_plane_t=l_plane_t.nonzero()
                if l_plane_t[0].shape[0]<1600:
                    l_plane_num_0.append(l_plane_t[0].shape[0])
                    l_plane_0.append([l_plane_t[0]+x1,l_plane_t[1]+y1])
                else:
                    if l_plane_t[0].shape[0]<6400:
                        l_plane_num_1.append(l_plane_t[0].shape[0])
                        l_plane_1.append([l_plane_t[0]+x1,l_plane_t[1]+y1])
                    else:
                        if l_plane_t[0].shape[0]<25600:
                            l_plane_num_2.append(l_plane_t[0].shape[0])
                            l_plane_2.append([l_plane_t[0]+x1,l_plane_t[1]+y1])
                        else:
                            l_plane_num_3.append(l_plane_t[0].shape[0])
                            l_plane_3.append([l_plane_t[0]+x1,l_plane_t[1]+y1])   
        #division of 1600,6400,25600
        plane_x_0=[]
        plane_y_0=[]
        plane_x_1=[]
        plane_y_1=[]
        plane_x_2=[]
        plane_y_2=[]
        plane_x_3=[]
        plane_y_3=[]
        for n in range(len(plane_num_0)):
            plane_x_0=np.concatenate([plane_x_0,np.concatenate([plane_0[n][0],-np.ones(1600-plane_num_0[n])])])
            plane_y_0=np.concatenate([plane_y_0,np.concatenate([plane_0[n][1],-np.ones(1600-plane_num_0[n])])])
        for n in range(len(plane_num_1)):
            plane_x_1=np.concatenate([plane_x_1,np.concatenate([plane_1[n][0],-np.ones(6400-plane_num_1[n])])])
            plane_y_1=np.concatenate([plane_y_1,np.concatenate([plane_1[n][1],-np.ones(6400-plane_num_1[n])])])
        for n in range(len(plane_num_2)):
            plane_x_2=np.concatenate([plane_x_2,np.concatenate([plane_2[n][0],-np.ones(25600-plane_num_2[n])])])
            plane_y_2=np.concatenate([plane_y_2,np.concatenate([plane_2[n][1],-np.ones(25600-plane_num_2[n])])])
        for n in range(len(plane_num_3)):
            plane_x_3=np.concatenate([plane_x_3,np.concatenate([plane_3[n][0],-np.ones(np.max(plane_num_3)-plane_num_3[n])])])
            plane_y_3=np.concatenate([plane_y_3,np.concatenate([plane_3[n][1],-np.ones(np.max(plane_num_3)-plane_num_3[n])])])

        s_plane_x_0=[]
        s_plane_y_0=[]
        s_plane_x_1=[]
        s_plane_y_1=[]
        s_plane_x_2=[]
        s_plane_y_2=[]
        s_plane_x_3=[]
        s_plane_y_3=[]
        for n in range(len(s_plane_num_0)):
            s_plane_x_0=np.concatenate([s_plane_x_0,np.concatenate([s_plane_0[n][0],-np.ones(1600-s_plane_num_0[n])])])
            s_plane_y_0=np.concatenate([s_plane_y_0,np.concatenate([s_plane_0[n][1],-np.ones(1600-s_plane_num_0[n])])])
        for n in range(len(s_plane_num_1)):
            s_plane_x_1=np.concatenate([s_plane_x_1,np.concatenate([s_plane_1[n][0],-np.ones(6400-s_plane_num_1[n])])])
            s_plane_y_1=np.concatenate([s_plane_y_1,np.concatenate([s_plane_1[n][1],-np.ones(6400-s_plane_num_1[n])])])
        for n in range(len(s_plane_num_2)):
            s_plane_x_2=np.concatenate([s_plane_x_2,np.concatenate([s_plane_2[n][0],-np.ones(25600-s_plane_num_2[n])])])
            s_plane_y_2=np.concatenate([s_plane_y_2,np.concatenate([s_plane_2[n][1],-np.ones(25600-s_plane_num_2[n])])])
        for n in range(len(s_plane_num_3)):
            s_plane_x_3=np.concatenate([s_plane_x_3,np.concatenate([s_plane_3[n][0],-np.ones(np.max(s_plane_num_3)-s_plane_num_3[n])])])
            s_plane_y_3=np.concatenate([s_plane_y_3,np.concatenate([s_plane_3[n][1],-np.ones(np.max(s_plane_num_3)-s_plane_num_3[n])])])

        l_plane_x_0=[]
        l_plane_y_0=[]
        l_plane_x_1=[]
        l_plane_y_1=[]
        l_plane_x_2=[]
        l_plane_y_2=[]
        l_plane_x_3=[]
        l_plane_y_3=[]
        for n in range(len(l_plane_num_0)):
            l_plane_x_0=np.concatenate([l_plane_x_0,np.concatenate([l_plane_0[n][0],-np.ones(1600-l_plane_num_0[n])])])
            l_plane_y_0=np.concatenate([l_plane_y_0,np.concatenate([l_plane_0[n][1],-np.ones(1600-l_plane_num_0[n])])])
        for n in range(len(l_plane_num_1)):
            l_plane_x_1=np.concatenate([l_plane_x_1,np.concatenate([l_plane_1[n][0],-np.ones(6400-l_plane_num_1[n])])])
            l_plane_y_1=np.concatenate([l_plane_y_1,np.concatenate([l_plane_1[n][1],-np.ones(6400-l_plane_num_1[n])])])
        for n in range(len(l_plane_num_2)):
            l_plane_x_2=np.concatenate([l_plane_x_2,np.concatenate([l_plane_2[n][0],-np.ones(25600-l_plane_num_2[n])])])
            l_plane_y_2=np.concatenate([l_plane_y_2,np.concatenate([l_plane_2[n][1],-np.ones(25600-l_plane_num_2[n])])])
        for n in range(len(l_plane_num_3)):
            l_plane_x_3=np.concatenate([l_plane_x_3,np.concatenate([l_plane_3[n][0],-np.ones(np.max(l_plane_num_3)-l_plane_num_3[n])])])
            l_plane_y_3=np.concatenate([l_plane_y_3,np.concatenate([l_plane_3[n][1],-np.ones(np.max(l_plane_num_3)-l_plane_num_3[n])])])
        match=[s_match_x,
                s_match_y,
                s_shfit,
                l_match_x,
                l_match_y,
                l_shfit]
        plane=[
        plane_x_0,
        plane_y_0,
        plane_x_1,
        plane_y_1,
        plane_x_2,
        plane_y_2,
        plane_x_3,
        plane_y_3,
        np.array(plane_num_0),
        np.array(plane_num_1),
        np.array(plane_num_2),
        np.array(plane_num_3)
        ]

        s_plane=[
        s_plane_x_0,
        s_plane_y_0,
        s_plane_x_1,
        s_plane_y_1,
        s_plane_x_2,
        s_plane_y_2,
        s_plane_x_3,
        s_plane_y_3,
        np.array(s_plane_num_0),
        np.array(s_plane_num_1),
        np.array(s_plane_num_2),
        np.array(s_plane_num_3)
        ]

        l_plane=[
        l_plane_x_0,
        l_plane_y_0,
        l_plane_x_1,
        l_plane_y_1,
        l_plane_x_2,
        l_plane_y_2,
        l_plane_x_3,
        l_plane_y_3,
        np.array(l_plane_num_0),
        np.array(l_plane_num_1),
        np.array(l_plane_num_2),
        np.array(l_plane_num_3)
        ]
        aggregation=[plane,s_plane,l_plane]

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
