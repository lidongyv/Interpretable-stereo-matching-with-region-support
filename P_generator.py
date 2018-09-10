# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-06-20 14:37:27
# @Last Modified by:   yulidong
# @Last Modified time: 2018-09-09 17:03:46
import cupy
import cv2
import numpy
import os
from python_pfm import *
from scipy import stats
import matplotlib.pyplot as plt
from skimage.filters import roberts, sobel, scharr, prewitt
#from skimage.feature import hog
from skimage import exposure
import time
division=20
p_left_image=r'/home/lidong/Documents/datasets/Driving/generation_test/left/'
p_right_image=r'/home/lidong/Documents/datasets/Driving/generation_test/right/'
p_disparity=r'/home/lidong/Documents/datasets/Driving/generation_test/disparity/'
p_sematnic=r'/home/lidong/Documents/datasets/Driving/generation_test/object/'
visual_dir=r'/home/lidong/Documents/datasets/Driving/generation_test/visual/'
n_l_image=os.listdir(p_left_image)
n_l_image.sort()
n_r_image=os.listdir(p_right_image)
n_r_image.sort()
n_disparity=os.listdir(p_disparity)
n_disparity.sort()
n_object=os.listdir(p_sematnic)
n_object.sort()
length=len(n_l_image)

for i in range(1):
    labels=[]
    start=time.time()
    l_image=np.array(cv2.imread(os.path.join(p_left_image,n_l_image[i]),0))
    r_image=np.array(cv2.imread(os.path.join(p_right_image,n_r_image[i])))
    disparity=np.array(readPFM(os.path.join(p_disparity,n_disparity[i]))[0])
    object=np.array(readPFM(os.path.join(p_sematnic,n_object[i]))[0])
    # #object statistics
    if i==i:
        j=0
        end=np.max(object)
        #print(end)
        while(j<end+1):
            count=np.sum(np.where(object==j))
            if count>0:
                if not j in (labels):
                    labels.append(j)
                #print(j,count)
            j=j+1
        #visual
        for k in range(len(labels)):
            object=np.where(object==labels[k],k,object)
        #object=object*np.floor((255/np.max(object)))
        #cv2.imwrite(os.path.join(visual_dir,n_l_image[i]),object)
#P1
image=disparity/np.max(disparity)
edge_roberts = roberts(image)
edge_sobel = sobel(image)
edge_prewitt = prewitt(image)
edge_scharr = scharr(image)
edge1=edge_roberts+edge_sobel+edge_prewitt+edge_scharr

image=object/np.max(object)
edge_roberts = roberts(image)
edge_sobel = sobel(image)
edge_prewitt = prewitt(image)
edge_scharr = scharr(image)
edge2=edge_roberts+edge_sobel+edge_prewitt+edge_scharr

# image=l_image/255
# edge_roberts = np.where(roberts(image)>0.05,1,0)
# edge_sobel = np.where(sobel(image)>0.05,1,0)
# edge_prewitt = np.where(prewitt(image)>0.05,1,0)
# edge_scharr = np.where(scharr(image)>0.05,1,0)
# edge3=edge_roberts+edge_sobel+edge_prewitt+edge_scharr
# #edge_roberts+edge_sobel+edge_prewitt+edge_scharr

# edge_roberts = roberts(image)
# edge_sobel = sobel(image)
# edge_prewitt = prewitt(image)
# edge_scharr = scharr(image)
# edge3=edge_roberts+edge_sobel+edge_prewitt+edge_scharr
##hog
#image=disparity/np.max(disparity)
#hd,hog_image=hog(image, orientations=8, pixels_per_cell=(8, 8),cells_per_block=(1, 1), visualize=True, multichannel=False,block_norm='L2-Hys')

#hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
#print(hog_image_rescaled)

edge1=np.where(edge1>0.1,1,0)
edge2=np.where(edge2>0.01,1,0)
#edge3=np.where(edge3>0.04,1,0)
edges=edge1+edge2
edges=np.where(edges>=1,1,0)
P1=edges
print(np.sum(np.where(P1>0,1,0))/(object.shape[0]*object.shape[1]))

# fig, ax = plt.subplots(ncols=3, sharex=True, sharey=True,
#                        figsize=(32, 16))
# a=np.sum(np.where(edges>0,1,0))/(edges.shape[0]*edges.shape[1])
# print(a)
# #edges=np.where(edges>=1,1,0)
# ax[0].imshow(edge1, cmap=plt.cm.gray)
# ax[1].imshow(edge2, cmap=plt.cm.gray)
# #ax[2].imshow (edge3, cmap=plt.cm.gray)
# ax[2].imshow(image, cmap=plt.cm.gray)
# for a in ax:
#     a.axis('off')

# plt.tight_layout()
# plt.show()
# print(np.max(object))
# image=np.reshape(object/np.max(object),[object.shape[0],object.shape[1],1])
# P2=np.concatenate([image,image,image],axis=2)

#P3
P2=object
image=P2
depth=disparity
segmentation=P2.astype(np.int32)
P4=np.zeros([segmentation.shape[0],segmentation.shape[1]])
P3=np.zeros([segmentation.shape[0],segmentation.shape[1]])
P3_v=np.zeros([segmentation.shape[0],segmentation.shape[1]])
for i in range(np.max(segmentation)+1):
    region=np.where(segmentation==i,depth,0)
    max=np.floor(np.max(region)).astype(np.int32)
    region=np.where(region==0,max+1,region)
    min=np.floor(np.min(region)).astype(np.int32)
    i_region=0
    for j in range(min,max+1):
        i_region+=1
        region_b=np.where(np.logical_and(region>=j,region<j+1),1.0,0.0)
        edge_size=np.sum(region_b)
        box=np.argwhere(region_b==1)
        if box.shape[0]>0:       
            P3+=region_b*i_region
            #P4
            #print(box.shape[0])
            x_min=np.min(box[:,0])
            x_max=np.max(box[:,0])
            y_min=np.min(box[:,1])
            y_max=np.max(box[:,1])
            b_size=(x_max-x_min)*(y_max-y_min)
            if b_size>0:
                image=region_b[x_min:x_max,y_min:y_max]
                edge_sobel = sobel(image)
                edges=edge_sobel
                edges=np.where(edges>0.01,1,0)
                if np.sum(edges)<edge_size:
                    region_b[x_min:x_max,y_min:y_max]=edges
                    P4+=region_b
                else:
                    for m in range(x_min,x_max+1):
                        line=np.argwhere(region_b[m,y_min:y_max]>0)
                        for n in range(np.ceil(line.shape[0]/division).astype(np.int32)):
                            P4[m,y_min+line[n,0]]=1
                        if line.shape[0]>0:
                            P4[m,y_min+line[-1,0]]=1                       

            else:
                #get 10% pixels as representive pixels
                for m in range(((y_max-y_min)/division).astype(np.int32)):
                    P4[x_max,np.floor(y_min+m*division).astype(np.int32)]=1
                P4[x_max,y_max]=1
    P3_v=P3_v+np.where(segmentation==i,P3,0)*255/i_region
    #print(i_region)
    #P3=np.where(segmentation==i,255/i_region*P3,P3)
#print(np.min(depth))
#P3=np.where(depth<1,255,0)
# fig, ax = plt.subplots(nrows=2,ncols=1, sharex=True, sharey=True,
#                        figsize=(32, 64))
# image=P2
# image=np.reshape(image/np.max(image),[object.shape[0],object.shape[1],1])
# P3_1=np.reshape(P3/np.max(P3),[object.shape[0],object.shape[1],1])
# P3_2=np.reshape(P2/np.max(P2),[object.shape[0],object.shape[1],1])
# P3_3=np.reshape(P1/np.max(P1),[object.shape[0],object.shape[1],1])
# P3_0=np.concatenate([P3_1,P3_2,P3_3],axis=2)
#print(depth)

#P4

fig, ax = plt.subplots(nrows=5,ncols=1, sharex=True, sharey=True,
                        figsize=(32, 64))
# # for i in range(np.max(P2).astype(np.int32)+1):
# #     region=np.where(P2==i,P3,0)
# #     for j in range(1,np.max(region).astype(np.int32)+1):
# #         image=np.where(region==j,1.0,0.0)
# #         size=np.sum(np.where(region>0,1,0))
# #         #print(np.max(region))
# #         edge_roberts = roberts(image)
# #         edge_sobel = sobel(image)
# #         edge_prewitt = prewitt(image)
# #         edge_scharr = scharr(image)
# #         edges=edge_roberts+edge_sobel+edge_prewitt+edge_scharr
# #         edges=np.where(edges>0.01,1,0)   
# #         if np.sum(edges)/size<0.9:
# #         #ax[i].imshow(edge2, cmap=plt.cm.gray)
# #             P4+=edges
P4=np.where(P4>0,1,0)
print(np.sum(np.where(P4>0,1,0))/(P4.shape[0]*P4.shape[1]))
overlap=P4-P1
overlap=np.where(overlap>0,1,0)

ax[0].imshow(P1, cmap=plt.cm.gray)
ax[1].imshow(P2, cmap=plt.cm.gray)
ax[2].imshow(P3_v, cmap=plt.cm.gray)
ax[3].imshow(P4, cmap=plt.cm.gray)
ax[4].imshow(overlap, cmap=plt.cm.gray)


    # cv2.imwrite(os.path.join(visual_dir,n_l_image[i]),object)
    # #object statistics
    # if i==i:
    #     j=0
    #     end=np.max(object)
    #     #print(end)
    #     while(j<end+1):
    #         count=np.sum(np.where(object==j))
    #         if count>0:
    #             if not j in (labels):
    #                 labels.append(j)
    #             #print(j,count)
    #         j=j+1
    #     # #visual
    #     # for k in range(len(labels)):
    #     #     object=np.where(object==labels[k],k,object)
    #     # object=object*np.floor((255/np.max(object)))
    #     # cv2.imwrite(os.path.join(visual_dir,n_l_image[i]),object)
    #     print(time.time()-start)
    #     print(labels)   
    #     print(len(labels))