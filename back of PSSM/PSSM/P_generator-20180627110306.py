# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-06-20 14:37:27
# @Last Modified by:   yulidong
# @Last Modified time: 2018-06-27 11:02:47
import cupy
import cv2
import numpy
import os
from python_pfm import *
from scipy import stats
import matplotlib.pyplot as plt
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.feature import hog
from skimage import exposure
import time
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
image=disparity/np.max(disparity)
fd,hog_image=hog(image, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualise=True, multichannel=True,block_norm='L2-Hys')

hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
fig, ax = plt.subplots(ncols=3, sharex=True, sharey=True,
                       figsize=(32, 16))
edge1=np.where(edge1>0.1,1,0)
edge2=np.where(edge2>0.01,1,0)
#edge3=np.where(edge3>0.04,1,0)
edges=edge1+edge2
edges=np.where(edges>=1,1,0)
a=np.sum(np.where(edges>0,1,0))/(edges.shape[0]*edges.shape[1])
print(a)
#edges=np.where(edges>=1,1,0)
ax[0].imshow(edge1, cmap=plt.cm.gray)
ax[1].imshow(edge2, cmap=plt.cm.gray)
#ax[2].imshow (edge3, cmap=plt.cm.gray)
ax[2].imshow(hog_image_rescaled, cmap=plt.cm.gray)
for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()










































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