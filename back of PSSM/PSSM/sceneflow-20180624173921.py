#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 14:52:01 2018

@author: lidong
"""
import argparse
import os
import sys

from python_pfm import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import disk
from skimage.data import camera
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.filters.rank import median

from skimage.segmentation import felzenszwalb
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import feature

pathr=[
        r'/home/lidong/Documents/SceneFlow/region/15mm_focallength/scene_forwards/slow/right',
        r'/home/lidong/Documents/SceneFlow/region/15mm_focallength/scene_forwards/slow/left',
        r'/home/lidong/Documents/SceneFlow/region/15mm_focallength/scene_forwards/fast/right',
        r'/home/lidong/Documents/SceneFlow/region/15mm_focallength/scene_forwards/fast/left',
        r'/home/lidong/Documents/SceneFlow/region/15mm_focallength/scene_backwards/slow/right',
        r'/home/lidong/Documents/SceneFlow/region/15mm_focallength/scene_backwards/slow/left',
        r'/home/lidong/Documents/SceneFlow/region/15mm_focallength/scene_backwards/fast/right',
        r'/home/lidong/Documents/SceneFlow/region/15mm_focallength/scene_backwards/fast/left',                
        r'/home/lidong/Documents/SceneFlow/region/35mm_focallength/scene_forwards/slow/right',
        r'/home/lidong/Documents/SceneFlow/region/35mm_focallength/scene_forwards/slow/left',
        r'/home/lidong/Documents/SceneFlow/region/35mm_focallength/scene_forwards/fast/right',
        r'/home/lidong/Documents/SceneFlow/region/35mm_focallength/scene_forwards/fast/left',
        r'/home/lidong/Documents/SceneFlow/region/35mm_focallength/scene_backwards/slow/right',
        r'/home/lidong/Documents/SceneFlow/region/35mm_focallength/scene_backwards/slow/left',
        r'/home/lidong/Documents/SceneFlow/region/35mm_focallength/scene_backwards/fast/right',
        r'/home/lidong/Documents/SceneFlow/region/35mm_focallength/scene_backwards/fast/left'      
      ]
path=[
        r'/home/lidong/Documents/SceneFlow/disparity/15mm_focallength/scene_forwards/slow/right',
        r'/home/lidong/Documents/SceneFlow/disparity/15mm_focallength/scene_forwards/slow/left',
        r'/home/lidong/Documents/SceneFlow/disparity/15mm_focallength/scene_forwards/fast/right',
        r'/home/lidong/Documents/SceneFlow/disparity/15mm_focallength/scene_forwards/fast/left',
        r'/home/lidong/Documents/SceneFlow/disparity/15mm_focallength/scene_backwards/slow/right',
        r'/home/lidong/Documents/SceneFlow/disparity/15mm_focallength/scene_backwards/slow/left',
        r'/home/lidong/Documents/SceneFlow/disparity/15mm_focallength/scene_backwards/fast/right',
        r'/home/lidong/Documents/SceneFlow/disparity/15mm_focallength/scene_backwards/fast/left',                
        r'/home/lidong/Documents/SceneFlow/disparity/35mm_focallength/scene_forwards/slow/right',
        r'/home/lidong/Documents/SceneFlow/disparity/35mm_focallength/scene_forwards/slow/left',
        r'/home/lidong/Documents/SceneFlow/disparity/35mm_focallength/scene_forwards/fast/right',
        r'/home/lidong/Documents/SceneFlow/disparity/35mm_focallength/scene_forwards/fast/left',
        r'/home/lidong/Documents/SceneFlow/disparity/35mm_focallength/scene_backwards/slow/right',
        r'/home/lidong/Documents/SceneFlow/disparity/35mm_focallength/scene_backwards/slow/left',
        r'/home/lidong/Documents/SceneFlow/disparity/35mm_focallength/scene_backwards/fast/right',
        r'/home/lidong/Documents/SceneFlow/disparity/35mm_focallength/scene_backwards/fast/left'      
      ]
for p in range(len(path)):
    file=os.listdir(path[p])
    #file=[file[0]]
    #file=['0074.pfm']
    for f in range(len(file)):
        disparity=readPFM(os.path.join(path[p],file[f]))[0]
        image = np.uint8(np.round(disparity/270*255)) 
        img=cv2.medianBlur(image,9)
        img=img_as_float(img) 
        
        segments_fz = felzenszwalb(img, scale=200, sigma=0.65, min_size=16)
        
        segs=mark_boundaries(img, segments_fz)
        #plt.imshow(mark_boundaries(img, segments_fz))
        means=[]
        for i in range(np.max(segments_fz)):
            mean=np.sum(np.where(segments_fz==i,image,0))/len(np.where(segments_fz==i)[0])
            means.append(mean)
        means=np.array(means)
        mean=np.argsort(-means)
        meanb=np.sort(-means)
        sets=[]
        
        tset=[mean[0]]
        for i in range(len(meanb)-1):
            if meanb[i+1]-meanb[i]<=1:
                tset.append(mean[i+1])
            else:
                sets.append(tset)
                tset=[mean[i+1]]
                
        mset=[]
        for i in range(32):
            if i <len(sets):
                mset.append(sets[i])
            else:
                mset.append([])
        if len(sets)>32:
            for i in range(len(sets)-32):
                mset[31].append(sets[32+i][0])
        results=np.zeros(img.shape)
        for i in range(len(mset)):
            for j in range(len(mset[i])):
                results=np.where(segments_fz==mset[i][j],i,results)
        #cv2.imwrite(r'/home/lidong/Documents/LRS/a.png',np.round(segments_fz/np.max(segments_fz)*255).astype('uint8'))
        #cv2.imwrite(r'/home/lidong/Documents/LRS/b.png',np.round(results*255/len(sets)).astype('uint8'))
        
        gray = np.uint8(np.round(disparity/np.max(disparity)*255))
        img_medianBlur=cv2.medianBlur(gray,7)  
        
        #edges
        canny = cv2.Canny(img_medianBlur, 5, 30,apertureSize=3)
        canny = np.uint8(np.absolute(canny))
        sobelx = cv2.Sobel(img_medianBlur, cv2.CV_64F, 1, 0)
        sobely = cv2.Sobel(img_medianBlur, cv2.CV_64F, 0, 1)
        
        sobelx = np.uint8(np.absolute(sobelx))
        sobely = np.uint8(np.absolute(sobely))
        sobelcombine = cv2.bitwise_or(sobelx,sobely)
        #cv2.imwrite(r'/home/lidong/Documents/LRS/a2.png',canny)
        #cv2.imwrite(r'/home/lidong/Documents/LRS/b2.png',sobelcombine)
        #cv2.imwrite(r'/home/lidong/Documents/LRS/disparity.png',gray)
        ground=[]
        ground.append(np.round(results*255/len(sets)).astype('uint8'))
        #ground.append(np.round(segments_fz/np.max(segments_fz)*255).astype('uint8'))
        #ground.append(sobelcombine)
        ground.append(np.round(segments_fz/np.max(segments_fz)*255).astype('uint8')+sobelcombine)
        edges=np.where(canny==255,canny,sobelcombine)
        ground.append(edges)
        ground=np.transpose(np.array(ground),(1,2,0))
        name=file[f].split('.')[0]
        imagename=name+'.png'
        print(os.path.join(pathr[p],imagename))
        cv2.imwrite(os.path.join(pathr[p],imagename),ground)
        #cv2.imwrite(r'/home/lidong/Documents/LRS/r1.png',ground[:,:,0])
        #cv2.imwrite(r'/home/lidong/Documents/LRS/r2.png',ground[:,:,1])
        #cv2.imwrite(r'/home/lidong/Documents/LRS/r3.png',ground[:,:,2])