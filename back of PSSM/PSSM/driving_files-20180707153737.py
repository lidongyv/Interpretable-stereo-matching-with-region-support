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

pathl=[

        r'/home/lidong/Documents/datasets/Driving/frames_finalpass/15mm_focallength/scene_forwards/slow/left',

        r'/home/lidong/Documents/datasets/Driving/frames_finalpass/15mm_focallength/scene_forwards/fast/left',

        r'/home/lidong/Documents/datasets/Driving/frames_finalpass/15mm_focallength/scene_backwards/slow/left',

        r'/home/lidong/Documents/datasets/Driving/frames_finalpass/15mm_focallength/scene_backwards/fast/left',

        r'/home/lidong/Documents/datasets/Driving/frames_finalpass/35mm_focallength/scene_forwards/slow/left',

        r'/home/lidong/Documents/datasets/Driving/frames_finalpass/35mm_focallength/scene_forwards/fast/left',

        r'/home/lidong/Documents/datasets/Driving/frames_finalpass/35mm_focallength/scene_backwards/slow/left',

        r'/home/lidong/Documents/datasets/Driving/frames_finalpass/35mm_focallength/scene_backwards/fast/left'      
      ]
pathl.sort()
pathr=[
        r'/home/lidong/Documents/datasets/Driving/frames_finalpass/15mm_focallength/scene_forwards/slow/right',

        r'/home/lidong/Documents/datasets/Driving/frames_finalpass/15mm_focallength/scene_forwards/fast/right',

        r'/home/lidong/Documents/datasets/Driving/frames_finalpass/15mm_focallength/scene_backwards/slow/right',

        r'/home/lidong/Documents/datasets/Driving/frames_finalpass/15mm_focallength/scene_backwards/fast/right',

        r'/home/lidong/Documents/datasets/Driving/frames_finalpass/35mm_focallength/scene_forwards/slow/right',

        r'/home/lidong/Documents/datasets/Driving/frames_finalpass/35mm_focallength/scene_forwards/fast/right',

        r'/home/lidong/Documents/datasets/Driving/frames_finalpass/35mm_focallength/scene_backwards/slow/right',

        r'/home/lidong/Documents/datasets/Driving/frames_finalpass/35mm_focallength/scene_backwards/fast/right',
     
      ]
pathr.sort()
pathd=[

        r'/home/lidong/Documents/datasets/Driving/disparity/15mm_focallength/scene_forwards/slow/left',

        r'/home/lidong/Documents/datasets/Driving/disparity/15mm_focallength/scene_forwards/fast/left',

        r'/home/lidong/Documents/datasets/Driving/disparity/15mm_focallength/scene_backwards/slow/left',

        r'/home/lidong/Documents/datasets/Driving/disparity/15mm_focallength/scene_backwards/fast/left',

        r'/home/lidong/Documents/datasets/Driving/disparity/35mm_focallength/scene_forwards/slow/left',

        r'/home/lidong/Documents/datasets/Driving/disparity/35mm_focallength/scene_forwards/fast/left',

        r'/home/lidong/Documents/datasets/Driving/disparity/35mm_focallength/scene_backwards/slow/left',

        r'/home/lidong/Documents/datasets/Driving/disparity/35mm_focallength/scene_backwards/fast/left'      
      ]
pathd.sort()
paths=[

        r'/home/lidong/Documents/datasets/Driving/object_index/15mm_focallength/scene_forwards/slow/left',

        r'/home/lidong/Documents/datasets/Driving/object_index/15mm_focallength/scene_forwards/fast/left',

        r'/home/lidong/Documents/datasets/Driving/object_index/15mm_focallength/scene_backwards/slow/left',

        r'/home/lidong/Documents/datasets/Driving/object_index/15mm_focallength/scene_backwards/fast/left',

        r'/home/lidong/Documents/datasets/Driving/object_index/35mm_focallength/scene_forwards/slow/left',

        r'/home/lidong/Documents/datasets/Driving/object_index/35mm_focallength/scene_forwards/fast/left',

        r'/home/lidong/Documents/datasets/Driving/object_index/35mm_focallength/scene_backwards/slow/left',

        r'/home/lidong/Documents/datasets/Driving/object_index/35mm_focallength/scene_backwards/fast/left'      
      ]
paths.sort()
p_left_image=[]
p_right_image=[]
p_disparity=[]
p_semantic=[]
output_dir=r'/home/lidong/Documents/datasets/Driving/train_data/'
path=pathd
for p in range(len(path)):
    file=os.listdir(path[p])
    file.sort()
    for f in range(len(file)):
        p_disparity.append(os.path.join(path[p],file[f]))
        #print(os.path.join(path[p],file[f]))
path=paths
for p in range(len(path)):
    file=os.listdir(path[p])
    file.sort()
    for f in range(len(file)):
        p_semantic.append(os.path.join(path[p],file[f]))
        #print(os.path.join(path[p],file[f]))
path=pathl
for p in range(len(path)):
    file=os.listdir(path[p])
    file.sort()
    for f in range(len(file)):
        p_left_image.append(os.path.join(path[p],file[f]))
        #print(os.path.join(path[p],file[f]))
path=pathr
for p in range(len(path)):
    file=os.listdir(path[p])
    file.sort()
    for f in range(len(file)):
        p_right_image.append(os.path.join(path[p],file[f]))
        #print(os.path.join(path[p],file[f]))