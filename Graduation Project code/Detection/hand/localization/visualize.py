# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 17:22:15 2020

@author: Mahmoud Nada
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os
import glob
import cv2
from PIL import Image
import scipy.io

image_paths = glob.glob('data/Detection_Data/hand_localization_dataset/training_dataset/training_data/images/*.jpg')
boxes_paths = glob.glob('data/Detection_Data/hand_localization_dataset/training_dataset/training_data/annotations/*.mat')

test_image = cv2.imread(image_paths[0], 0)
box_test = scipy.io.loadmat(boxes_paths[0])

plt.imshow(test_image)