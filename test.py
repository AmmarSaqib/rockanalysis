#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 12-1-2020 11:55 GMT +5:00
# @Author  : Ammar Saqib
# @Site    :
# @File    : test.py
# @IDE: Visual Studio Code

"""
Testing different python functions and details.
"""

import os
import cv2
import numpy as np
import pandas as pd

DATASET_DIR = "..\\slices\\set_2"

img_xray = []
img_neutron = []

img_path = os.listdir(DATASET_DIR)

for image in img_path:

    if "XCT" in image:
        img_xray.append(image)
    elif "NCT" in image:
        img_neutron.append(image)


print("X-ray images :", img_xray)
print("Neutron images :", img_neutron)

img_x = cv2.imread(os.path.join(DATASET_DIR, img_xray[0]), cv2.IMREAD_GRAYSCALE)
img_n = cv2.imread(os.path.join(DATASET_DIR, img_neutron[0]), cv2.IMREAD_GRAYSCALE)

print("X-ray image shape : ", img_x.shape)
print("Neutron image shape : ", img_n.shape)

img_n = cv2.resize(img_n, dsize=(550, 550), interpolation=cv2.INTER_LANCZOS4)

print("X-ray image shape : ", img_x.shape)
print("Neutron image shape : ", img_n.shape)

img_x_hist = cv2.calcHist([img_x], [0], None, [256], [0, 256]).flatten()
img_n_hist = cv2.calcHist([img_x, img_n], [0], None, [256], [0, 256], accumulate=True).flatten()

hist_2d = np.histogram2d(img_x.flatten(), img_n.flatten(), bins=255)
hist_2d_norm = hist_2d[0] / (550*550)

non_zero = np.nonzero(hist_2d[0].flatten())

print(np.sum(hist_2d_norm.flatten()[non_zero]))

hist_1d = np.histogram(img_x.flatten(), bins=255)
print(np.sum(hist_1d[0] / (550*550)))


# print(np.sum(hist_2d[0]/(550*550)))
# print(hist_2d[1].shape)
# print(hist_2d[2].shape)

# print(hist_2d[0].shape, hist_2d[1].shape, hist_2d[2].shape)
# dt = pd.DataFrame(data=hist_2d[0])
# print(dt)
# for x, n in zip(img_x_hist, img_n_hist):
#     print(x, n)

# print(img_x_hist == img_n_hist)

# cv2.namedWindow("X-Ray")
# cv2.imshow("X-Ray", img_x)

# cv2.namedWindow("Neutron")
# cv2.imshow("Neutron", img_n)

# cv2.waitKey()