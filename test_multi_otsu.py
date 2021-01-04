import os 
import cv2
import tools

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from skimage import data
from skimage.filters import threshold_multiotsu

# base directories for raw stacks
# BASE_DIR = "D:\##DrFareehaResearch\\filtered_data\\handpicked"
BASE_DIR = "/media/ammarsaqib/New Volume/##DrFareehaResearch/filtered_data/handpicked"
NCT_DIR = os.path.join(BASE_DIR, "01_NCT_F1")
XCT_DIR = os.path.join(BASE_DIR, "01_XCT_F1")

# stack dimenstions
SLICES = 99
WIDTH = 550
HEIGHT = 550
DIMENSIONS = (SLICES, WIDTH, HEIGHT)

# color masks 
green = np.zeros((WIDTH,HEIGHT,3), dtype=np.uint8)
blue = np.zeros((WIDTH,HEIGHT,3), dtype=np.uint8)
green[:] = [0, 255, 0]
blue[:] = [255, 0, 0]

# pre-processing parameters
KERNEL_SIZE = 5
THRESHOLD_NCT_START = 120
THRESHOLD_NCT_END = 230
THRESHOLD_XCT_START = 150
THRESHOLD_XCT_END = 230

processed_stack_xct = []
processed_stack_nct = []
processed_final = []

# getting list of slices from directory
NCT_FILES = os.listdir(NCT_DIR)
NCT_FILES = [NCT_FILES[i] for i in range(len(NCT_FILES)-1,-1,-1)] # reversing the list to match the order
XCT_FILES = os.listdir(XCT_DIR)

# print(NCT_FILES)
# print(XCT_FILES)

xct = XCT_FILES[0]
nct = NCT_FILES[0]

xct = tools.open_img(os.path.join(XCT_DIR, xct))
nct = tools.open_img(os.path.join(NCT_DIR, nct))

print(xct.shape)
print(nct.shape)

# filtering image with median blur
nct = cv2.medianBlur(nct, ksize=KERNEL_SIZE)

# resample neutron image
nct = tools.adjust_size(nct, 530)
nct = tools.affine_transform(nct, -7, -10)

thresholds_xct = threshold_multiotsu(xct, classes=5)
thresholds_nct = threshold_multiotsu(nct, classes=5)

print("Thresholds XCT: {}".format(thresholds_xct))
print("Thresholds NCT: {}".format(thresholds_nct))

xct_regions = np.digitize(xct, bins=thresholds_xct)
nct_regions = np.digitize(xct, bins=thresholds_nct)

