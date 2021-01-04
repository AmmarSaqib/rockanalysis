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
nct_colors = [(255, 217, 0), (255, 0, 72), (255, 170, 255), (255, 0, 191), (43, 0, 43)]
xct_colors = [(0, 0, 255), (0, 11, 255), (0, 255, 208), (0, 255, 234), (0, 255, 149)]
# green = np.zeros((WIDTH,HEIGHT,3), dtype=np.uint8)
# green[:] = [0, 255, 0]
nct_phase_colors = [np.zeros((WIDTH,HEIGHT,3), dtype=np.uint8) for i in range(len(nct_colors))]
xct_phase_colors = [np.zeros((WIDTH,HEIGHT,3), dtype=np.uint8) for i in range(len(xct_colors))]
for i in range(len(nct_colors)):
    nct_phase_colors[i][:] = nct_colors[i]
for i in range(len(xct_colors)):
    xct_phase_colors[i][:] = xct_colors[i]

# pre-processing parameters
KERNEL_SIZE = 5
SIZE = 530
AFFINE_TRANS = (-7, -10)
THRESHOLD_NCT_START = 120
THRESHOLD_NCT_END = 230
THRESHOLD_XCT_START = 150
THRESHOLD_XCT_END = 230
MULTI_OTSU_CLASSES = 5

processed_stack_xct_multiotsu = [[] for i in range(MULTI_OTSU_CLASSES)]
processed_final = []
processed_stack_nct_multiotsu = [[] for i in range(MULTI_OTSU_CLASSES)]

# getting list of slices from directory
NCT_FILES = os.listdir(NCT_DIR)
NCT_FILES = [NCT_FILES[i] for i in range(len(NCT_FILES)-1,-1,-1)] # reversing the list to match the order
XCT_FILES = os.listdir(XCT_DIR)

# preprocessing slices
for i in range(SLICES):

    # opening XCT and NCT images
    img_nct = tools.open_img(os.path.join(NCT_DIR, NCT_FILES[i]))
    img_xct = tools.open_img(os.path.join(XCT_DIR, XCT_FILES[i]))

    # pre-processing NCT slices
    img_nct = tools.preprocess_nct(img_nct, KERNEL_SIZE, SIZE, AFFINE_TRANS)
    
    # temporary neutron and x-ray images
    temp_slice_xct = img_xct
    temp_slice_nct = img_nct

    # multi-otsu thresholding for NCT & XCT slices 
    nct_thresh_phases = tools.get_multiotsu_slice(temp_slice_nct, MULTI_OTSU_CLASSES)
    xct_thresh_phases = tools.get_multiotsu_slice(temp_slice_xct, MULTI_OTSU_CLASSES)

    # binary thresholding image
    # threshold_xct = cv2.inRange(temp_slice_xct, THRESHOLD_XCT_START, THRESHOLD_XCT_END)

    # inverse thresholding image
    # inv_thresh_nct_phases = [cv2.bitwise_not(threshold_nct) for threshold_nct in nct_thresh_phases]
    inv_thresh_nct_phases = nct_thresh_phases
    inv_threshold_xct_phases = xct_thresh_phases
    # inv_threshold_xct = cv2.bitwise_not(threshold_xct)

    # sub_threshold_xct = threshold_xct
    # for inv_nct_thresh in inv_thresh_nct_phases:
    #     sub_threshold_xct = cv2.bitwise_and(threshold_xct, threshold_xct, mask=inv_nct_thresh)

    sub_threshold_nct_phases = inv_thresh_nct_phases
    sub_threshold_xct_phases = inv_threshold_xct_phases
    
    # color segments
    # clr_map_xct = cv2.bitwise_and(green, green, mask=sub_threshold_xct) # use green
    clr_map_xct_phases = [cv2.bitwise_and(xct_phase_colors[_class], xct_phase_colors[_class],  mask=sub_threshold_xct_phases[_class]) for _class in range(MULTI_OTSU_CLASSES)]
    clr_map_nct_phases = [cv2.bitwise_and(nct_phase_colors[_class], nct_phase_colors[_class],  mask=sub_threshold_nct_phases[_class]) for _class in range(MULTI_OTSU_CLASSES)]

    # tools.display_img(clr_map_nct_phases, ['p1','p2', 'p3', 'p4'])

    # making final image i.e. segmented x-ray + different phases of Neutron image
    # img_fin =  clr_map_xct
    # for _class in range(MULTI_OTSU_CLASSES):
    #     img_fin = cv2.add(img_fin, clr_map_nct_phases[_class])

    # adding images to stack
    # processed_stack_xct.append(clr_map_xct)
    # processed_final.append(img_fin)
    for _class in range(MULTI_OTSU_CLASSES):
        processed_stack_nct_multiotsu[_class].append(clr_map_nct_phases[_class])
    for _class in range(MULTI_OTSU_CLASSES):
        processed_stack_xct_multiotsu[_class].append(clr_map_xct_phases[_class])

# img_stack_xct = np.array(processed_stack_xct)
img_stack_xct_phases = [np.array(xct_phase_stack) for xct_phase_stack in processed_stack_xct_multiotsu]
img_stack_nct_phases = [np.array(nct_phase_stack) for nct_phase_stack in processed_stack_nct_multiotsu]
# img_stack_final = np.array(processed_final)

# disp_imgs_phase_1_nct = {
#     "imgs": [img_stack_nct_phases[0][0], img_stack_nct_phases[0][20], img_stack_nct_phases[0][40]],
#     "wins": ["nct_phase_1_slice_0", "nct_phase_1_slice_20", "nct_phase_1_slice_40"]
# }
# disp_imgs_phase_2_nct = {
#     "imgs": [img_stack_nct_phases[1][0], img_stack_nct_phases[1][20], img_stack_nct_phases[1][40]],
#     "wins": ["nct_phase_2_slice_0", "nct_phase_2_slice_20", "nct_phase_2_slice_40"]
# }
# disp_imgs_phase_3_nct = {
#     "imgs": [img_stack_nct_phases[2][0], img_stack_nct_phases[2][20], img_stack_nct_phases[2][40]],
#     "wins": ["nct_phase_3_slice_0", "nct_phase_3_slice_20", "nct_phase_3_slice_40"]
# }
# disp_imgs_phase_4_nct = {
#     "imgs": [img_stack_nct_phases[3][0], img_stack_nct_phases[3][20], img_stack_nct_phases[3][40]],
#     "wins": ["nct_phase_4_slice_0", "nct_phase_4_slice_20", "nct_phase_4_slice_40"]
# }
# # disp_imgs_final = {
# #     "imgs": [img_stack_final[0], img_stack_final[20], img_stack_final[40]],
# #     "wins": ["final_slice_0", "final_slice_20", "final_slice_40"]
# # }


# disp_imgs_phase_1_xct = {
#     "imgs": [img_stack_xct_phases[0][0], img_stack_xct_phases[0][20], img_stack_xct_phases[0][40]],
#     "wins": ["xct_phase_1_slice_0", "xct_phase_1_slice_20", "xct_phase_1_slice_40"]
# }
# disp_imgs_phase_2_xct = {
#     "imgs": [img_stack_xct_phases[1][0], img_stack_xct_phases[1][20], img_stack_xct_phases[1][40]],
#     "wins": ["xct_phase_2_slice_0", "xct_phase_2_slice_20", "xct_phase_2_slice_40"]
# }
# disp_imgs_phase_3_xct = {
#     "imgs": [img_stack_xct_phases[2][0], img_stack_xct_phases[2][20], img_stack_xct_phases[2][40]],
#     "wins": ["xct_phase_3_slice_0", "xct_phase_3_slice_20", "xct_phase_3_slice_40"]
# }
# disp_imgs_phase_4_xct = {
#     "imgs": [img_stack_xct_phases[3][0], img_stack_xct_phases[3][20], img_stack_xct_phases[3][40]],
#     "wins": ["xct_phase_4_slice_0", "xct_phase_4_slice_20", "xct_phase_4_slice_40"]
# }

# # diplaying images
# tools.display_img(disp_imgs_phase_1_nct["imgs"], disp_imgs_phase_1_nct["wins"])
# tools.display_img(disp_imgs_phase_2_nct["imgs"], disp_imgs_phase_2_nct["wins"])
# tools.display_img(disp_imgs_phase_3_nct["imgs"], disp_imgs_phase_3_nct["wins"])
# tools.display_img(disp_imgs_phase_4_nct["imgs"], disp_imgs_phase_4_nct["wins"])
# # tools.display_img(disp_imgs_final["imgs"], disp_imgs_final["wins"])

# tools.display_img(disp_imgs_phase_1_xct["imgs"], disp_imgs_phase_1_xct["wins"])
# tools.display_img(disp_imgs_phase_2_xct["imgs"], disp_imgs_phase_2_xct["wins"])
# tools.display_img(disp_imgs_phase_3_xct["imgs"], disp_imgs_phase_3_xct["wins"])
# tools.display_img(disp_imgs_phase_4_xct["imgs"], disp_imgs_phase_4_nct["wins"])

# saving raw file
"""
NOTE: to open in ImageJ or FIJI use the following setting:
Image Type: 24-bit BGR
"""

# img_stack_xct.tofile("xct_stack_proc_clr_map.raw")
for _class in range(MULTI_OTSU_CLASSES):
    img_stack_nct_phases[_class].tofile("nct_stack_proc_clr_map_phase_{}.raw".format(_class + 1))

for _class in range(MULTI_OTSU_CLASSES):
    img_stack_xct_phases[_class].tofile("xct_stack_proc_clr_map_phase_{}.raw".format(_class + 1))
# img_stack_final.tofile("fused_xct_nct_clr_map.raw")

