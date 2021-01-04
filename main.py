import os 
import sys
import cv2
import tools
import numpy as np 

# base directories for raw stacks
BASE_DIR = "D:\##DrFareehaResearch\cropped_sample_slices"
NCT_DIR = "NCT_F1"
XCT_DIR = "XCT_F1"
XCT_FILE = "SlicesYNew.raw"
NCT_FILE = "SlicesYNew.raw"

XCT_SLICE_DIR = os.path.join(BASE_DIR, XCT_DIR, XCT_FILE)
NCT_SLICE_DIR = os.path.join(BASE_DIR, NCT_DIR, NCT_FILE)

# stack dimenstions
SLICES = 500
WIDTH = 550
HEIGHT = 550
DIMENSIONS = (SLICES, WIDTH, HEIGHT)

# pre-processing parameters
KERNEL_SIZE = 7
THRESHOLD_NCT_START = 138
THRESHOLD_NCT_END = 230
THRESHOLD_XCT_START = 150
THRESHOLD_XCT_END = 230

# color masks
green = np.zeros((WIDTH,HEIGHT,3), dtype=np.uint8)
blue = np.zeros((WIDTH,HEIGHT,3), dtype=np.uint8)
green[:] = [0, 255, 0]
blue[:] = [255, 0, 0]


img_stack_xct = tools.open_img_stack(XCT_SLICE_DIR, DIMENSIONS)
img_stack_nct = tools.open_img_stack(NCT_SLICE_DIR, (500,1000,1000))

# display slices
tools.display_img([img_stack_xct[0], img_stack_nct[0]], ['xct', 'nct'])


processed_stack_xct = []
processed_stack_nct = []
processed_final = []

# preprocessing slices
for i in range(SLICES):

    # filtering image with median blur
    img_nct = cv2.medianBlur(img_stack_nct[i], ksize=KERNEL_SIZE)
    
    # resample neutron image
    img_nct = tools.adjust_size(img_nct, 530)
    img_nct = tools.affine_transform(img_nct, -7, -10)
    
    # temporary neutron and x-ray images
    temp_slice_xct = img_stack_xct[i]
    temp_slice_nct = img_nct

    # binary thresholding image
    threshold_nct = cv2.inRange(temp_slice_nct, THRESHOLD_NCT_START, THRESHOLD_NCT_END)
    threshold_xct = cv2.inRange(temp_slice_xct, THRESHOLD_XCT_START, THRESHOLD_XCT_END)

    # inverse thresholding image
    inv_threshold_nct = cv2.bitwise_not(threshold_nct)
    inv_threshold_xct = cv2.bitwise_not(threshold_xct)

    sub_threshold_xct = cv2.bitwise_and(threshold_xct, threshold_xct, mask=inv_threshold_nct)

    # converting to BGR channel
    temp_slice_nct_clr = cv2.cvtColor(img_stack_nct[i], cv2.COLOR_GRAY2RGB)
    temp_slice_xct_clr = cv2.cvtColor(img_stack_xct[i], cv2.COLOR_GRAY2RGB)

    # color segments
    clr_map_nct = cv2.bitwise_and(green, green,  mask=threshold_nct)
    clr_map_xct = cv2.bitwise_and(blue, blue, mask=sub_threshold_xct)

    # making final image i.e. segmented x-ray + segmented neutron image
    img_fin = cv2.add(clr_map_xct, clr_map_nct)

    # adding images to stack
    processed_stack_xct.append(clr_map_xct)
    processed_stack_nct.append(clr_map_nct)
    processed_final.append(img_fin)

    # print('xct size: {}'.format(clr_map_xct.shape))
    # print('nct size: {}'.format(clr_map_nct.shape))
    # print('fin size: {}'.format(img_fin.shape))
    # break

img_stack_xct = np.array(processed_stack_xct)
img_stack_nct = np.array(processed_stack_nct)
img_stack_final = np.array(processed_final)

disp_imgs_xct = {
    "imgs": [img_stack_xct[0], img_stack_xct[20], img_stack_xct[40]],
    "wins": ["xct_slice_0", "xct_slice_20", "xct_slice_40"]
}
disp_imgs_nct = {
    "imgs": [img_stack_nct[0], img_stack_nct[20], img_stack_nct[40]],
    "wins": ["nct_slice_0", "nct_slice_20", "nct_slice_40"]
}
disp_imgs_final = {
    "imgs": [img_stack_final[0], img_stack_final[20], img_stack_final[40]],
    "wins": ["final_slice_0", "final_slice_20", "final_slice_40"]
}
# diplaying images
tools.display_img(disp_imgs_xct["imgs"], disp_imgs_xct["wins"])
tools.display_img(disp_imgs_nct["imgs"], disp_imgs_nct["wins"])
tools.display_img(disp_imgs_final["imgs"], disp_imgs_final["wins"])

# saving raw file
"""
NOTE: to open in ImageJ or FIJI use the following setting:
Image Type: 24-bit BGR
"""

img_stack_xct.tofile("xct_stack_proc_clr_map.raw")
img_stack_nct.tofile("nct_stack_proc_clr_map.raw")
img_stack_final.tofile("fused_xct_nct_clr_map.raw")