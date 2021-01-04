import os 
import sys
import cv2
import tools
import numpy as np 

# base directories for raw stacks
BASE_DIR = "D:\##DrFareehaResearch\\filtered_data\\handpicked"
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
SIZE = 530
AFFINE_TRANS = (-7, -10)
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

    # binary thresholding image
    threshold_nct = cv2.inRange(temp_slice_nct, THRESHOLD_NCT_START, THRESHOLD_NCT_END)
    threshold_xct = cv2.inRange(temp_slice_xct, THRESHOLD_XCT_START, THRESHOLD_XCT_END)

    # inverse thresholding image
    inv_threshold_nct = cv2.bitwise_not(threshold_nct)
    inv_threshold_xct = cv2.bitwise_not(threshold_xct)

    sub_threshold_xct = cv2.bitwise_and(threshold_xct, threshold_xct, mask=inv_threshold_nct)

    # color segments
    clr_map_nct = cv2.bitwise_and(blue, blue,  mask=threshold_nct) # use blue
    clr_map_xct = cv2.bitwise_and(green, green, mask=sub_threshold_xct) # use green

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