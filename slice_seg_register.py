"""
The script is performing segmentation, registering and fusion of single slices of Neutron (NCT) and X-Ray (XCT) Images. The following steps have been followed:
1- Load XCT and NCT images.
2- Median filter of (KERNEL_SIZE) applied on NCT
3- Use XCT as reference and applied registerion on NCT Images that involved scaling, translation.
4- Binary thresholding on both XCT and NCT images.
5- Color segmentation
6- Fusion
7- Saving pictures as JPEGs
"""

import os
import sys
import cv2
import tools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score

DATASET_DIR = "..\\slices\\set_3"
KERNEL_SIZE = 5
green = np.zeros((550,550,3), dtype=np.uint8)
blue = np.zeros((550,550,3), dtype=np.uint8)
green[:] = [0, 255, 0]
blue[:] = [255, 0, 0]

img_paths = os.listdir(DATASET_DIR)

for i in img_paths:
    if 'X' in i:
        img_x_path = os.path.join(DATASET_DIR,i)
    elif 'N' in i:
        img_n_path = os.path.join(DATASET_DIR,i)

# opening images
img_x = tools.open_img(img_x_path)
img_n = tools.open_img(img_n_path)

# median filter neutron image
img_n = cv2.medianBlur(img_n, KERNEL_SIZE)
# median filter x-ray image
# img_x = cv2.medianBlur(img_x, 7)

# resmaple neutron image
temp_size = (500, 500)
# img_n = cv2.resize(img_n, dsize=img_x.shape, interpolation=cv2.INTER_LANCZOS4)
img_n = tools.adjust_size(img_n, 530)
img_n = tools.affine_transform(img_n, -7, -10)
print(img_n.shape)

# temporary neutron and x-ray images
temp_n = img_n
temp_x = img_x

# thresholding images
threshold_n = cv2.inRange(temp_n, 120, 230) # neutron image threshold mask
threshold_x = cv2.inRange(temp_x, 150, 230) # x-ray image threshold mask
threshold_n_inv = cv2.bitwise_not(threshold_n) # neutron image threshold inverse mask
threshold_x_inv = cv2.bitwise_not(threshold_x) # x-ray image threshold inverse mask
sub_threshold_x = cv2.bitwise_and(threshold_x, threshold_x, mask=threshold_n_inv) # x-ray minus neutron threshold mask

# converting channels to RGB
img_n_clr = cv2.cvtColor(img_n,cv2.COLOR_GRAY2RGB)
img_x_clr = cv2.cvtColor(img_x, cv2.COLOR_GRAY2RGB)

# creating color segments
clr_map_n = cv2.bitwise_and(green, green,  mask=threshold_n) # neutron green map
clr_map_x = cv2.bitwise_and(blue, blue, mask=sub_threshold_x) # xray blue map

# making final image i.e. segmented x-ray + segmented neutron image
img_fin = cv2.add(clr_map_x, clr_map_n)

temp_clr_x = cv2.bitwise_and(img_x_clr, green)
temp_clr_n = cv2.bitwise_and(img_n_clr, blue)

# test = cv2.addWeighted(temp_clr_x, 0.3, temp_clr_n, 0.8, 0)
test = cv2.add(temp_clr_n, temp_clr_x)
# tools.display_img([temp_clr_x, temp_clr_n,test], ["temp xray cologreen","temp neutron cologreen","original overlapped"])

disp_images_x = {
    "xray_preproc" : img_x,
    "threshold_xray" : threshold_x,
    "threshold_inv_xray" : threshold_x_inv,
    "color_map_xray" : clr_map_x
}

disp_images_n = {
    "neut_preproc" : img_n,
    "threshold_neut" : threshold_n,
    "threshold_inv_neut" : threshold_n_inv,
    "color_map_neut" : clr_map_n
}

disp_images_misc = {
    "sub_x_n_threshold" : sub_threshold_x,
    "final_registeration": img_fin
}

tools.display_img(list(disp_images_x.values()), list(disp_images_x.keys()))
tools.display_img(list(disp_images_n.values()), list(disp_images_n.keys()))
tools.display_img(list(disp_images_misc.values()), list(disp_images_misc.keys()))

def save_results():
    DST_DIR = os.path.join(DATASET_DIR, "results")
    os.makedirs(DST_DIR, exist_ok=True)

    n_name = os.path.split(img_n_path)[1][:-4]
    x_name = os.path.split(img_x_path)[1][:-4]

    for i in range(len(disp_images_x.keys())):
        file_name = os.path.join(DST_DIR, x_name) + "_" + list(disp_images_x.keys())[i] + ".png"
        img = list(disp_images_x.values())[i]
        cv2.imwrite(file_name, img)
    
    for i in range(len(disp_images_n.keys())):
        file_name = os.path.join(DST_DIR, n_name) + "_" + list(disp_images_n.keys())[i] + ".png"
        img = list(disp_images_n.values())[i]
        cv2.imwrite(file_name, img)

    for i in range(len(disp_images_misc.keys())):
        file_name = os.path.join(DST_DIR, "_" + list(disp_images_misc.keys())[i]) + ".png"
        img = list(disp_images_misc.values())[i]
        cv2.imwrite(file_name, img)

save_results()