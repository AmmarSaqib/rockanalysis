import os
import sys
import cv2
import tools
import numpy as np
import matplotlib.pyplot as plt

DATASET_DIR = "..\\slices\\set_3"
red = np.zeros((550,550,3), dtype=np.uint8)
blue = np.zeros((550,550,3), dtype=np.uint8)
red[:] = [0, 255, 0]
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

# resizing image
img_n = cv2.resize(img_n, dsize=img_x.shape, interpolation=cv2.INTER_LANCZOS4)

MAX_FEATURES = 1000
GOOD_MATCH_PERCENT = 0.15
 
 
def alignImages(im1, im2):
 
    # # Convert images to grayscale
    # im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    # im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
   
    im1Gray = im1
    im2Gray = im2

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    # height, width, channels = im2.shape
    height, width = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h


img, h = alignImages(img_x, img_n)

tools.display_img(img, "image_feat_align")


print(h)