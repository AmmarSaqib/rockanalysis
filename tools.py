from operator import pos
import cv2
import numpy as np
from math import log2

from skimage import data
from skimage.filters import threshold_multiotsu


def calc_entropy(prob_dist):
    """
    Calculates the shanon entropy of the probability distribution passed.
    :param prob_dist: (Numpy.Array) the probability distribution of image(s)

    :return: (float) the entropy of the probability distribution
    """

    # get non zero indexes
    non_zero_idx = np.nonzero(prob_dist)

    # print(prob_dist[non_zero_idx].flatten().shape)

    return np.sum(prob_dist[non_zero_idx] * np.log2(1/prob_dist[non_zero_idx]))

def open_img(img_path):
    """
    Opens the image using OpenCV in grayscale.
    :param img_path: (String) path to the image
    :return: (Numpy.Array) image
    """
    return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

def rotate_bound(image, angle):
    """
    Rotates the image passed clockwise with the angle passed.
    :param image: (Numpy.Array)  image to apply rotation on
    :param angle: (int) angle for rotation
    :return: (Numpy.Array) rotated image
    """
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def crop_img(img):
    """
    Crops the image to size.
    :param img: (Numpy.Array) the img
    :return: (Numpy.Array) the cropped img
    """

    (h, w) = img.shape
    (cX, cY) = (w // 2, h // 2)

    return img[cY-275:cY+275, cX-275:cX+275]

def affine_transform(img, tx, ty):
    """

    """
    rows, cols = img.shape

    M = np.float32([[1, 0, tx],[0, 1, ty]])
    dst = cv2.warpAffine(img, M, (cols, rows))

    return dst


def adjust_size(img, size):
    """

    """
    SIZE = 550
    offset = SIZE - size

    dst = cv2.resize(img, dsize=(size, size), interpolation=cv2.INTER_LANCZOS4)

    return np.pad(dst, offset//2, mode="constant")

def display_img(imgs, winname):
    """
    Displays the image(s) in a named window
    :param img: (Numpy.Array) image
    :param winname: (String) name of the window
    :return: None
    """
    if type(imgs) != list and type(winname) != list:
        imgs= [imgs]
        winname = [winname]

    if len(imgs) != len(winname):
        print("Error: List lengths must be the same")
        return None

    for img, win in zip(imgs, winname):
        cv2.namedWindow(win)
        cv2.imshow(win, img)
    cv2.waitKey()

def find_optimal_rotation(img_x, img_n, angles):
    """
    Rotates the image on different angles and computes the normalized mutual information for the images.
    Returns the angle at which the mutual information is 
    the maximun for neutron image.
    Xray image is used as reference image
    :param img_x: (Numpy.Array) X-Ray image
    :param img_n: (Numpy.Array) Neutron image
    :param angles: (List) list of angles to test
    """
    norm_mutual_info = []
    for i in range(len(angles)):
        temp_n = rotate_bound(img_n, angles[i])
        temp = crop_img(temp_n)
        norm_mutual_info.append(normalized_mutual_information(img_x, temp_n))

    idx = np.argmin(norm_mutual_info)
    return angles[idx], norm_mutual_info[idx]

def normalized_mutual_information(img_x, img_n):
    """
    Calculates the Normalized Mutual Information for the images passed
    :param img_x_path: (Numpy.Array) X-Ray image
    :param img_n_path: (Numpy.Array) Neutron image

    :return: (float) normalized mutual information
    """

    # opening images
    # img_x = cv2.imread(img_x_path, cv2.IMREAD_GRAYSCALE)
    # img_n = cv2.imread(img_n_path, cv2.IMREAD_GRAYSCALE)

    # get reference size of X-Ray image
    img_size = img_x.shape

    # resize neutron image using Lanczos interpolation
    img_n = cv2.resize(img_n, dsize=img_size, interpolation=cv2.INTER_LANCZOS4)

    # calculating histograms and joint histogram
    hist_X = np.histogram(img_x.flatten(), bins=256)[0]
    hist_N = np.histogram(img_n.flatten(), bins=255)[0]
    hist_XN = np.histogram2d(img_x.flatten(), img_n.flatten(), bins=256)[0].flatten()

    # print(hist_X.shape)
    # print(hist_N.shape)
    # print(hist_XN.shape)

    # calculating probability distribution of histograms
    hist_X = hist_X / (img_size[0] * img_size[1])
    hist_N = hist_N / (img_size[0] * img_size[1])
    hist_XN = hist_XN / (img_size[0] * img_size[1])

    # getting marginal and joing entropy
    H_X = calc_entropy(hist_X)
    H_N = calc_entropy(hist_N)
    H_XN = calc_entropy(hist_XN)

    return (H_X + H_N) / H_XN

def open_img_stack(path, dimensions):
    """
    Opens the stack of images from a ".raw" extension reshapes the image data using the expected dimenstion and returns a
    Numpy.Array.
    :param path: (String) path to the ".raw" file of the stack of slices
    :param dimensions: (Tuple) containing the following in the order->
                                            - number of slices
                                            - width
                                            - height
    :return: (Numpy.Array) containing slices of images reshaped according to the dimensions passed.
    """
    # opening the file
    raw_file_data = open(path)
    
    count = dimensions[0] * dimensions[1] * dimensions[2]

    # constructing an array from the raw file using numpy uint8 datatype and expected nulber of values
    raw_file_data = np.fromfile(raw_file_data, dtype=np.uint8, count=count)

    # reshaping the array into the form of the image stack
    img_stack = raw_file_data.reshape(dimensions)

    return img_stack

def get_multiotsu_slice(slice, classes):
    """
    The function takes in the slice and the number of classes that are 
    to be performed via the multi-otsu thresholding and returns the resulting 
    threshold slices in a list.

    :params slice: (Numpy.Array) slice to perform multi-otsu on
    :params classes: (int) the number of classes for multi-otsu
    """

    post_threshold_slices = [] # to store the thersholded slices

    # generating thresholding values
    threshold_slice = threshold_multiotsu(slice, classes=classes) 
    
    # getting classification on pixel level for the identified classes
    slice_regions = np.digitize(slice, bins=threshold_slice) 

    unique_val = np.unique(slice_regions) # getting unique class values

    # fetching slice for each class
    for _class in unique_val:
        temp = np.where(slice_regions == _class, 255, 0)
        temp = temp.astype(np.uint8)
        post_threshold_slices.append(temp)

    return post_threshold_slices
    
def preprocess_nct(nct_slice, kernel_size, size, affine_trans):
    """
    The function applies pre-processing for NCT slices that include the following:
    
    1- median blur
    2- downsample (resizing)
    3- affine adjustment

    """
    
    nct_slice = cv2.medianBlur(nct_slice, ksize=kernel_size)
    nct_slice = adjust_size(nct_slice, size)
    nct_slice = affine_transform(nct_slice, affine_trans[0], affine_trans[1])

    return nct_slice