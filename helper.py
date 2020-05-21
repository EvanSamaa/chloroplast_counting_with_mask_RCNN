import os
import numpy as np
from typing import List

import cv2
from skimage import measure, transform, draw, filters
from matplotlib import pyplot as plt
from matplotlib import image
from matplotlib.patches import Ellipse
from PIL import Image

# import ellipse as el
from mrcnn import config as mrcnn_config

# Typing
image_set = List[np.array]
# class Cell_data_set(mrcnn_config.Config):

####################################  Misc #####################################
def load_dir(dir_name: str) -> image_set:
    file_names = sorted(os.listdir(dir_name))
    images = []
    for name in file_names:
        full_name = dir_name + "/" + name
        images.append(np.asarray(Image.open(full_name)))
    return images
def display_image(image_arr: np.array):
    if image_arr.mean() <= 1:
        plt.imshow(image_arr, cmap='gray', vmin=0, vmax=1)
    else:
        plt.imshow(image_arr, cmap='gray', vmin=0, vmax=255)
    plt.axis("off")
    plt.show()
def display_side_by_side(im1: np.array, im2:np.array):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig.suptitle('Horizontally stacked subplots')
    if im1.mean() <= 1:
        vmax1 = 1
    else:
        vmax1 = 255
    if im2.mean() <= 1:
        vmax2 = 1
    else:
        vmax2 = 255
    ax1.imshow(im1, cmap='gray', vmin=0, vmax=vmax1)
    ax1.axis("off")
    ax2.imshow(im2, cmap='gray', vmin=0, vmax=vmax2)
    ax2.axis("off")
    plt.show()

########################  Generate mask from raw labeled data ########################
def prep_data():
    magnification_mask = np.load("7000x_ref_mask.npy")
    # mask_creation(target_dir="./masks/", processed_dir="./processed", raw_dir ="./raw", magnification_mask = magnification_mask)

    dir_tree = [x for x in os.walk("Full Dataset Raw")]
    for x in dir_tree:
        if x[0] == 'Full Dataset Raw/Data':
            dir_list = x[1]
            for item in dir_list:
                try:
                    used = mask_creation('Full Dataset Processed/Masks/',
                                         'Full Dataset Raw/Data/' + item,
                                         'Full Dataset Raw/Label/' + item,
                                         magnification_mask)
                    image_cropping('Full Dataset Processed/Data/', 'Full Dataset Raw/Data/' + item + '/', used)
                except:
                    print("error on ", item)
def concave_point_extraction(p_approx: np.array) -> np.array:
    rtl = []
    net_orient = 0
    orientation = []
    for i in range(0, p_approx.shape[0]):
        p_i = p_approx[i]
        if i == 0:
            p_i_left = p_approx[-1]
        else:
            p_i_left = p_approx[i-1]
        if i == p_approx.shape[0]-1:
            p_i_right = p_approx[-1]
        else:
            p_i_right = p_approx[i+1]
        v_i = p_i_left - p_i
        v_i_right = p_i - p_i_right
        orientation.append(np.sign(np.cross(v_i, v_i_right)))
        net_orient = net_orient + orientation[-1]
    net_orient = np.sign(net_orient)
    for i in range(0, p_approx.shape[0]):
        if orientation[i] != net_orient and orientation[i] != 0:
            rtl.append(p_approx[i])
    rtl = np.array(rtl)
    return rtl
def contour_clustering(concave_pts:np.array, contour:np.array) -> List[np.array]:
    i = 0
    j = 0
    cluster = []
    current_contour = []
    while i <= (concave_pts.shape[0]):
        if np.linalg.norm(contour[j]-concave_pts[i%concave_pts.shape[0]]) == 0:
            if current_contour != []:
                current_contour.append(contour[j])
                cluster.append(np.array(current_contour))
                current_contour = []
                current_contour.append(contour[j])
            else:
                current_contour = []
                current_contour.append(contour[j])
            i = i + 1
        else:
            if current_contour != []:
                current_contour.append(contour[j])
        j = j + 1
        if j == contour.shape[0]:
            j = 0
    return cluster
def concave_cell_mask_generation_unfiinished():
    processed = load_dir("processed")
    raw = load_dir("raw")
    for i in range(0, len(raw)):
        test = processed[i] - raw[i]
        test = cv2.threshold(test, 15, 255, cv2.THRESH_BINARY)[1]
        # find the contours present in the image
        contours = cv2.findContours(test, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        # generates filled contours so that the edges are smooth and not jagged
        cv2.drawContours(test, contours[0], contourIdx=-2, color=255, thickness=-1)
        # new and more accurate contours and generated
        contours = measure.find_contours(test, level=1)

        fig, ax = plt.subplots()
        ax.imshow(test, cmap='gray')
        plt.show()
        for n, contour in enumerate(contours):
            # approximate the contour with polygon using RDP (as per Abhinav 2018)
            appr_contour = measure.approximate_polygon(contour, tolerance=10)
            # concave points calculated
            concave_points = concave_point_extraction(appr_contour)
            # if contour has concave points aka contains more than one cell
            if concave_points.shape[0] != 0:
                # segment the contour into segments
                concave_cluster = contour_clustering(concave_points, contour)
                for curve in concave_cluster:
                    # reg = el.LsqEllipse().fit(curve)
                    center, width, height, phi = reg.as_parameters()
                    ellipse = Ellipse(xy=list(reversed(center)), width=2 * width, height=2 * height,
                                      angle=np.rad2deg(np.real(phi)),
                                      edgecolor='b', fc='None', lw=2, label='Fit', zorder=2)
                    ax.add_patch(ellipse)

            # ax.plot(appr_contour[:, 1], appr_contour[:, 0], linewidth=2)

        # display_image(test)
def mask_creation(target_dir, processed_dir, raw_dir, magnification_mask):
    processed = load_dir(processed_dir)
    processed_file_names = sorted(os.listdir(processed_dir))
    raw = load_dir(raw_dir)
    used = []
    for i in range(0, len(raw)):
        # compare masks to see if the magnification is what we need
        if not np.array_equal(processed[i][1071:1101, 655:882], magnification_mask):
            continue
        test = processed[i] - raw[i]
        # skipping unlabeled images
        if test.mean() == 0:
            continue
        used.append(processed_file_names[i])
        test = cv2.threshold(test, 15, 255, cv2.THRESH_BINARY)[1] # get binary image, as the contour could be damanged in the previous step
        test = filters.gaussian(test, sigma=2) # smooth the edges so the contour detector can pick then up better
        test = cv2.threshold(test, 0.01, 1, cv2.THRESH_BINARY)[1] # get binary image again to denoise
        contours = measure.find_contours(test, level=0.5)
        masks = []

        # now all the contours are selected, we only want relatively not concave contours. (as the contour finder would
        # often create heirrarchical contours)
        for n, contour in enumerate(contours):
            appr_contour = measure.approximate_polygon(contour, tolerance=10)
            concavepoints = concave_point_extraction(appr_contour)
            # make sure the shape is not a concave shape (which most likely include the traces of multiple cells)
            if concavepoints.shape[0] < 2:
               # generate a mask based on the contour
                mask = np.zeros(test.shape)
                mask = draw.polygon2mask(mask.shape, contour)
                masks.append(mask)
        # the contour finder would also output a contour for the HOLE, i.e. the inside of the original contour. So I will
        # eleminate them in the enxt step
        unique_masks = []
        while len(masks) != 0:
            duplicate = False
            keep = 0
            # compare the first element with the rest
            for k in range(1, len(masks)):
                temp = np.multiply(masks[0], masks[k])
                if temp.sum() >= masks[0].sum()/3:
                    duplicate = True
                    if mask[0].sum() >= mask[k].sum():
                        keep = k
                    break
            unique_masks.append(masks[keep])
            if duplicate:
                masks.pop(k)
                masks.pop(0)
            else:
                masks.pop(0)

        for m in range(0, len(unique_masks)):
            # print(target_dir + processed_file_names[i][:-4] + "_mask_" + str(m) + ".jpeg")
            # image.imsave(target_dir + processed_file_names[i][:-4] + "_mask_" + str(m) + ".png", unique_masks[m][:1023, :], cmap='grayscale')
            im = Image.fromarray(unique_masks[m][:1023, :])
            im.save(target_dir + processed_file_names[i][:-4] + "_mask_" + str(m) + ".png")
    return used
def image_cropping(target_dir, image_dir, used):
    for item in used:
        img = np.asarray(Image.open(image_dir + item))
        img_aug = img[:1023, :]
        # image.imsave(target_dir + item[:-4] + ".png", img_aug, cmap='grayscale')
        im = Image.fromarray(img_aug)
        im.save(target_dir + item[:-4] + ".png")
        # print(np.asarray(Image.open(target_dir + item[:-4] + ".png")))

if __name__ == "__main__":
    prep_data()




