#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 18:54:39 2025

@author: cmalili
"""

import rawpy
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read raw images and postprocess them
raw1 = rawpy.imread("0.dng")
img1 = raw1.postprocess(half_size=True)
raw2 = rawpy.imread("-2.dng")
img2 = raw2.postprocess(half_size=True)
raw3 = rawpy.imread("1.dng")
img3 = raw3.postprocess(half_size=True)

raw_up1 = rawpy.imread("up4.dng")
img_up1 = raw_up1.postprocess(half_size=True)
raw_up2 = rawpy.imread("up2.dng")
img_up2 = raw_up2.postprocess(half_size=True)
raw_up3 = rawpy.imread("up3.dng")
img_up3 = raw_up3.postprocess(half_size=True)


scale_factor = 0.4  # Change to desired scaling factor
img1 = cv2.resize(img1, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
img2 = cv2.resize(img2, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
img3 = cv2.resize(img3, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

img_up1 = cv2.resize(img_up1, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
img_up2 = cv2.resize(img_up2, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
img_up3 = cv2.resize(img_up3, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

# Convert to grayscale for feature detection
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

gray_up1 = cv2.cvtColor(img_up1, cv2.COLOR_BGR2GRAY)
gray_up2 = cv2.cvtColor(img_up2, cv2.COLOR_BGR2GRAY)
gray_up3 = cv2.cvtColor(img_up3, cv2.COLOR_BGR2GRAY)


# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and descriptors for image 1, image 2, and image 3
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
kp3, des3 = sift.detectAndCompute(gray3, None)

kp_up1, des_up1 = sift.detectAndCompute(gray_up1, None)
kp_up2, des_up2 = sift.detectAndCompute(gray_up2, None)
kp_up3, des_up3 = sift.detectAndCompute(gray_up3, None)

# Function to compute homography between image A (reference) and image B (to be warped)
def compute_homography(desA, desB, kpA, kpB):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(desA, desB, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    if len(good_matches) < 4:
        raise ValueError("Not enough good matches!")

    ptsA = np.float32([kpA[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    ptsB = np.float32([kpB[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, 5.0)
    if H is None:
        raise ValueError("Homography computation failed!")
    return H

# Compute homographies: H2 warps img2 to img1, H3 warps img3 to img1
H2 = compute_homography(des1, des2, kp1, kp2)
print("Homography for img2:\n", H2)
H3 = compute_homography(des1, des3, kp1, kp3)
print("Homography for img3:\n", H3)

H1_up = compute_homography(des1, des_up2, kp1, kp_up2)
print("Homography for img2:\n", H1_up)
H2_up = compute_homography(des1, des_up2, kp1, kp_up2)
print("Homography for img2:\n", H2_up)
H3_up = compute_homography(des1, des_up3, kp1, kp_up3)
print("Homography for img3:\n", H3_up)

# Get dimensions for images
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
h3, w3 = img3.shape[:2]

h1_up, w1_up = img_up1.shape[:2]
h2_up, w2_up = img_up2.shape[:2]
h3_up, w3_up = img_up3.shape[:2]

# Compute corners for each image in the coordinate system of image1
corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)

# For image 2
corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
warped_corners_img2 = cv2.perspectiveTransform(corners_img2, H2)

# For image 3
corners_img3 = np.float32([[0, 0], [0, h3], [w3, h3], [w3, 0]]).reshape(-1, 1, 2)
warped_corners_img3 = cv2.perspectiveTransform(corners_img3, H3)

corners_img_up1 = np.float32([[0, 0], [0, h1_up], [w1_up, h1_up], [w1_up, 0]]).reshape(-1, 1, 2)
warped_corners_img_up1 = cv2.perspectiveTransform(corners_img_up1, H1_up)

corners_img_up2 = np.float32([[0, 0], [0, h2_up], [w2_up, h2_up], [w2_up, 0]]).reshape(-1, 1, 2)
warped_corners_img_up2 = cv2.perspectiveTransform(corners_img_up2, H2_up)

corners_img_up3 = np.float32([[0, 0], [0, h3_up], [w3_up, h3_up], [w3_up, 0]]).reshape(-1, 1, 2)
warped_corners_img_up3 = cv2.perspectiveTransform(corners_img_up3, H3_up)

# Combine all corners to determine the panorama dimensions
all_corners = np.concatenate((corners_img1, warped_corners_img2, warped_corners_img3, warped_corners_img_up1, warped_corners_img_up2, warped_corners_img_up3), axis=0)
[x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
[x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

# Define an extra left margin (shift to right)
extra_margin_x = 6000  # Increase this value to have more room on the left
extra_margin_y = 6000

# Update the translation distance to include extra margin on the x-axis
translation_dist = [-x_min + extra_margin_x, -y_min + extra_margin_y]
T = np.array([[1, 0, translation_dist[0]],
              [0, 1, translation_dist[1]],
              [0, 0, 1]])

# Adjust panorama size; add extra margin width to account for the shift.
#panorama_width = (x_max - x_min) + extra_margin + 1000  # You can adjust the extra width if needed
panorama_width = (x_max - x_min) + extra_margin_x
#panorama_height = (y_max - y_min) + 2000  # Adjust as needed
panorama_height = (y_max - y_min) + extra_margin_y
panorama_size = (panorama_width, panorama_height)

# Warp image 2 and image 3 into image1's coordinate system using the new translation
warped_img2 = cv2.warpPerspective(img2, T.dot(H2), panorama_size, borderMode=cv2.BORDER_TRANSPARENT)
warped_img3 = cv2.warpPerspective(img3, T.dot(H3), panorama_size, borderMode=cv2.BORDER_TRANSPARENT)

warped_img_up1 = cv2.warpPerspective(img_up1, T.dot(H1_up), panorama_size, borderMode=cv2.BORDER_TRANSPARENT)
warped_img_up2 = cv2.warpPerspective(img_up2, T.dot(H2_up), panorama_size, borderMode=cv2.BORDER_TRANSPARENT)
warped_img_up3 = cv2.warpPerspective(img_up3, T.dot(H3_up), panorama_size, borderMode=cv2.BORDER_TRANSPARENT)

# Create a canvas and place image 1 into the panorama at the translated position
panorama = np.zeros((panorama_size[1], panorama_size[0], 3), dtype=np.uint8)
panorama[translation_dist[1]:translation_dist[1]+h1, translation_dist[0]:translation_dist[0]+w1] = img1

# Composite the warped images onto the canvas.
mask2 = (warped_img2 > 0)
mask3 = (warped_img3 > 0)

mask_up1 = (warped_img_up1 > 0)
mask_up2 = (warped_img_up2 > 0)
#mask_up3 = (warped_img_up3 > 0)

panorama[mask2] = warped_img2[mask2]
panorama[mask3] = warped_img3[mask3]

panorama[mask_up1] = warped_img_up1[mask_up1]
panorama[mask_up2] = warped_img_up2[mask_up2]
#panorama[mask_up3] = warped_img_up3[mask_up3]

plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
plt.title("Stitched Panorama")
plt.axis('off')
plt.show()