#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 13:10:08 2025

@author: cmalili
"""

import rawpy
import cv2
import numpy as np
import matplotlib.pyplot as plt

raw1 = rawpy.imread("0.dng")
img1 = raw1.postprocess()

raw2 = rawpy.imread("1.dng")
img2 = raw2.postprocess()

'''
# Load the two images
img1 = cv2.imread('image1.jpg')  # Base image (destination)
img2 = cv2.imread('image2.jpg')  # Image to warp (source)
'''

# Convert images to grayscale for feature detection
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Detect SIFT features and compute descriptors
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# Use BFMatcher with L2 norm (SIFT uses float descriptors)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.knnMatch(des1, des2, k=2)

# Filter good matches using Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print(f'Found {len(good_matches)} good matches.')

# Ensure there are enough matches
if len(good_matches) < 4:
    print("Not enough good matches!")
    exit()

# Extract location of good matches
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Compute homography from img2 (source) to img1 (destination) using RANSAC
H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
if H is None:
    print("Homography computation failed!")
    exit()
print("Homography matrix:\n", H)

# Get image dimensions
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

# Compute corners of img2 and warp them to img1's coordinate system
corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
warped_corners_img2 = cv2.perspectiveTransform(corners_img2, H)

# Determine the size of the resulting panorama
all_corners = np.concatenate((np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2), warped_corners_img2), axis=0)
[x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
[x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

# Translation matrix to shift the panorama so that coordinates are non-negative
translation_dist = [-x_min, -y_min]
T = np.array([[1, 0, translation_dist[0]],
              [0, 1, translation_dist[1]],
              [0, 0, 1]])

# Warp img2 using the composed transformation T*H
panorama_size = (x_max - x_min, y_max - y_min)
warped_img2 = cv2.warpPerspective(img2, T.dot(H), panorama_size)

# Create a canvas for the panorama and place img1 on it
panorama = warped_img2.copy()
panorama[translation_dist[1]:translation_dist[1]+h1, translation_dist[0]:translation_dist[0]+w1] = img1

# OPTIONAL: Create blending masks for smoother compositing in the overlapping region
# Create masks for both images in the panorama coordinate system
mask1 = np.zeros((y_max-y_min, x_max-x_min), dtype=np.float32)
mask1[translation_dist[1]:translation_dist[1]+h1, translation_dist[0]:translation_dist[0]+w1] = 1.0
mask2 = cv2.warpPerspective(np.ones((h2, w2), dtype=np.float32), T.dot(H), panorama_size)

# Create a simple blended panorama: in the overlapping region, compute weighted average
blend_mask = np.clip(mask1 + mask2, 1e-6, 1)  # Avoid division by zero
alpha = mask1 / blend_mask  # Weight for img1; weight for warped_img2 is (1-alpha)

# Convert images to float for blending
img_panorama = panorama.astype(np.float32)
img1_on_panorama = np.zeros_like(img_panorama)
img1_on_panorama[translation_dist[1]:translation_dist[1]+h1, translation_dist[0]:translation_dist[0]+w1] = img1.astype(np.float32)

# Blend the overlapping areas
for c in range(3):  # for each color channel
    img_panorama[:, :, c] = img1_on_panorama[:, :, c] * alpha + img_panorama[:, :, c] * (1 - alpha)

blended_panorama = np.clip(img_panorama, 0, 255).astype(np.uint8)

'''
# Save and display the result
cv2.imwrite('panorama.jpg', blended_panorama)
cv2.imshow('Panorama', blended_panorama)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

plt.imshow(blended_panorama)
plt.show()