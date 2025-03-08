#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 10:27:49 2025

@author: cmalili
"""

import rawpy
import numpy as np
import cv2
import matplotlib.pyplot as plt


files = ['11.dng', '54.dng', '07.dng', '02.dng', '16.dng', '23.dng', '31.dng']
rgbs = []
# Open and process RAW file
for file in files:
    raw = rawpy.imread(file)
    rgb = raw.postprocess()
    rgbs.append(rgb)
    #rgbs[0]
    #plt.imshow(rgb)
    #plt.show()

'''
# Display image
plt.imshow(rgbs[2])
plt.show() '''

gray1 = cv2.cvtColor(rgbs[2], cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(rgbs[6], cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

img1_with_keypoints = cv2.drawKeypoints(rgbs[2], keypoints1, None)
img2_with_keypoints = cv2.drawKeypoints(rgbs[6], keypoints2, None)

#plt.imshow(img1_with_keypoints)
#plt.show()

#plt.imshow(img2_with_keypoints)
#plt.show()

#print(keypoints1, keypoints2)

if len(keypoints1) == 0:
    print("No keypoints detected in image 1")
else:
    print(f"Detected {len(keypoints1)} keypoints in image 1")

if len(keypoints2) == 0:
    print("No keypoints detected in image 2")
else:
    print(f"Detected {len(keypoints2)} keypoints in image 2")
'''
cv2.imshow('Image 1 Keypoints', img1_with_keypoints)
cv2.imshow('Image 2 Keypoints', img2_with_keypoints)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

#img_matches = cv2.drawMatches(rgbs[0], keypoints1, rgbs[1], keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

print(f"Number of good matches: {len(good_matches)}")

# If no good matches, print a message
if len(good_matches) == 0:
    print("No good matches found!")
else:
    # Draw the good matches
    img_matches = cv2.drawMatches(rgbs[2], keypoints1, rgbs[6], keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
'''
    # Display the matched image
    cv2.imshow('Matched Keypoints', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''
pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
# Print the homography matrix
print("Homography Matrix:\n", H)

inliers = mask.ravel() == 1
good_inliers = [good_matches[i] for i in range(len(good_matches)) if inliers[i]]

# Draw the matches (only inliers)
img_matches = cv2.drawMatches(rgbs[2], keypoints1, rgbs[6], keypoints2, good_inliers, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
'''
cv2.imshow('Inlier Matches', img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Warp image1 into the reference frame of image2 using the homography matrix
height, width, _ = rgbs[0].shape
warped_image1 = cv2.warpPerspective(rgbs[1], H, (width, height))

mosaic = np.copy(rgbs[0])

mask_warped = np.all(warped_image1 == 0, axis=-1)

for y in range(height):
    for x in range(width):
        if not mask_warped[y, x]:
            mosaic[y, x] = warped_image1[y, x]

# Display the result
cv2.imshow('Mosaic', mosaic)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Warp image 2 (rgbs[1]) into the reference frame of image 1 (rgbs[0])
height1, width1 = rgbs[0].shape[:2]
height2, width2 = rgbs[1].shape[:2]

# Warp image 2
warped_image2 = cv2.warpPerspective(rgbs[1], H, (width1, height1))

# Display the warped image
cv2.imshow("Warped Image 2", warped_image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

# Get dimensions of image1 (reference) and image2
height1, width1 = rgbs[0].shape[:2]
height2, width2 = rgbs[1].shape[:2]

# Define the corner points of image1 and image2
corners_image1 = np.float32([[0, 0], [width1, 0], [width1, height1], [0, height1]]).reshape(-1, 1, 2)
corners_image2 = np.float32([[0, 0], [width2, 0], [width2, height2], [0, height2]]).reshape(-1, 1, 2)

# Warp image2 corners to image1's coordinate frame using the homography matrix
warped_corners_image2 = cv2.perspectiveTransform(corners_image2, H)

# Combine the corners of image1 and the warped image2 to determine the overall bounds
all_corners = np.concatenate((corners_image1, warped_corners_image2), axis=0)
[x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
[x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

# Compute the size of the panorama canvas
panorama_width = x_max - x_min
panorama_height = y_max - y_min

# Compute the translation that shifts the panorama so that all coordinates are positive
translation = np.array([[1, 0, -x_min],
                        [0, 1, -y_min],
                        [0, 0, 1]], dtype=np.float32)

# --------------------------------------------------------------------
# STEP 2: Warp image2 into the new panorama canvas
# --------------------------------------------------------------------
# Adjust homography by the translation to get the correct warped image2
H_translated = translation.dot(H)
warped_image2 = cv2.warpPerspective(rgbs[1], H_translated, (panorama_width, panorama_height))

# --------------------------------------------------------------------
# STEP 3: Place image1 into the panorama canvas
# --------------------------------------------------------------------
# Create a canvas for the panorama
panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)
# Place image1 at the appropriate translated location
panorama[-y_min:height1 - y_min, -x_min:width1 - x_min] = rgbs[0]

# --------------------------------------------------------------------
# STEP 4: Composite the images to form the panorama
# --------------------------------------------------------------------
# Create a mask from the warped image2 where there is valid data (non-black pixels)
mask_warped = np.any(warped_image2 != 0, axis=2)

# For overlapping regions, you could blend or simply override. Here we override.
panorama[mask_warped] = warped_image2[mask_warped]

# Display the final panorama
cv2.imshow("Panorama", panorama)
cv2.waitKey(0)
cv2.destroyAllWindows()