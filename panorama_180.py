#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 10:27:49 2025

@author: cmalili
"""

import rawpy
import numpy as np
import cv2
#import matplotlib.pyplot as plt


files = ['12.dng', '19.dng', '25.dng', '30.dng', '36.dng']
rgbs = []
# Open and process RAW file
for file in files:
    raw = rawpy.imread(file)
    rgb = raw.postprocess()
    rgbs.append(rgb)
    
    #plt.imshow(rgb)
    #plt.show()

'''
# Display image
plt.imshow(rgbs[2])
plt.show() '''

gray1 = cv2.cvtColor(rgbs[0], cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(rgbs[1], cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

img1_with_keypoints = cv2.drawKeypoints(rgbs[0], keypoints1, None)
img2_with_keypoints = cv2.drawKeypoints(rgbs[1], keypoints2, None)

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
    img_matches = cv2.drawMatches(rgbs[0], keypoints1, rgbs[1], keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
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
img_matches = cv2.drawMatches(rgbs[0], keypoints1, rgbs[1], keypoints2, good_inliers, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
'''
cv2.imshow('Inlier Matches', img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

# Warp image1 into the reference frame of image2 using the homography matrix
height, width, _ = rgbs[1].shape
warped_image1 = cv2.warpPerspective(rgbs[0], H, (width, height))

mosaic = np.copy(rgbs[1])

mask_warped = np.all(warped_image1 == 0, axis=-1)

for y in range(height):
    for x in range(width):
        if not mask_warped[y, x]:
            mosaic[y, x] = warped_image1[y, x]

# Display the result
cv2.imshow('Mosaic', mosaic)
cv2.waitKey(0)
cv2.destroyAllWindows()











