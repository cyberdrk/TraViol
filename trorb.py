import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('1.png',0)          # queryImage
img2 = cv2.imread('3.png',0) # trainImage

# Initiate ORB detector
orb = cv2.ORB_create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE)

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
'''
for keyPoint in kp1:
    x = keyPoint.pt[0]
    y = keyPoint.pt[1]
    s = keyPoint.size
    print x, y
'''
Y = 0
'''
for i in range(1557, 1578):
	x1 = kp1.pt[0]
	x2 = kp1.pt[0]

	y1 = kp1.pt[0]
	y2 = kp1.pt[1]

	print "Distance"
	Y = y2 - y1
	print Y

'''

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[1557:1575], None, flags=2)
plt.imshow(img3),plt.show()