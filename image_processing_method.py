import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('hw1pic/puppies.jpg')

blurred_img = cv2.blur(img, (5, 5))

laplacian_img = cv2.Laplacian(blurred_img, cv2.CV_64F)

sharpen_kernel = np.array([[0,-1,0],
                           [-1,5,-1],
                           [0,-1,0]])
sharpen = cv2.filter2D(img, -1, sharpen_kernel)

cv2.imshow('Original Image', img)
cv2.imshow('Blurred Image', blurred_img)
cv2.imshow('Laplacian Image', laplacian_img)
cv2.imshow('Sharpened Image', sharpen)
cv2.waitKey(0)
cv2.destroyAllWindows()