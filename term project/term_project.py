import cv2, numpy as np, matplotlib.pyplot as plt

# Read the input image in grayscale
img = cv2.imread('testdata1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('testdata2.jpg', cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread('testdata3.jpg', cv2.IMREAD_GRAYSCALE)
img4 = cv2.imread('testdata4.jpg', cv2.IMREAD_GRAYSCALE)
# cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Original Image', img.shape[1]//2, img.shape[0]//2)
# cv2.imshow('Original Image', img )


# crop the image to focus on the chest area
# according to https://www.kaggle.com/code/davidbroberts/cropping-chest-x-rays

## Make a binarized copy of the image
thresh = 50
img_bin = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]

# plt.figure(figsize=(15,5))
# plt.imshow(img_bin,cmap="gray")
# plt.axis('off')
# plt.show()

## find the row that darker than threshold and crop it out
intensity_threshold = 50
down = img_bin.shape[0]//2
up = img_bin.shape[0]//2
line_thickness = 5

for i in range(img_bin.shape[0]//2,img_bin.shape[0],+5):
    # median = np.median(img_bin[i])
    row_mean = img_bin[i].mean()
    if row_mean < intensity_threshold:
        down = i
        print("Found row to crop at: ", down)
        ### Draw a line where we want to crop
        cv2.line(img_bin, (0,down), (img_bin.shape[1], down), (255), line_thickness)
        break

for i in range(img_bin.shape[0]//2,0,-5):
    # median = np.median(img_bin[i])
    row_mean = img_bin[i].mean()
    if row_mean < intensity_threshold:
        up = i
        print("Found row to crop at: ", down)
        ### Draw a line where we want to crop
        cv2.line(img_bin, (0,down), (img_bin.shape[1], down), (255), line_thickness)
        break 

img_cropped = img[up:down, :]
thresh = 80
img_bin2 = cv2.threshold(img_cropped, thresh, 255, cv2.THRESH_BINARY)[1]

intensity_threshold2 = 230
bottom = img_bin2.shape[1]//2

for i in range(0,img_bin2.shape[1],10):
    row_mean = img_bin2[0][i].mean()
    if row_mean < intensity_threshold2:
        # Add 100 pixels of padding so we don't cut the costophrenic angles off
        bottom = i
        print("Found column to crop at: ", bottom)
        ### Draw a line where we want to crop
        cv2.line(img_bin2, (bottom, 0), (bottom, img_bin.shape[0]), (255), thickness=line_thickness)
        break
    print(row_mean)
print(img_bin2.shape)
    
cv2.namedWindow('Cropped Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Cropped Image', img_bin2.shape[1]//2, img_cropped.shape[0]//2)
cv2.imshow('Cropped Image', img_bin2)

cv2.waitKey(0)
cv2.destroyAllWindows()
