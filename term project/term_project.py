import cv2, numpy as np, matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Read the input image in grayscale
img = cv2.imread('data/testdata1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('data/testdata2.jpg', cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread('data/testdata3.jpg', cv2.IMREAD_GRAYSCALE)
img4 = cv2.imread('data/testdata4.jpg', cv2.IMREAD_GRAYSCALE)

img = img4

def show_image(image, title='Image'):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, image.shape[1]//2, image.shape[0]//2)
    cv2.imshow(title, image)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()

# apply CLAHE to enhance the contrast of the image
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enh = clahe.apply(img)

# 1. crop the image to focus on the chest area

## 1.1 OTSU threshold

inv = enh
_, img_bin = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print("OTSU threshold value: ", _)
# show_image(img_bin, 'Initial Lung Binary Mask')

## draw top and bottom crop lines
intensity_threshold = 50
down = img_bin.shape[0]//2
up = img_bin.shape[0]//2
line_thickness = 5

for i in range(img_bin.shape[0]//2,img_bin.shape[0],+5):
    row_mean = img_bin[i].mean()
    if row_mean < intensity_threshold:
        down = i
        print("Found row to crop at: ", down)
        ### Draw a line where we want to crop
        cv2.line(img_bin, (0,down), (img_bin.shape[1], down), (255), line_thickness)
        break

for i in range(img_bin.shape[0]//2,0,-5):
    row_mean = img_bin[i].mean()
    if row_mean < intensity_threshold:
        up = i
        print("Found row to crop at: ", up)
        ### Draw a line where we want to crop
        cv2.line(img_bin, (0,up), (img_bin.shape[1], up), (255), line_thickness)
        break 

img_bin_cropped = img_bin[up:down, : ]
show_image(img_bin_cropped, 'Cropped Binary Mask')

# blur the image to reduce noise before thresholding
img_bin_cropped = cv2.GaussianBlur(img_bin_cropped, (5,5), 0)

mean_thresh = cv2.adaptiveThreshold(img_bin_cropped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, img_bin_cropped.shape[1]//100, 0)
show_image(mean_thresh, 'Mean Adaptive Thresholding')

# find the colume that max intenity compare with its left and right side
mean_cols = np.mean(mean_thresh, axis=0)
plt.plot(mean_cols)
plt.title('Mean of Columns Intensity')
plt.xlabel('Column Index')
plt.ylabel('Mean of Intensity')
plt.show()

lowest_peaks = find_peaks(-mean_cols, distance=100, prominence=5)
print(lowest_peaks)

# cv2.line(img_bin_cropped, (200,0), (200, img_bin_cropped.shape[0]), (255), line_thickness)
# cv2.line(img_bin_cropped, (771,0), (771, img_bin_cropped.shape[0]), (255), line_thickness)  
# show_image(img_bin_cropped, 'Max Column Line')