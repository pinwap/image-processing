import cv2, numpy as np, matplotlib.pyplot as plt

def show_image(image, title='Image'):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, image.shape[1]//2, image.shape[0]//2)
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def line_lung_area(image, root, side='top', intensity_threshold=50, line_thickness=5):
    if side == 'bottom':
        bottom = 0
        for i in range(root,image.shape[0],+5):
            row_mean = image[i].mean()
            if row_mean < intensity_threshold:
                bottom = i
                print("Found row to crop at: ", bottom)
                ### Draw a line where we want to crop
                cv2.line(image, (0,bottom), (image.shape[1], bottom), (255), line_thickness)
                break
        return bottom, image

    elif side == 'top':
        top = 0
        for i in range(root,0,-5):
            row_mean = image[i].mean()
            if row_mean < intensity_threshold:
                top = i
                print("Found row to crop at: ", top)
                ### Draw a line where we want to crop
                cv2.line(image, (0,top), (image.shape[1], top), (255), line_thickness)
                break
        return top, image
    
    elif side == 'left':
        left = image.shape[1]//2
        for i in range(root,0,-5):
            column_mean = image[:, i].mean()
            if column_mean > intensity_threshold:
                left = i
                print("Found column to crop at: ", left)
                ### Draw a line where we want to crop
                cv2.line(image, (left, 0), (left, image.shape[0]), (0), line_thickness)
                break
        return left, image
    else: # right
        right = image.shape[1]//2
        for i in range(root, image.shape[1], +5):
            column_mean = image[:, i].mean()
            if column_mean > intensity_threshold:
                right = i
                print("Found column to crop at: ", right)
                ### Draw a line where we want to crop
                cv2.line(image, (right, 0), (right, image.shape[0]), (0), line_thickness)
                break
        return right, image       

def OTSU_threshold(image):
    inv = image.copy()
    _, image = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("OTSU threshold value: ", _)
    return image  


# Read the input image in grayscale
img = cv2.imread('data/testdata1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('data/testdata2.jpg', cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread('data/testdata3.jpg', cv2.IMREAD_GRAYSCALE)
img4 = cv2.imread('data/testdata4.jpg', cv2.IMREAD_GRAYSCALE)

img = img

# apply CLAHE to enhance the contrast of the image
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enh = clahe.apply(img)
# show_image(enh, 'Enhanced Image')

# 1. Crop the image to focus on the chest area
## 1.1 OTSU threshold and crop top-bottom
img_bin = OTSU_threshold(enh)
# show_image(img_bin, 'Initial Lung Binary Mask')

## draw top and bottom crop lines
bottom, img_bin_line_bottom = line_lung_area(img_bin, side='bottom', intensity_threshold=50, root=img_bin.shape[0]//2, line_thickness=5)
top, img_bin_line_top = line_lung_area(img_bin_line_bottom, side='top', intensity_threshold=50, root=img_bin.shape[0]//2, line_thickness=5)

enh_cropped = enh[top:bottom, : ]
show_image(enh_cropped, 'Cropped Enhanced Image')

img_bin2 = OTSU_threshold(enh_cropped)
img_bin2 = cv2.GaussianBlur(img_bin2, (5, 5), 0)
show_image(img_bin2, 'Re-Thresholded Cropped Image')

## 1.2 Isolate lung area
im_flood = img_bin2
h, w = img_bin2.shape
mask = np.zeros((h+2, w+2), np.uint8)

# flood fill จากมุมภาพ ซึ่งเป็นพื้นหลังแน่นอน
cv2.floodFill(im_flood, mask, seedPoint=(w//2, h-1), newVal=255)
# show_image(im_flood, 'Floodfilled Image')

holes = cv2.bitwise_not(im_flood)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((holes > 0).astype('uint8')*255, connectivity=8)
# show_image(holes, 'Holes Image')
print("Number of connected components (including background): ", num_labels)
print("Stats: ", stats)

areas = stats[1:, cv2.CC_STAT_AREA]             # ข้าม label 0
labels_ids = np.arange(1, num_labels)

cx_center = w / 2

best_label = None
best_score = -1

for lbl, area, (cx, cy) in zip(labels_ids, areas, centroids[1:]):
    # ให้คะแนนจาก "ขนาดใหญ่" และ "อยู่ใกล้กลางแนวนอน"
    size_score = area
    center_score = -abs(cx - cx_center) * 5     # ยิ่งใกล้กลางยิ่งดี
    score = size_score + center_score

    if score > best_score:
        best_score = score
        best_label = lbl

lung_mask_most_left = stats[best_label][cv2.CC_STAT_LEFT]
print("Lung mask most left: ", lung_mask_most_left)
# lung mask center cx cy of best_label
cx_lung, cy_lung = centroids[best_label + 1]

lung_mask = np.zeros_like(img_bin2)
lung_mask[labels == best_label] = 255 
# show_image(lung_mask, 'Isolated Lung Mask')
lung_mask_inv = cv2.bitwise_not(lung_mask)
show_image(lung_mask_inv, 'Final Lung Mask')

## 1.3 Crop left-right
root_lung = int((cx_lung + cy_lung) // 2)
# print ("Root lung for left-right cropping: ", root_lung)
left, img_bin_line_left = line_lung_area(lung_mask_inv, side='left', intensity_threshold=220, root=root_lung, line_thickness=5)
# img_bin_line_left_crop = lung_mask_inv[:, left:]
right, img_bin_line_right = line_lung_area(lung_mask_inv, side='right', intensity_threshold=220, root=root_lung, line_thickness=5)

show_image(img_bin_line_right, 'Final Crop Lines')
final_crop = enh_cropped[:, left:right]
show_image(final_crop, 'Final Cropped Image')