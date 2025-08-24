import cv2
import matplotlib.pyplot as plt

#  โหลดภาพ
images = ["puppies.jpg", "lake.jpg", "cartoon.jpg"]
for img_path in images:
    img = cv2.imread('hw1pic/'+img_path)
    # img = cv2.imread('hw1pic/puppies.jpg')
    cv2.imshow('Original Image', img)
    
# แปลงเป็น grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Grayscale Image', gray)

# 1. replication
# 1.1 zoom the given images to be three times as big
    zoomed = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    cv2.imshow('Replication Zoomed Image', zoomed)
# 1.2 shrink the enlarged images to be one-third of their size
    shrinked = cv2.resize(zoomed, None, fx=1/3, fy=1/3, interpolation=cv2.INTER_NEAREST)
    cv2.imshow('Replication Shrinked Image', shrinked)

# 2. bilinear interpolation
# 2.1 zoom the given images to be three times as big
    zoomed_bilinear = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
    cv2.imshow('Bilinear Zoomed Image', zoomed_bilinear)
# 2.2 shrink the enlarged images to be one-third of their size
    shrinked_bilinear = cv2.resize(zoomed_bilinear, None, fx=1/3, fy=1/3, interpolation=cv2.INTER_LINEAR)
    cv2.imshow('Bilinear Shrinked Image', shrinked_bilinear)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # # แสดงผล
    # plt.subplot(3, 2, 1), plt.title('Original Image'), plt.imshow(img), plt.axis('off')
    # plt.subplot(3, 2, 2), plt.title('Grayscale Image'), plt.imshow(gray, cmap='gray'), plt.axis('off')
    # plt.subplot(3, 2, 3), plt.title('Replication Zoomed Image'), plt.imshow(zoomed, cmap='gray'), plt.axis('off')
    # plt.subplot(3, 2, 4), plt.title('Replication Shrinked Image'), plt.imshow(shrinked, cmap='gray'), plt.axis('off')
    # plt.subplot(3, 2, 5), plt.title('Bilinear Zoomed Image'), plt.imshow(zoomed_bilinear, cmap='gray'), plt.axis('off')
    # plt.subplot(3, 2, 6), plt.title('Bilinear Shrinked Image'), plt.imshow(shrinked_bilinear, cmap='gray'), plt.axis('off')
    # plt.show()