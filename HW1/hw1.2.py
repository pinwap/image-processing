import cv2
import numpy as np
import matplotlib.pyplot as plt

# contrast stretching
def contrast_stretching(image, r1, s1, r2, s2):
    # สร้าง lookup table
    lut = np.arange(256, dtype=np.uint8)

    # หาสมการเส้นตรงตรงกลาง
    m = (s2 - s1) / (r2 - r1)
    c = s1 - m * r1

    # เติมค่าใน lookup table 
    lut[:r1] = s1
    lut[r1:r2] = m*lut[r1:r2] + c
    lut[r2:] = s1
    
    # จำกัดค่าให้อยู่ในช่วง 0-255 เผื่อ
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    
    # ใช้ lookup table กับภาพ
    stretched_image = cv2.LUT(image, lut)
    return stretched_image

images = ["fruit_tree.jpg", "cartoon2.png", "kittens.jpg"]
for img_path in images:
    img = cv2.imread('hw1pic/'+img_path)
    cv2.imshow('Original Image', img)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Grayscale Image', gray)
    
    #กำหนดค่าให้ตัวแปร โดยที่ 8 bit
    L = 256
    s1 = int(5 * L / 6)
    s2 = int(L / 6)
    r1 = int(L / 4)
    r2 = int(3 * L / 4)
    contrast_img = contrast_stretching(gray, r1, s1, r2, s2)
    cv2.imshow('Contrast Stretched Image', contrast_img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()