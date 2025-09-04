import cv2
import numpy as np
import matplotlib.pyplot as plt

# Write a program to enhance an image so that the dark part on the right side of an image brings out more details using:
# 1. global histogram equalization
img = cv2.imread('D:/Pin/STUDY/0CU/Programming/image processing/HW2/hw2pic_Tungsten.jpg', 0)
global_eq = cv2.equalizeHist(img)

# 2. local histogram equalization with 3 neighborhood sizes: 3x3, 7x7, and 11x11.
# Find the values for 𝑘0, 𝑘1, and 𝑘2 that you think are the most suitable values.

# เลือกพิกเซลที่จะ enhance ต้องมีคุณสมบัติ
# 1. ค่าเฉลี่ย local น้อยกว่าค่าเฉลี่ย global
# 2. local sd อยู่ระหว่าง global sd

# หา global mean, sd
global_mean = np.mean(img)
global_sd = np.std(img)

def local_hist_eq(img, grid_size, k0, k1, k2, global_mean, global_sd):
    # หา local mean, sd
    m = cv2.blur(img.astype(np.float32),(grid_size, grid_size)) #เบลอ = ค่าเฉลี่ยทุกพิกเซล = local mean = E(x)
    m2 = cv2.blur((img.astype(np.float32)**2),(grid_size, grid_size)) # E(x^2)
    local_sd = np.sqrt(m2 - m**2) #sd = sqrt(E(x^2) - E(x)^2)

    # เงื่อนไขการ enhance
    condition = (m<k0*global_mean)&(local_sd>=k1*global_sd)&(local_sd<=k2*global_sd)

    # ทำ histogram equalization ทั้งภาพก่อน
    he = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(grid_size, grid_size)).apply(img)
    enhanced_img = img.copy()
    enhanced_img[condition] = he[condition]

    return enhanced_img

# 3. local gamma collection with 3 neighborhood sizes: 5x5, 9x9, and 15x15. Find the most suitable value for gamma.
def local_gamma_correction(img, global_mean, grid_size, gamma, thr_ratio):
    # ทำ gamma correction เฉพาะโซนที่ local mean < thr_ratio*global_mean
    local_mean = cv2.blur(img.astype(np.float32), (grid_size, grid_size))
    condition = local_mean < thr_ratio * global_mean

    # gamma correction
    norm = img / 255.0
    gamma_img = np.power(norm, gamma)
    gamma_img = np.clip(gamma_img * 255, 0, 255).astype(np.uint8)

    # ใส่ condition
    out = img.copy()
    out[condition] = gamma_img[condition]
    
    return out

cv2.imshow('Original Image', img)
cv2.imshow('Global Histogram Equalization', global_eq)
 

# การเลือก k0แปรผันตรงกับค่าเฉลี่ยความสว่าง k2มากลายน้อย 
cv2.imshow('Local Histogram Equalization 3*3', local_hist_eq(img, 3, 0.4, 1.0, 0.3, global_mean, global_sd))
cv2.imshow('Local Histogram Equalization 7*7', local_hist_eq(img, 7, 0.7, 1.5, 0.4, global_mean, global_sd))
cv2.imshow('Local Histogram Equalization 11*11', local_hist_eq(img, 11, 0.1, 1.5, 0.4, global_mean, global_sd))
# cv2.imshow('Local Histogram Equalization 2', local_hist_eq(img, 11, 0.1, 0.5, 0.4, global_mean, global_sd))
# cv2.imshow('Local Histogram Equalization 3', local_hist_eq(img, 11, 0.1, 1, 0.4, global_mean, global_sd))
# cv2.imshow('Local Histogram Equalization 4', local_hist_eq(img, 11, 0.1, 1.5, 0.4, global_mean, global_sd))
# cv2.imshow('Local Histogram Equalization 5', local_hist_eq(img, 11, 0.1, 2, 0.4, global_mean, global_sd))

# การเลือก thr_ratio แปรผันตรงกับค่าเฉลี่ยความสว่าง 
cv2.imshow('Local Gamma correction 5*5', local_gamma_correction(img, global_mean, 5, 0.85, 0.8)) # 0.85
cv2.imshow('Local Gamma correction 9*9', local_gamma_correction(img, global_mean, 9, 0.7, 1.9))
cv2.imshow('Local Gamma correction 15*15', local_gamma_correction(img, global_mean, 15, 0.7, 0.5))
# cv2.imshow('Local Gamma correction 9*9 2', local_gamma_correction(img, global_mean, 9, 0.7, 0.8))

cv2.waitKey(0)
cv2.destroyAllWindows()


# test
# print('global mean:', global_mean, 'global sd:', global_sd)

# m = cv2.blur(img.astype(np.float32),(3, 3)) #เบลอ = ค่าเฉลี่ยทุกพิกเซล = local mean = E(x)
# m2 = cv2.blur((img.astype(np.float32)**2),(3, 3)) # E(x^2)
# local_sd = np.sqrt(m2 - m**2) #sd = sqrt(E(x^2) - E(x)^2)

# condition = (m<0.7*global_mean)&(local_sd>=1.5*global_sd)&(local_sd<=0.4*global_sd).astype(np.uint8)
# print(condition)
# he = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(7, 7)).apply(img)
# print(he)
# print(img)
# enhanced_img = img.copy()
# enhanced_img[condition.astype(bool)] = he[condition.astype(bool)]
# print(enhanced_img)