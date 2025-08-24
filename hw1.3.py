import cv2
import numpy as np
# enhance the 8-bit gray scale images using the power-law transformationwith different combinations of 𝑐 and 𝛾 when 
# 𝑐 = 0.4, 1, 1.6 and 𝛾 = 0.3, 2.4
# s=c*r^𝛾
def power_law_transformation(image, c, gamma):
    # สร้าง lookup table
    lut = np.arange(256, dtype=np.uint8)

    # ใช้สมการ power-law transformation เพื่อเติมค่าใน lookup table
    lut = c * (lut ** gamma)

    # จำกัดค่าให้อยู่ในช่วง 0-255 และแปลงเป็น uint8
    lut = np.clip(lut, 0, 255).astype(np.uint8)

    # ใช้ lookup table กับภาพ
    transformed_image = cv2.LUT(image, lut)
    return transformed_image

images = ["hw1pic/fish.jpg", "hw1pic/amusement-park.jpg", "hw1pic/cartoon.jpg"]
c_values = [0.4, 1, 1.6]
gamma_values = [0.3, 2.4]

for image_path in images:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Original Image', image)
    for c in c_values:
        for gamma in gamma_values:
            transformed_image = power_law_transformation(image, c, gamma)
            cv2.imshow(f"c={c}, gamma={gamma}", transformed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()