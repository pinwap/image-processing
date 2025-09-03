import cv2
import numpy as np
import matplotlib.pyplot as plt
# enhance the 8-bit gray scale images using the power-law transformationwith different combinations of 𝑐 and 𝛾 when 
# 𝑐 = 0.4, 1, 1.6 and 𝛾 = 0.3, 2.4
# s=c*r^𝛾
def power_law_transformation(image, c, gamma):
    # Normalize เป็น [0,1]
    norm = image / 255.0
    # Apply power-law
    out = c * np.power(norm, gamma)
    # Scale กลับ [0,255]
    out = np.clip(out * 255, 0, 255).astype(np.uint8)
    return out

images = ["hw1pic/fish.jpg", "hw1pic/amusement-park.jpg", "hw1pic/cartoon.jpg"]
c_values = [0.4, 1, 1.6]
gamma_values = [0.3, 2.4]

for image_path in images:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # cv2.imshow('Original Image', image)
    # for c in c_values:
    #     for gamma in gamma_values:
    #         transformed_image = power_law_transformation(image, c, gamma)
    #         cv2.imshow(f"c={c}, gamma={gamma}", transformed_image)
    #         cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Plot the original and transformed images
    plt.figure(figsize=(12, 8))
    plt.subplot(4, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image'), plt.axis('off')

    for i in range(len(c_values)):
        for j in range(len(gamma_values)):
            transformed_image = power_law_transformation(image, c_values[i], gamma_values[j])
            plt.subplot(4, 2, 3+2*i+j)
            plt.imshow(transformed_image, cmap='gray')
            plt.title(f"c={c_values[i]}, gamma={gamma_values[j]}")
            plt.axis('off')
    plt.tight_layout()
    plt.show()