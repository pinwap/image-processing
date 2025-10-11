import cv2
import numpy as np
import matplotlib.pyplot as plt

# โหลดภาพ (grayscale)
img = cv2.imread("your_image.jpg", 0)

# 1. คำนวณ DFT และ shift ศูนย์ความถี่มาไว้กลาง
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# 2. คำนวณ magnitude spectrum
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))

# แสดงภาพ
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.axis('off')

plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.axis('off')
plt.show()

# 3. การสร้าง Low-pass filter mask
rows, cols = img.shape
crow, ccol = rows//2 , cols//2
mask = np.zeros((rows, cols, 2), np.uint8)
r = 30  # radius ของวงกลม (Low-pass)
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = (x - center[0])**2 + (y - center[1])**2 <= r*r
mask[mask_area] = 1

# 4. Apply filter
fshift = dft_shift * mask

# 5. Inverse DFT
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])

# แสดงผล
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.axis('off')

plt.subplot(122), plt.imshow(img_back, cmap='gray')
plt.title('Low Pass Filtered Image'), plt.axis('off')
plt.show()