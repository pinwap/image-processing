#อ่านภาพ
import cv2
img = cv2.imread("image/IMG_20200111_135346.jpg")

#ปรับเปลี่ยนขนาดภาพ
imgResized = cv2.resize(img,(400,400) )#ภาพ,(กว้างมยาว)

# แสดงผลภาพ
cv2.imshow("Output",imgResized) #title , image
cv2.waitKey(delay = 5000)#wait until dalay(milli second) to close the image ถ้าไม่อยากให้ปิดใส่ 0
cv2.destroyAllWindows() #คืนทรัพยากรให้เครื่อง