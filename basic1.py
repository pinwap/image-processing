# อ่านภาพ
import cv2
img = cv2.imread("image/IMG_20200111_135346.jpg")
# (ชื่อภาพ, รูปแบบสี) ถ้าไม่ใส่รูปแบบสีอ่านภาพสีก็เป็นภาพสี แต่ถ้าอยากเปลี่ยนรูปแบบสีภาพ 
# ใส่ 0 -> grey scale 
# 1 -> full color
# -1 -> เติมalpha channel


print(type(img.ndim))
# print(img) shows the metrix of the image
# img.ndim shows how many dimention the metrix are (n dimension) = 3
# type(img) shows ehat type is the data in matrix = int
