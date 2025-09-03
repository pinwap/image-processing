import cv2

img = cv2.imread("image/IMG_20200111_135346.jpg")   

eye_cascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#จะตรวจจับอะไรได้ดีกว่าเมื่อเป็นภาพสีเทา

# ตรวจจับหน้าในภาพ
scaleFactor = 1.1 # ค่าที่ใช้ในการปรับขนาดภาพให้ลดลง 1.1 = ลด10%
minNeighbors = 3 #ต้องมีความคล้ายคลึงใบหน้ากี่จุด ถึงจะถือว่าเป็นตา
eye_detect = eye_cascade.detectMultiScale(gray, scaleFactor, minNeighbors)

for (x, y, w, h) in eye_detect:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

cv2.imshow("Output", img)
cv2.waitKey(0)
cv2.destroyAllWindows()