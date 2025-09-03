import cv2

img = cv2.imread("image/IMG_20200111_135346.jpg")

grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")

scaleFactor = 1.1
minNeighbors = 3
faces = face_cascade.detectMultiScale(grey, scaleFactor, minNeighbors)
eyes = eye_cascade.detectMultiScale(grey, scaleFactor, minNeighbors)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

for (x, y, w, h) in eyes:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), thickness=2)

cv2.imshow("Output", img)
cv2.waitKey(0)
cv2.destroyAllWindows()