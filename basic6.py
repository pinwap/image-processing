# เปิดวีดีโอด้วย OpenCV 
import cv2

cap = cv2.VideoCapture("image/VID20201121121032.mp4")

while(cap.isOpened()):
    check , frame = cap.read() #รับภาพจากกล้อง frame ต่อ frame
    
    if check == True:
        ##เปลี่ยนเป็นสีเทา
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        cv2.imshow("Output", gray)
        if cv2.waitKey(1) & 0xFF == ord("e"):
           break
    else:
        break

cap.release()#close the camera
cv2.destroyAllWindows()