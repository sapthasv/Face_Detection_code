import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # this xml file should be in working directory or elese get an error
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') # same here too , this file to detect eyes

img = cv2.imread('group.jpg') # give input sample image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # coverting binary to gray

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow('img',img) # printing file
cv2.waitKey(0) # waits until user presses any key
cv2.destroyAllWindows()