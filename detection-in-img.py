import cv2
import os

image_path = os.path.join(os.path.dirname(__file__), "portrait.jpg")

img = cv2.imread(image_path)

cv2.imshow("face", img)
cv2.waitKey(1000)

face_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades,
                                     "haarcascade_frontalface_default.xml"))

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("face", gray_img)
cv2.waitKey(1000)

faces = face_cascade.detectMultiScale(gray_img,
                                      scaleFactor=1.3,
                                      minNeighbors=5)

for x, y, w, h in faces:
    img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=3)

cv2.imshow("face", img)
cv2.waitKey()

cv2.destroyAllWindows()
