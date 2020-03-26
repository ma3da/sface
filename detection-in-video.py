import cv2
import os

video_path = os.path.join(os.path.dirname(__file__), "ap_run.mp4")

img = cv2.imread(video_path)

cap = cv2.VideoCapture(video_path)

face_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades,
                                     "haarcascade_frontalface_default.xml"))

while True:
    _, img = cap.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(img_gray,
                                          scaleFactor=1.3,
                                          minNeighbors=5)

    for x, y, w, h in faces:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=3)

    cv2.imshow("face", img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
