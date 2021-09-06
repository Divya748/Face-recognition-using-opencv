import cv2
import os
dataset = "dataset"
name = "anupama"

path = os.path.join(dataset, name)
if not os.path.isdir(path):
    os.mkdir(path)

(width, height) = (150,150)
algo = "haarcascade_frontalface_default.xml"
har_cascade = cv2.CascadeClassifier(algo)
cam = cv2.VideoCapture(0)

count = 1

while count < 500:
    print(count)
    _,img = cam.read()
    rgb_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    face = har_cascade.detectMultiScale(rgb_img, 1.3, 4)
    for (x,y,w,h) in face:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),2)
        faceonly = rgb_img[y:y+h, x:x+w]
        resize_img = cv2.resize(faceonly, (width, height))
        cv2.imwrite("%s/%s.jpg"%(path,count),resize_img)
        count +=1
    cv2.imshow("FaceDetection",img)
    key = cv2.waitKey(10)
    if key == 27:
        break
print("Image captured Successfully")
cam.release()
cv2.destroyAllWindows()