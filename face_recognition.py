import cv2
import os
import numpy as np

algo = "haarcascade_frontalface_default.xml"

har_cascade = cv2.CascadeClassifier(algo)
datasets = "dataset"
(images, labels, names, id) = ([],[],{},0)

for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id]=subdir
        subdir_path = os.path.join(datasets, subdir)
        for filename in os.listdir(subdir_path):
            path = subdir_path + "/" + filename
            label = id
            images.append(cv2.imread(path,0))
            labels.append(int(label))
        id = id+1

(images,labels) = [np.array(lis) for lis in [images, labels]]

(width, height) = (150,150)
#model = cv2.face.LBPHFaceRecognizer_create()
model = cv2.face.FisherFaceRecognizer_create()
print("Training.....")

model.train(images, labels)

cam = cv2.VideoCapture(0)
cnt =0
while True:
    _, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = har_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),2)
        face = gray[y:y+h, x:x+w]
        face_resize = cv2.resize(face, (width,height))
        prediction = model.predict(face_resize)
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),2)
        if prediction[1] < 800:
            cv2.putText(img, "%s - %.0f" %(names[prediction[0]], prediction[1]), (x-10,y-10),cv2.FONT_HERSHEY_TRIPLEX,0.5,(0,0,255),2)
            cnt =0
        else:
            cnt = cnt+1
            cv2.putText(img, "unknown" , (x-10,y-10),cv2.FONT_HERSHEY_TRIPLEX,0.5,(0,0,255),2 )
            if(cnt>100):
                cv2.imwrite("unknown.jpg",img)
                cnt = 0
    cv2.imshow('facerecognition', img)
    key = cv2.waitKey(10)
    if key == 27:
        break
cam.release()
cv2.destroyAllWindows()