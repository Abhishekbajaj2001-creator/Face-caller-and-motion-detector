import os
from PIL import Image
import numpy as np
import pickle
import cv2


base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(base_dir, "faces train")

face_cascade=cv2.CascadeClassifier("C:\\Users\\abhi\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")
recognizer =cv2.face.LBPHFaceRecognizer_create()
current_id = 0
label_ids ={}
y_label=[]
x_train=[]

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-")
            #print(label, path)
            if not label in label_ids:
                label_ids[label]=current_id
                current_id+=1
            id_=label_ids[label]
           # print(label_ids)
            pil_image = Image.open(path).convert("L")
            num_images=np.array(pil_image,"uint8")
            #print(num_images)
            faces=face_cascade.detectMultiScale(num_images,1.5,5)

            for x,y,w,h in faces:
                roi=num_images[y:y+h,x:x+w]
                x_train.append(roi)
                y_label.append(id_)





file= open('labels.pickle','wb')
pickle.dump(label_ids, file)

recognizer.train(x_train, np.array(y_label))
recognizer.save("trainer.yml")