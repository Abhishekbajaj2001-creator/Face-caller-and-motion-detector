import cv2,time
import pickle
import os
from gtts import gTTS

language='en'
first_frame=None
video = cv2.VideoCapture(0,cv2.CAP_DSHOW)
face_cascade=cv2.CascadeClassifier("path.xml")
recognizer =cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
label={}
fi= open('labels.pickle','rb')
olabel=pickle.load( fi)
label={ v:k for k,v in olabel.items()}

img =0
while True:
    check,frame=video.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    detect=face_cascade.detectMultiScale(gray,1.5,5)
    onetime=0
    for (x,y,w,h) in detect:
        roi_gray=gray[y:y+h,x:x+h]
        roi_color=frame[y:y+h,x:x+w]

        id_,conf=recognizer.predict(roi_gray)
        if conf>=45:

            print(id_)
            print(label[id_])
            mytext=label[id_]
            if onetime<1:
                myobj=gTTS(text=mytext,lang=language,slow=False)
                myobj.save("welcome.mp3")
                os.system("mpg321 welcome.mp3")
                onetime+=1
            cv2.putText(frame,label[id_],(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),3,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,x+h),(255,0,255),2)

    cv2.imshow("face detection",frame)
    if first_frame is None:
        first_frame=gray
        continue
    diff = cv2.absdiff(first_frame,gray)
    retreval,delta=cv2.threshold(diff,50,255,cv2.THRESH_BINARY)
    delta=cv2.dilate(delta,None,iterations=0)
    cnt,_=cv2.findContours(delta,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for contou in cnt:
        if cv2.contourArea(contou)<1000:
            continue
        (x,y,w,h)=cv2.boundingRect(contou)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

    #cv2.imshow('movement and face detection',frame)
    #cv2.imshow('gray',gray)
    #cv2.imshow('diffe',diff)
    #cv2.imshow('delta', delta)
    key=cv2.waitKey(1)
    if key==ord('q'):
        break
video.release()
cv2.destroyAllWindows()
