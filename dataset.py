import os
import cv2
import numpy as np
import sqlite3

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam=cv2.VideoCapture(0)

def insertorupdate(Id,Name,age):               #function is for sqlite database
    conn=sqlite3.connect("sqlite.db")         #connect database
    cmd="SELECT * FROM STUDENTS WHERE ID="+str(Id)
    cursor=conn.execute(cmd)             #cursor to execute statement
    isRecordExist=0              #assume there is no record in our table
    for row in cursor:
        isRecordExist=1
    if(isRecordExist==1):                            #if there is a record exist in our table
        conn.execute("UPDATE STUDENTS SET Name=? WHERE Id=?",(Name,Id,))
        conn.execute("UPDATE STUDENTS SET age=? WHERE Id=?", (age, Id,))
    else:                                #if there is no record exist we insert the values
        conn.execute("INSERT INTO STUDENTS (Id,Name,age) values(?,?,?)",(Id,Name,age))

    conn.commit()
    conn.close()

#insert user defined values into table

Id=input('Enter User Id:')
Name=input('Enter User Name:')
age=input('Enter User Age:')

insertorupdate(Id,Name,age)

#detect face in web camera coding

sampleNum=0;              #assume ther is no samples in dataset
while(True):
    ret,img=cam.read();                    #OPEN CAMERA
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)             # IMAGE CONVERT INTO BGRGRAY COLOR
    faces=faceDetect.detectMultiScale(gray,1.3,5)            #scale faces
    for(x,y,w,h) in faces:
        sampleNum=sampleNum+1;              #if face is detected incremnets
        cv2.imwrite("dataset/user."+str(Id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.waitKey(100)                     #delay time
    cv2.imshow("Face",img)                 #show faces detected in web camera
    cv2.waitKey(1);
    if(sampleNum>20):                 #if asdjhassjdhahsghdhasgd
        break;

cam.release()
cv2.destroyAllWindows() 