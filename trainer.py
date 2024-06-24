import os                    #read and write file
import cv2                  #open camera
import numpy as np          #array
from PIL import Image       #image file read and write


recognizer=cv2.face.LBPHFaceRecognizer_create()         #it recognise the faces in camera
path ="dataset"


def get_images_with_id(path):
    images_paths=[os.path.join(path,f) for f in os.listdir(path)]    #set images path to os
    faces=[]
    ids=[]
    for single_image_path in images_paths:
        faceImg=Image.open(single_image_path).convert('L')             #image converted into gray color  l= luminance
        faceNp=np.array(faceImg,np.uint8)
        id=int(os.path.split(single_image_path)[-1].split(".")[1])
        print(id)
        faces.append(faceNp)
        ids.append(id)
        cv2.imshow("Training",faceNp)
        cv2.waitKey(100)

    return np.array(ids),faces

ids,faces=get_images_with_id(path)
recognizer.train(faces,ids)
recognizer.save("recognizer/trainingdata.yml")
cv2.destroyAllWindows()





