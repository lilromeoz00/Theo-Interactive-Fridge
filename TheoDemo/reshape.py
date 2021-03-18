from mtcnn.mtcnn import MTCNN
import cv2
from squeeze import *
import glob
import os

# For loop jordan{}.format(z)

detector1= MTCNN()


     

def reshape(img, name):
    #imgq = cv2.imread(img)
    result = detector1.detect_faces(img)
    count = 0
    for person in result:
        bounding_box = person['box']
        x=bounding_box[0]
        y=bounding_box[1]
        w=bounding_box[2]
        h=bounding_box[3]
        keypoints = person['keypoints']
            
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,0), 2)
        cv2.imwrite("Faces/{}.jpg".format(name), img[y:y+h,x:x+w])
        count +=1
    return img
