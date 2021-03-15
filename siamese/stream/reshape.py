from mtcnn.mtcnn import MTCNN
import cv2
from squeeze import *
import glob
import os

# For loop jordan{}.format(z)
image= cv2.imread('Faces/trai.jpg')
detector1= MTCNN()
result=detector1.detect_faces(image)
print(result)

count=0
for person in result:
            bounding_box = person['box']
            x=bounding_box[0]
            y=bounding_box[1]
            w=bounding_box[2]
            h=bounding_box[3]
            keypoints = person['keypoints']
            
            cv2.rectangle(image, (x, y), (x+w, y+h), (0,0,0), 2)
            cv2.imwrite("Faces/Miranda.jpg", image[y:y+h,x:x+w])
            count +=1
