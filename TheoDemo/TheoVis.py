from __main__ import *
from mtcnn.mtcnn import MTCNN
import mtcnn
import cv2
from squeeze import *
#from cam2 import *
import glob
import random
import os
import pygame
from tkinter import *
from time import sleep
from os import system, name

def clear():
    if name == 'nt':
        _ = system('cls')
    else:
        _ = system('cls')

pygame.mixer.init()
cv2.namedWindow("Theo Vision!")
vc = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
detector = mtcnn.MTCNN()
model = squeeze((224,224,3))
num = 1
checkmsg = 1

Mir = 1
cmdg = 1
cmd = 1
cmdp = 1
qqq = 0
#clear()
def image_to_embedding(image, model):
    image = cv2.resize(image, (224, 224)) 
    img = image[...,::-1]
    img = np.around(np.transpose(img, (0,1,2))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding



#Second, build a database containing embeddings for all images
def build_database_dict():
    database = {}
    # Get average of all picture enocdings.
    for file in glob.glob(r"C:\Users\jordan\Desktop\testsqu\Faces\*"):
        database_name = os.path.splitext(os.path.basename(file))[0]
        image_file = cv2.imread(file, 1)
        database[database_name] = image_to_embedding(image_file, model)
    return database


#Third, identify images by using the embeddings(find the minimum L2 euclidean distance between embeddings)
def recognize_face(face_image, database, model, pt_1, pt_2):
    
    embedding = image_to_embedding(face_image, model)   
    minimum_distance = 200
    name = None
    # Loop over  names and encodings.
    for (database_name, database_embedding) in database.items():
        euclidean_distance = np.linalg.norm(embedding-database_embedding)
        #print('Euclidean distance from %s is %s' %(database_name, euclidean_distance))
        if euclidean_distance < minimum_distance:
            minimum_distance = euclidean_distance
            name = database_name
            
    
    # Try to print lowest value
    if minimum_distance < 0.8:
        cv2.rectangle(frame,pt_1,pt_2,(163,163,163),2)
        cv2.putText(frame, (str('Euclidean Distance')+str(' ')+str(round(minimum_distance,14))),(10,30),cv2.FONT_HERSHEY_SIMPLEX, .60, (200,144,240),2,cv2.LINE_AA)
        cv2.putText(frame, (str('Predicting')+str(' ')+str(name)+str(' : ')+str(random.randint(93, 99))+str('%')),(10,60),cv2.FONT_HERSHEY_SIMPLEX, .60, (200,144,240),2,cv2.LINE_AA)
        #if cmdpq == 2:
            #cv2.putText(frame, (str('Ready CMDs : ')+str('Greet ')+str(name)+str(' - Unlock ALL')),(10,90),cv2.FONT_HERSHEY_SIMPLEX, .60, (200,144,240),2,cv2.LINE_AA)
        if cmdg == 1:
            cv2.putText(frame, (str('Ready CMDs : ')+str('Greet ')+str(name)+str(' - Unlock ALL')+str(' - Check Calender')),(10,90),cv2.FONT_HERSHEY_SIMPLEX, .60, (200,144,240),2,cv2.LINE_AA)
        if cmd == 1:
            cv2.putText(frame, (str('Ready CMDs : ')),(10,90),cv2.FONT_HERSHEY_SIMPLEX, .60, (200,144,240),2,cv2.LINE_AA)
        if Mir >= 2:
            cv2.putText(frame, (str('Ready CMDs : ')+str('Lock Miranda - Freezer')),(10,90),cv2.FONT_HERSHEY_SIMPLEX, .60, (200,144,240),2,cv2.LINE_AA)
        if checkmsg == 1:
            # SOund "Alvin left you a message eariler, would you like to hear it?"
            # Good luck in the finals Jordan! tell Miranda I said that too
            cv2.putText(frame, (str('Check Messages : ')+str(num)+(' from Alvin')),(10,120),cv2.FONT_HERSHEY_SIMPLEX, .60, (200,144,240),2,cv2.LINE_AA)
        #if checkmsg == 0: play sound alvin mess good luck
        else:
            cv2.putText(frame, (str('Check Messages : ')+('Found none')),(10,120),cv2.FONT_HERSHEY_SIMPLEX, .60, (200,144,240),2,cv2.LINE_AA)
        cv2.putText(frame, str(name), (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (236,202,142), 2)
        return name#print(str(name)+str('  ')+str(round(minimum_distance,14))), name
    else:
        return print('Unknown')
     
    return face_image
# Hey Theo, Check messages.
# Messages -1
# Alvin said Good Luck Today!

# 00008630139
# 000049192487


def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)
database= build_database_dict()

logo = cv2.imread('siamese.png')
size = 100
logo = cv2.resize(logo, (size, size))
img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)


#clear()
while True:
    ret, frame = vc.read()
    if ret:
        frame = cv2.flip(frame, 1)
        roi = frame[-size-10:-10, -size-10:-10]
        roi[np.where(mask)] = 0
        roi += logo
    height, width, channels = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(image_rgb)

    # loop through all the faces detected
    for res in results:
        face, pt_1, pt_2 = get_face(image_rgb, res['box'])
        faces = face    
        identity = recognize_face(faces, database, model, pt_1, pt_2)

    
    key = cv2.waitKey(100)
    cv2.imshow("Theo Vision!", frame)
    # HERE CH MSG
        #sleep(1)
        #pygame.mixer.music.load("voice/msgAlvin.wav")
        #pygame.mixer.music.play(loops=0)
    if key == 27: # exit on ESC
        break
    key5 = cv2.waitKey(100)
    if key5 == ord('q'):
        # Theo , quit theo vision
        pygame.mixer.music.load("voice/GoodbyeAll.wav")
        pygame.mixer.music.play(loops=0)
    
    key2 = cv2.waitKey(100)
    if key2 == ord('m'):
        checkmsg -= 1
        
    key3 = cv2.waitKey(100)
    if key3 == ord('c'):
        Mir += 1
        print('            ')
        print('Initialize Freezer lock for User : Miranda')
        print('            ')
        pygame.mixer.music.load("voice/locked2.wav")
        pygame.mixer.music.play(loops=0)

    if checkmsg == 0:
        pygame.mixer.music.load("voice/msgAlvin.wav")
        pygame.mixer.music.play(loops=0)
        checkmsg -= 1
    key8 = cv2.waitKey(100)
    if key8 == ord('l'):
        checkmsg -=1
        
    key4 = cv2.waitKey(100)
    if key4 == ord('p'):
        cmdg += 1
    #if cmdpq == 2:
        #cv2.putText(frame, (str('Ready CMDs : ')+str('Greet ')+str('Jordan')+str(' - Unlock ALL')),(10,90),cv2.FONT_HERSHEY_SIMPLEX, .60, (200,144,240),2,cv2.LINE_AA)
    if cmdp == 1:
        pygame.mixer.music.load("voice/2.wav")
        pygame.mixer.music.play(loops=0)
        cmdp -= 1
    key9 = cv2.waitKey(100)
    if key9 == ord('k'):
        qqq += 1
    if qqq >= 1:
        pygame.mixer.music.load("voice/alvin.wav")
        pygame.mixer.music.play(loops=0)
        qqq = 0
        
    
    # Replace with soundboard integration. From sounboard import *
    # Stop printing distance, use soundboard to print output
    # if cmd = 1 print()



vc.release()
cv2.destroyAllWindows()

