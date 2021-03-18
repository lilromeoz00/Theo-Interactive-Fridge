import cv2
from reshape import *
from time import sleep
from os import system, name

def clear():
    if name == 'nt':
        _ = system('cls')
    else:
        _ = system('cls')

cam = cv2.VideoCapture(0)

#cv2.namedWindow("test")

img_counter = 0

for i in range(1):
    sleep(1)
    clear()
    print('What is your name?')
    name = input()

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    #print(img_counter)
    # ADD PYGAME ON CLICK GUI TAKE PICTURE!!!
    #cv2.putText(frame, str('Press SPACE to capture image'),(10,60),cv2.FONT_HERSHEY_SIMPLEX, .75, (255,255,255),2,cv2.LINE_AA)
    if img_counter == 0:
        cv2.putText(frame, str('Press SPACE when you want to take a picture {} !'.format(name)),(10,60),cv2.FONT_HERSHEY_SIMPLEX, .60, (255,255,255),2,cv2.LINE_AA)
        #play sound (LET ME KNOW)
        pass
    if img_counter == 1:
        cv2.putText(frame, str('Took ')+str(img_counter)+str(' picture'),(10,60),cv2.FONT_HERSHEY_SIMPLEX, .60, (255,255,255),2,cv2.LINE_AA)
    if img_counter == 2:
        #playsound(TOOK PICTURE)
        cv2.putText(frame, str('Took ')+str(img_counter)+str(' pictures'),(10,60),cv2.FONT_HERSHEY_SIMPLEX, .60, (255,255,255),2,cv2.LINE_AA)
    if img_counter == 3:
        #playsound(TOOK PICTURE)
        cv2.putText(frame, str('Took ')+str(img_counter)+str(' pictures'),(10,60),cv2.FONT_HERSHEY_SIMPLEX, .60, (255,255,255),2,cv2.LINE_AA)
    if img_counter == 4:
        #playsound(TOOK PICTURE)
        cv2.putText(frame, str('Took ')+str(img_counter)+str(' pictures'),(10,60),cv2.FONT_HERSHEY_SIMPLEX, .60, (255,255,255),2,cv2.LINE_AA)    
    if img_counter == 5:
        #playsound(TOOK PICTURE)
        cv2.putText(frame, str('Took ')+str(img_counter)+str(' pictures'),(10,60),cv2.FONT_HERSHEY_SIMPLEX, .60, (255,255,255),2,cv2.LINE_AA)
    cv2.imshow("Theo Vision! - Setup New Friend", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        #Playsound(ill be sure to remeber you)
        break
    elif k%256 == 32:
        # SPACE pressed
        reshape(frame, name)
        img_counter += 1

cam.release()

cv2.destroyAllWindows()
