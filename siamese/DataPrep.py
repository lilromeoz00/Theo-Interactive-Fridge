import sys
import numpy as np
from matplotlib.pyplot import imread
import pickle
import os
import matplotlib.pyplot as plt
import cv2

data_path = os.path.join(r'C:\Users\jordan\CFP_Data')
train_folder = os.path.join(data_path,'train')
valpath = os.path.join(data_path,'val')

save_path = r'C:\Users\jordan\CFP_Data'

lang_dict = {}

from PIL import Image
def resize(path):
    n = 0
    for CFSet in os.listdir(path):
        print("loading CFSet: " + CFSet)
        CFSet_path = os.path.join(path,CFSet)
        for z, person in enumerate(os.listdir(CFSet_path)):
            person_path = os.path.join(CFSet_path, person)
            if not os.path.isdir('RTrain/{}'.format(z)):
                    pa = r'C:/Users/jordan/CFP_Data'
                    rt = os.path.join(pa, 'RTrain/{}'.format(z))
                    os.mkdir(rt)
                    # I made the RTrain folder myself and it worked
            for i, filename in enumerate(os.listdir(person_path)):
        
                image_path = os.path.join(person_path, filename)
                image = cv2.imread(image_path)
                image = np.array(image)
                image = cv2.resize(image, (224,224), interpolation=cv2.INTER_LINEAR)
                imager = np.float32(image)
                    
                cv2.imwrite('C:/Users/jordan/CFP_Data/RTrain/{}/{}.jpg'.format(z, i), imager)

                print('Folder', z, 'made & filled')
            # For anyone reading this, This function took me a lot of hours to get right :)
    return imager

def resizeV(path):
    n = 0
    for CFSet in os.listdir(path):
        print("loading CFSet: " + CFSet)
        CFSet_path = os.path.join(path,CFSet)
        for z, person in enumerate(os.listdir(CFSet_path)):
            person_path = os.path.join(CFSet_path, person)
            if not os.path.isdir('RVal/{}'.format(z)):
                    pa = r'C:/Users/jordan/CFP_Data'
                    rt = os.path.join(pa, 'RVal/{}'.format(z))
                    os.mkdir(rt)
                    # I made the RVal folder myself and it worked
            for i, filename in enumerate(os.listdir(person_path)):
        
                image_path = os.path.join(person_path, filename)
                image = cv2.imread(image_path)
                image = np.array(image)
                image = cv2.resize(image, (224,224), interpolation=cv2.INTER_LINEAR)
                imager = np.float32(image)
                    
                cv2.imwrite('C:/Users/jordan/CFP_Data/RVal/{}/{}.jpg'.format(z, i), imager)

                print('Folder', z, 'made & filled')
            # For anyone reading this, This function took me a lot of hours to get right :)
    return imager

resize(train_folder)
resizeV(valpath)

def grey(path):
    n = 0
    for CFSet in os.listdir(path):
        print("loading CFSet: " + CFSet)
        CFSet_path = os.path.join(path,CFSet)
        for z, person in enumerate(os.listdir(CFSet_path)):
            person_path = os.path.join(CFSet_path, person)
            if not os.path.isdir('GTrain/{}'.format(z)):
                    pa = r'C:/Users/jordan/CFP_grey'
                    rt = os.path.join(pa, 'GTrain/{}'.format(z))
                    os.mkdir(rt)
            else:
                pass
            for i, filename in enumerate(os.listdir(person_path)):
        
                image_path = os.path.join(person_path, filename)
                image = cv2.imread(image_path)
                image = np.array(image)
                image = cv2.resize(image, (224,224), interpolation=cv2.INTER_LINEAR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = np.float32(image)
                    
                cv2.imwrite('C:/Users/jordan/CFP_grey/GTrain/{}/{}.jpg'.format(z, i), image)

                print('Folder', z, 'made & filled')
    return image

def greyV(path):
    n = 0
    for CFSet in os.listdir(path):
        print("loading CFSet: " + CFSet)
        CFSet_path = os.path.join(path,CFSet)
        for z, person in enumerate(os.listdir(CFSet_path)):
            person_path = os.path.join(CFSet_path, person)
            if not os.path.isdir('GVal/{}'.format(z)):
                    pa = r'C:/Users/jordan/CFP_grey'
                    rt = os.path.join(pa, 'GVal/{}'.format(z))
                    os.mkdir(rt)
            else:
                pass
            for i, filename in enumerate(os.listdir(person_path)):
        
                image_path = os.path.join(person_path, filename)
                image = cv2.imread(image_path)
                image = np.array(image)
                image = cv2.resize(image, (224,224), interpolation=cv2.INTER_LINEAR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                imager = np.float32(image)
                    
                cv2.imwrite('C:/Users/jordan/CFP_grey/GVal/{}/{}.jpg'.format(z, i), imager)

                print('Folder', z, 'made & filled')
    return imager

grey(train_folder)
greyV(valpath)

image_path = r'C:\Users\jordan\CFP_grey\GVal\CFSet\43\8.jpg'
image = cv2.imread(image_path)
print(image.dtype)
print(image.shape)

# So I want to go from (224,224,3) to (224,224,1)
image = cv2.imread(image_path)
image = image[:,:,1]
print(image.shape)
print(image)
import matplotlib.pyplot as plt

plt.imshow(image)
plt.show()
# It works, will try to do this for all


data_path = os.path.join(r'C:\Users\jordan\CFP_grey')
train_folder = os.path.join(data_path,'GTrain')
valpath = os.path.join(data_path,'GVal')

save_path = r'C:\Users\jordan\CFP_grey'

# Make script like above that turns img into 2x2
def scalT(path):
    n = 0
    for CFSet in os.listdir(path):
        print("loading CFSet: " + CFSet)
        CFSet_path = os.path.join(path,CFSet)
        for z, person in enumerate(os.listdir(CFSet_path)):
            person_path = os.path.join(CFSet_path, person)
            if not os.path.isdir('VTrain/{}'.format(z)):
                    pa = r'C:/Users/jordan/CFP_grey'
                    rt = os.path.join(pa, 'VTrain/{}'.format(z))
                    os.mkdir(rt)
            else:
                pass
            for i, filename in enumerate(os.listdir(person_path)):
        
                image_path = os.path.join(person_path, filename)
                image1 = cv2.imread(image_path)
                image2 = image1[:,:,1]
                cv2.imwrite('C:/Users/jordan/CFP_grey/VTrain/{}/{}.jpg'.format(z, i), image2)
                hath = r'C:/Users/jordan/CFP_grey/VTrain/0/0.jpg'
                image3 = cv2.imread(hath)
                print(image3.shape)
    return image2

scalT(train_folder)

image_path = r'C:\Users\jordan\CFP_grey\VTrain\0\0.jpg'
image = cv2.imread(image_path)
print(image.dtype)
print(image.shape)
print(image)

plt.imshow(image)
plt.show()
# Well that was a fail, guess we just gotta deal with rgb then
