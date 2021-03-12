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

