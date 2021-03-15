import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import ZeroPadding2D,Convolution2D,MaxPooling2D
from tensorflow.keras.layers import Dense,Dropout,Softmax,Flatten,Activation,BatchNormalization
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Lambda, Input, Activation, GlobalAvgPool2D, MaxPool2D, Input, Add, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Concatenate, Lambda, add, Convolution2D, concatenate, ZeroPadding2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K


def scaling(x, scale):
	return x * scale

def squeeze(inputShape): #inputShape inside
    
        def fire(x, fs, fe):
                s = Conv2D(fs, 1, activation='relu')(x)
                e1 = Conv2D(fe, 1, activation='relu')(s)
                e3 = Conv2D(fe, 3, padding='same', activation='relu')(s)
                output = concatenate([e1, e3])
                return output
    
        input = Input(inputShape)

        x = Conv2D(96, 7, strides=2, padding='same', activation='relu')(input)
        x = MaxPool2D(3, strides=2, padding='same')(x)

        x = fire(x, 16, 64)
        x = fire(x, 16, 64)
        x = fire(x, 32, 128)
        x = MaxPool2D(3, strides=2, padding='same')(x)

        x = fire(x, 32, 128)
        x = fire(x, 48, 192)
        x = fire(x, 48, 192)
        x = fire(x, 64, 256)
        x = MaxPool2D(3, strides=2, padding='same')(x)

        x = fire(x, 64, 256)
        x = Conv2D(450, 1)(x)

        x = GlobalAvgPool2D()(x)
        output = Dense(450)(x)

        model = Model(input, output)
        tf.keras.models.load_model(r'C:\Users\jordan\Desktop\Siamv1')
        #tf.keras.models.load_model(r'C:\Users\jordan\Desktop\squeeze\Siamesev1.h5')
        #model.load_weights(r'C:\Users\jordan\Desktop\squeeze\Siamesev1.h5')
        return model
