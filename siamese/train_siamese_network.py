#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Lambda, Input, Activation, GlobalAvgPool2D, MaxPool2D, Input, Add, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Concatenate, Lambda, add, Convolution2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K


# In[ ]:


def build_siamese_model(inputShape, embeddingDim=48):
    
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
    output = Dense(embeddingDim)(x)

    model = Model(input, output)
    return model


# In[2]:


XTrain = np.load('XTrain.npy')
YTrain = np.load('YTrain.npy')
XTest = np.load('XTest.npy')
YTest = np.load('YTest.npy')


# In[3]:


XTrain = XTrain / 255
XTest = XTest / 255


# In[4]:


def make_pairs(images, labels):
    # initialize two empty lists to hold the (image, image) pairs and
    # labels to indicate if a pair is positive or negative
    pairImages = []
    pairLabels = []
    # calculate the total number of classes present in the dataset
    # and then build a list of indexes for each class label that
    # provides the indexes for all examples with a given label
    numClasses = len(np.unique(labels))
    idx = [np.where(labels == i)[0] for i in range(0, numClasses)]
    # loop over all images
    for idxA in range(len(images)):
        # grab the current image and label belonging to the current
        # iteration
        currentImage = images[idxA]
        label = labels[idxA]
        # randomly pick an image that belongs to the *same* class
        # label
        idxB = np.random.choice(idx[label])
        posImage = images[idxB]
        # prepare a positive pair and update the images and labels
        # lists, respectively
        pairImages.append([currentImage, posImage])
        pairLabels.append([1])
        # grab the indices for each of the class labels *not* equal to
        # the current label and randomly pick an image corresponding
        # to a label *not* equal to the current label
        negIdx = np.where(labels != label)[0]
        negImage = images[np.random.choice(negIdx)]
        # prepare a negative pair of images and update our lists
        pairImages.append([currentImage, negImage])
        pairLabels.append([0])
    # return a 2-tuple of our image pairs and labels
    return (np.array(pairImages), np.array(pairLabels))


# In[5]:


print("getting positive and negative pairs...")
(pairTrain, labelTrain) = make_pairs(XTrain, YTrain)
(pairTest, labelTest) = make_pairs(XTest, YTest)
print('made')


# In[7]:


IMG_SHAPE = (224, 224, 3)
print("Building siamese network")
imgA = Input(shape=IMG_SHAPE)
imgB = Input(shape=IMG_SHAPE)
featureExtractor = build_siamese_model(IMG_SHAPE)
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)
print('done')


# In[8]:


def euclidean_distance(vectors):
    # unpack the vectors into separate lists
    (featsA, featsB) = vectors
    # compute the sum of squared distances between the vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)
    # return the euclidean distance between the vectors
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))


# In[9]:


distance = Lambda(euclidean_distance)([featsA, featsB])
outputs = Dense(1, activation="sigmoid")(distance)
model = Model(inputs=[imgA, imgB], outputs=outputs)


# In[11]:


print("Compiling model")
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

print("Training model")
history = model.fit(
    [pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:],
    validation_data=([pairTest[:, 0], pairTest[:, 1]], labelTest[:]),
    batch_size=64, 
    epochs=100, verbose=1)


# In[ ]:


# Figure out why it isnt training

