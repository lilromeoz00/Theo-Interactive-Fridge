#!/usr/bin/env python
# coding: utf-8

# In[3]:


XTrain = np.load('XTrain.npy')
YTrain = np.load('YTrain.npy')
XTest = np.load('XTest.npy')
YTest = np.load('YTest.npy')
from ipynb.fs.full.BuildPairs import make_pairs
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np


# In[4]:


def euclidean_distance(vectors):
    # unpack the vectors into separate lists
    (featsA, featsB) = vectors
    # compute the sum of squared distances between the vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)
    # return the euclidean distance between the vectors
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))


# In[5]:


def plot_training(H, plotPath):
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(H.history["loss"], label="train_loss")
    plt.plot(H.history["val_loss"], label="val_loss")
    plt.plot(H.history["accuracy"], label="train_acc")
    plt.plot(H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower_left")
    plt.savefig(plotPath)


# In[ ]:




