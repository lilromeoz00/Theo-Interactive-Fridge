#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
IMG_SHAPE = (224, 224, 3)
BATCH_SIZE = 64
EPOCHS = 100

BASE_OUTPUT = r'C:\Users\jordan\output'
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])


# In[ ]:




