#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir('D:\ImageData')
from scipy.io import loadmat
import h5py
import numpy as np

data = h5py.File('imagedata.mat')
x = np.transpose(np.array(data["x"]))
y = np.transpose(np.array(data["y"]))

x = x.reshape(x.shape[0], 256, 256, 1)
y = y.reshape(x.shape[0], 512, 512, 1)
print(x.shape)
print(y.shape)

xcv = x[:1000, :, :, :]
ycv = y[:1000, :, :, :]

xtrain = x[1000:, :, :, :]
ytrain = y[1000:, :, :, :]

print(xtrain.shape)
print(ytrain.shape)
print(xcv.shape)
print(ycv.shape)


# In[ ]:


import tensorflow as tf

a = np.array([0.0001,0.0005,0.001,0.005])

mirrored_strategy = tf.distribute.MirroredStrategy()

for i in range (len(a)):
    with mirrored_strategy.scope():

        model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(8, (3, 3), activation='elu', padding = 'same', input_shape=(256, 256, 1), kernel_regularizer=tf.keras.regularizers.l2(a[i])),
                tf.keras.layers.Conv2D(8, (3, 3), activation='elu', padding = 'same', kernel_regularizer=tf.keras.regularizers.l2(a[i])),
                tf.keras.layers.MaxPool2D((2, 2)),

                tf.keras.layers.Conv2D(16, (3, 3), activation='elu', padding = 'same', kernel_regularizer=tf.keras.regularizers.l2(a[i])),
                tf.keras.layers.Conv2D(16, (3, 3), activation='elu', padding = 'same', kernel_regularizer=tf.keras.regularizers.l2(a[i])),
                tf.keras.layers.MaxPool2D((2, 2)),

                tf.keras.layers.Conv2D(32, (3, 3), activation='elu', padding = 'same', kernel_regularizer=tf.keras.regularizers.l2(a[i])),
                tf.keras.layers.Conv2D(32, (3, 3), activation='elu', padding = 'same', kernel_regularizer=tf.keras.regularizers.l2(a[i])),
                tf.keras.layers.MaxPool2D((2, 2)),

                tf.keras.layers.Conv2DTranspose(32, (3, 3), strides = (2,2), activation='elu', padding = 'same', kernel_regularizer=tf.keras.regularizers.l2(a[i])),
                tf.keras.layers.Conv2D(16, (3, 3), activation='elu', padding = 'same', kernel_regularizer=tf.keras.regularizers.l2(a[i])),
                tf.keras.layers.Conv2D(16, (3, 3), activation='elu', padding = 'same', kernel_regularizer=tf.keras.regularizers.l2(a[i])),

                tf.keras.layers.Conv2DTranspose(16, (3, 3), strides = (2,2), activation='elu', padding = 'same', kernel_regularizer=tf.keras.regularizers.l2(a[i])),
                tf.keras.layers.Conv2D(8, (3, 3), activation='elu', padding = 'same', kernel_regularizer=tf.keras.regularizers.l2(a[i])),
                tf.keras.layers.Conv2D(8, (3, 3), activation='elu', padding = 'same', kernel_regularizer=tf.keras.regularizers.l2(a[i])),

                tf.keras.layers.Conv2DTranspose(8, (3, 3), strides = (2,2), activation='elu', padding = 'same', kernel_regularizer=tf.keras.regularizers.l2(a[i])),
                tf.keras.layers.Conv2D(4, (3, 3), activation='elu', padding = 'same', kernel_regularizer=tf.keras.regularizers.l2(a[i])),
                tf.keras.layers.Conv2D(4, (3, 3), activation='elu', padding = 'same', kernel_regularizer=tf.keras.regularizers.l2(a[i])),

                tf.keras.layers.Conv2DTranspose(1, (3, 3), strides = (2,2), activation='elu', padding = 'same', kernel_regularizer=tf.keras.regularizers.l2(a[i])),
            ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                          loss='MSE',
                          metrics=['MeanSquaredError'])

    model.fit(xtrain, ytrain, batch_size=256, epochs=250)
    [_,loss[i]] = model.evaluate(xcv, ycv)
    
maxinx = np.unravel_index(loss.argmax(), loss.shape)
a = a[maxinx]

with mirrored_strategy.scope():

    model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(8, (3, 3), activation='elu', padding = 'same', input_shape=(256, 256, 1), kernel_regularizer=tf.keras.regularizers.l2(a)),
            tf.keras.layers.Conv2D(8, (3, 3), activation='elu', padding = 'same', kernel_regularizer=tf.keras.regularizers.l2(a)),
            tf.keras.layers.MaxPool2D((2, 2)),

            tf.keras.layers.Conv2D(16, (3, 3), activation='elu', padding = 'same', kernel_regularizer=tf.keras.regularizers.l2(a)),
            tf.keras.layers.Conv2D(16, (3, 3), activation='elu', padding = 'same', kernel_regularizer=tf.keras.regularizers.l2(a)),
            tf.keras.layers.MaxPool2D((2, 2)),

            tf.keras.layers.Conv2D(32, (3, 3), activation='elu', padding = 'same', kernel_regularizer=tf.keras.regularizers.l2(a)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='elu', padding = 'same', kernel_regularizer=tf.keras.regularizers.l2(a)),
            tf.keras.layers.MaxPool2D((2, 2)),

            tf.keras.layers.Conv2DTranspose(32, (3, 3), strides = (2,2), activation='elu', padding = 'same', kernel_regularizer=tf.keras.regularizers.l2(a)),
            tf.keras.layers.Conv2D(16, (3, 3), activation='elu', padding = 'same', kernel_regularizer=tf.keras.regularizers.l2(a)),
            tf.keras.layers.Conv2D(16, (3, 3), activation='elu', padding = 'same', kernel_regularizer=tf.keras.regularizers.l2(a)),

            tf.keras.layers.Conv2DTranspose(16, (3, 3), strides = (2,2), activation='elu', padding = 'same', kernel_regularizer=tf.keras.regularizers.l2(a)),
            tf.keras.layers.Conv2D(8, (3, 3), activation='elu', padding = 'same', kernel_regularizer=tf.keras.regularizers.l2(a)),
            tf.keras.layers.Conv2D(8, (3, 3), activation='elu', padding = 'same', kernel_regularizer=tf.keras.regularizers.l2(a)),

            tf.keras.layers.Conv2DTranspose(8, (3, 3), strides = (2,2), activation='elu', padding = 'same', kernel_regularizer=tf.keras.regularizers.l2(a)),
            tf.keras.layers.Conv2D(4, (3, 3), activation='elu', padding = 'same', kernel_regularizer=tf.keras.regularizers.l2(a)),
            tf.keras.layers.Conv2D(4, (3, 3), activation='elu', padding = 'same', kernel_regularizer=tf.keras.regularizers.l2(a)),

            tf.keras.layers.Conv2DTranspose(1, (3, 3), strides = (2,2), activation='elu', padding = 'same', kernel_regularizer=tf.keras.regularizers.l2(a)),
        ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                      loss='MSE',
                      metrics=['MeanSquaredError'])

model.fit(x, y, batch_size=256, epochs=250)


# In[3]:


testdata = loadmat('imagetestdata.mat')
xtest = np.array(testdata["xtest"])
ytest = np.array(testdata["ytest"])

xtest = xtest.reshape(1000, 256, 256, 1)
ytest = ytest.reshape(1000, 512, 512, 1)

ytesthat = np.array(model.predict(xtest))


# In[4]:


from scipy.io import savemat
savemat('Prediction.mat', {'ytesthat': ytesthat})


# In[ ]:




