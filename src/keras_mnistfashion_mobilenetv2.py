from __future__ import absolute_import, division, print_function, unicode_literals



import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow import keras
from time import time
from sklearn.model_selection import train_test_split

import numpy as np
import PIL

### Variablen
"""
learningrate
dropout
optimizer
loss
epoch
batch_size
"""

#%%
### Daten

mnist = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0


#extra feature um Datenset künstlich zu verkleinern
small_train_images, small_test_images, small_train_labels, small_test_labels = train_test_split(
    train_images, train_labels, test_size=0.9, shuffle=False)

#%%
### Model

model = keras.models.Sequential([
  keras.layers.Flatten(input_shape=(28, 28)),
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dropout(0.2),
  keras.layers.Dense(10, activation='softmax')
])


#%%si
### Optimizer
adam = keras.optimizers.Adam(lr=0.001)

model.compile(optimizer=adam,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#%%
### Tensorboard

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

#%%
### Model fit

model.fit(small_train_images, small_train_labels, epochs=10,batch_size=64, callbacks=[tensorboard])

#%%
### Model evalu

test_loss, test_acc = model.evaluate(small_test_images, small_test_labels)
print("test_loss: ",test_loss , "test_acc: ", test_acc)

#%%
### Model predict

predictions = model.predict(small_test_images)
print("Predictions: ", predictions[0])


class_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
class_array_real = ["T-Shirt","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

### test 10 Pic and show them
for i in range(0,10):
    print("----------------")
    print("class prediction: ", class_array[predictions[i].argmax()], " = ", class_array_real[class_array[predictions[i].argmax()]])
    print("ground truth: ", small_test_labels[i], " = ", class_array_real[small_test_labels[i]])
    two_d = (np.reshape(small_test_images[i], (28, 28)) * 255).astype(np.uint8)
    img = PIL.Image.fromarray(two_d, 'L')
    img.show()
