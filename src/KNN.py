from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow import keras
from sklearn.model_selection import train_test_split

import gc


def train_and_evalu(var_learningrate,var_dropout,var_epoch,var_batch_size):
    #%%
    ### Daten
    print("var_learningrate", var_learningrate, "var_dropout", var_dropout, "var_epoch", var_epoch, "var_batch_size", var_batch_size)
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
      keras.layers.Dropout(var_dropout),
      keras.layers.Dense(10, activation='softmax')
    ])


    #%%si
    ### Optimizer
    adam = keras.optimizers.Adam(lr=var_learningrate)

    model.compile(optimizer=adam,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    #%%
    ### Tensorboard

    #tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    #%%
    ### Model fit

    model.fit(small_train_images, small_train_labels, epochs=int(var_epoch),batch_size=int(var_batch_size))

    #%%
    ### Model evalu

    test_loss, test_acc = model.evaluate(small_test_images, small_test_labels)
    print("test_loss: ",test_loss , "test_acc: ", test_acc)
    gc.collect()
    return test_loss, test_acc