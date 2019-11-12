from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np

def model_fit_evalu():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    test_loss,test_acc =  model.evaluate(x_test,  y_test, verbose=2)
    variables = 0
    variables = np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])
    print("test_loss: ",test_loss , "test_acc: ", test_acc, "variables",variables)


if __name__ == "__main__":
    model_fit_evalu()
    model_fit_evalu()
    model_fit_evalu()