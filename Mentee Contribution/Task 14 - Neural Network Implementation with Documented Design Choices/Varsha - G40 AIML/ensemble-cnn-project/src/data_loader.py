import tensorflow as tf


def load_cifar10():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    return x_train, y_train, x_test, y_test