import tensorflow as tf


def build_regularized_cnn():

    model = tf.keras.Sequential([

        tf.keras.layers.Input(shape=(32, 32, 3)),

        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            padding="same"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Activation("relu"),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(
            64,
            (3, 3),
            padding="same"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Activation("relu"),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(
            10,
            activation="softmax"
        )
    ])

    return model