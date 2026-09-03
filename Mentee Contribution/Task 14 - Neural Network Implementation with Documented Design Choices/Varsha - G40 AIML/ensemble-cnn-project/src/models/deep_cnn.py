import tensorflow as tf


def build_deep_cnn():

    model = tf.keras.Sequential([

        tf.keras.layers.Input(shape=(32, 32, 3)),

        # Block 1
        tf.keras.layers.Conv2D(
            32,
            3,
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.Conv2D(
            32,
            3,
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPooling2D(),

        # Block 2
        tf.keras.layers.Conv2D(
            64,
            3,
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.Conv2D(
            64,
            3,
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPooling2D(),

        # Block 3
        tf.keras.layers.Conv2D(
            128,
            3,
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.Conv2D(
            128,
            3,
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(
            10,
            activation="softmax"
        )
    ])

    return model