import os
import tensorflow as tf

from data_loader import load_cifar10
from preprocessing import preprocess_data
from augmentation import create_augmentation

from models.baseline_cnn import build_baseline_cnn
from models.regularized_cnn import build_regularized_cnn
from models.deep_cnn import build_deep_cnn


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 32
EPOCHS = 10
RANDOM_SEED = 42

MODEL_DIR = "../models"
RESULTS_DIR = "../results"


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

tf.random.set_seed(RANDOM_SEED)


# ============================================================
# LOAD CIFAR-10
# ============================================================

print("\n" + "=" * 60)
print("LOADING CIFAR-10 DATASET")
print("=" * 60)

x_train, y_train, x_test, y_test = load_cifar10()


# ============================================================
# PREPROCESS DATA
# ============================================================

print("\n" + "=" * 60)
print("PREPROCESSING DATA")
print("=" * 60)

(
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test
) = preprocess_data(
    x_train,
    y_train,
    x_test,
    y_test
)

print("\nDataset shapes:")
print("Training   :", x_train.shape)
print("Validation :", x_val.shape)
print("Testing    :", x_test.shape)


# ============================================================
# DATA AUGMENTATION
# ============================================================

augmentation = create_augmentation()


# ============================================================
# CREATE TF.DATA DATASETS
# ============================================================

print("\nCreating TensorFlow datasets...")

train_dataset = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)
)

train_dataset = (
    train_dataset
    .shuffle(
        buffer_size=10000,
        seed=RANDOM_SEED
    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


val_dataset = tf.data.Dataset.from_tensor_slices(
    (x_val, y_val)
)

val_dataset = (
    val_dataset
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


# ============================================================
# MODEL COMPILATION
# ============================================================

def compile_model(model):

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),

        loss="sparse_categorical_crossentropy",

        metrics=[
            "accuracy"
        ]
    )

    return model


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_model(model, model_name):

    print("\n" + "=" * 60)
    print(f"TRAINING {model_name}")
    print("=" * 60)

    # Compile model
    model = compile_model(model)

    # Display model information
    model.summary()

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    checkpoint_path = os.path.join(
        MODEL_DIR,
        f"{model_name}.keras"
    )

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history = model.fit(

        # Augmentation happens batch-by-batch
        train_dataset.map(
            lambda images, labels: (
                augmentation(
                    images,
                    training=True
                ),
                labels
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        ),

        validation_data=val_dataset,

        epochs=EPOCHS,

        callbacks=callbacks,

        verbose=1
    )

    print("\nFinished:", model_name)

    print(
        "Best validation accuracy:",
        max(history.history["val_accuracy"])
    )

    print(
        "Model saved to:",
        checkpoint_path
    )

    return model, history


# ============================================================
# CNN 1 — BASELINE
# ============================================================

print("\n")
print("#" * 60)
print("# CNN 1 — BASELINE CNN")
print("#" * 60)

model1, history1 = train_model(
    build_baseline_cnn(),
    "cnn_baseline"
)


# ============================================================
# CNN 2 — REGULARIZED
# ============================================================

print("\n")
print("#" * 60)
print("# CNN 2 — REGULARIZED CNN")
print("#" * 60)

model2, history2 = train_model(
    build_regularized_cnn(),
    "cnn_regularized"
)


# ============================================================
# CNN 3 — DEEP
# ============================================================

print("\n")
print("#" * 60)
print("# CNN 3 — DEEP CNN")
print("#" * 60)

model3, history3 = train_model(
    build_deep_cnn(),
    "cnn_deep"
)


# ============================================================
# TRAINING COMPLETE
# ============================================================

print("\n")
print("=" * 60)
print("ALL TRAINING COMPLETED")
print("=" * 60)

print("\nSaved models:")

print("1.", os.path.join(MODEL_DIR, "cnn_baseline.keras"))
print("2.", os.path.join(MODEL_DIR, "cnn_regularized.keras"))
print("3.", os.path.join(MODEL_DIR, "cnn_deep.keras"))

print("\nNext step:")
print("Evaluate all three CNNs and build the ensemble.")