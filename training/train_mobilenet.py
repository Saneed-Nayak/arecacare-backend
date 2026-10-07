import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "balanced_dataset"

MODEL_DIR = BASE_DIR.parent / "model"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "arecanut_mobilenetv2.h5"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 8
INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 15
SEED = 42


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / "train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / "valid",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / "test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


class_names = train_ds.class_names
num_classes = len(class_names)

print("\nClasses:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")

print(f"\nNumber of classes: {num_classes}")


# Save class names
with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=2)


# ============================================================
# PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
valid_ds = valid_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.10),
], name="data_augmentation")


# ============================================================
# MOBILENETV2 BASE MODEL
# ============================================================

print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
)

# First stage: freeze pretrained model
base_model.trainable = False


# ============================================================
# BUILD MODEL
# ============================================================

inputs = layers.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

# MobileNetV2 expects pixels in approximately [-1, 1]
x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.35)(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.25)(x)

outputs = layers.Dense(
    num_classes,
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    ),

	ModelCheckpoint(
    MODEL_DIR / "mobilenetv2_best.weights.h5",
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=True,
    verbose=1,
),
]


# ============================================================
# STAGE 1 — TRAIN CLASSIFICATION HEAD
# ============================================================

print("\n========================================")
print("STAGE 1 — TRAINING CLASSIFICATION HEAD")
print("========================================\n")

history1 = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=INITIAL_EPOCHS,
    callbacks=callbacks,
)


# ============================================================
# STAGE 2 — FINE TUNING
# ============================================================

print("\n========================================")
print("STAGE 2 — FINE TUNING MOBILENETV2")
print("========================================\n")

base_model.trainable = True

# Freeze earlier layers.
# Fine-tune only the later MobileNetV2 layers.
for layer in base_model.layers[:-30]:
    layer.trainable = False


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


history2 = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=callbacks,
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best saved weights...")

BEST_WEIGHTS = MODEL_DIR / "mobilenetv2_best.weights.h5"

model.load_weights(BEST_WEIGHTS)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


# ============================================================
# TEST
# ============================================================

print("\n========================================")
print("FINAL TEST")
print("========================================\n")

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1,
)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# ============================================================
# SAVE FINAL MODEL
# ============================================================



print("\n========================================")
print("TRAINING COMPLETE")
print("========================================")

print(f"\nModel saved to:")
print(MODEL_PATH)

print("\nClass names saved to:")
print(CLASS_NAMES_PATH)

print(f"\nFinal test accuracy: {test_accuracy * 100:.2f}%")