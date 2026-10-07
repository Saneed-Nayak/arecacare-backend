import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models

BASE = Path(
    r"C:\Users\sanee\Videos\websites codes\New folder\arecacare-backend\training"
)

DATASET = BASE / "balanced_dataset"
MODEL_OUTPUT = BASE / "arecanut_9class.h5"
CLASS_OUTPUT = BASE / "class_names.json"

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 8
EPOCHS = 30

print("TensorFlow version:", tf.__version__)

# Load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET / "train",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET / "valid",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET / "test",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("\nClasses:")
for i, name in enumerate(class_names):
    print(i, "->", name)

print("\nNumber of classes:", num_classes)

# Save class names
with open(CLASS_OUTPUT, "w") as f:
    json.dump(class_names, f, indent=4)

# Improve input pipeline
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(200).prefetch(AUTOTUNE)
valid_ds = valid_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.1),
])

# CNN model
model = models.Sequential([
    layers.Input(shape=(256, 256, 3)),

    data_augmentation,

    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),

    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Training
print("\nStarting training...\n")

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS
)

# Evaluate
print("\nEvaluating test dataset...\n")

test_loss, test_accuracy = model.evaluate(test_ds)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save model
model.save(MODEL_OUTPUT)

print("\nModel saved:")
print(MODEL_OUTPUT)

print("\nClass names saved:")
print(CLASS_OUTPUT)

print("\nTraining completed successfully!")