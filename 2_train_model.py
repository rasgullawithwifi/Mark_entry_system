"""
Script 2: 2_train_model.py
───────────────────────────
Trains a CNN cell classifier on your prepared dataset
Saves the trained model to models/cell_classifier.h5

Run: python3 2_train_model.py
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import json

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR  = "cell_data/dataset"
MODEL_DIR    = "models"
MODEL_PATH   = os.path.join(MODEL_DIR, "cell_classifier.h5")
CELL_SIZE    = 64
BATCH_SIZE   = 32
EPOCHS       = 30
CLASS_NAMES  = ["blank", "0", "0.5", "1", "1.5", "2", "2.5", "3"]

os.makedirs(MODEL_DIR, exist_ok=True)

print("\n" + "="*50)
print("  STEP 2: TRAIN CELL CLASSIFIER")
print("="*50)

# ── Count images per class ────────────────────────────────────────────────────
print(f"\nDataset summary:")
total = 0
for cls in CLASS_NAMES:
    cls_dir = os.path.join(DATASET_DIR, cls)
    count = len(os.listdir(cls_dir)) if os.path.exists(cls_dir) else 0
    total += count
    print(f"  {cls:>6}: {count} images")
print(f"  {'Total':>6}: {total} images")

if total == 0:
    print("\n❌ No images found! Run 1_prepare_dataset.py first.")
    exit()

# ── Data generators ───────────────────────────────────────────────────────────
print(f"\nLoading dataset...")

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(CELL_SIZE, CELL_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='training',
    shuffle=True,
    classes=CLASS_NAMES
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(CELL_SIZE, CELL_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='validation',
    shuffle=False,
    classes=CLASS_NAMES
)

print(f"\n  Training samples   : {train_gen.samples}")
print(f"  Validation samples : {val_gen.samples}")
print(f"  Classes            : {train_gen.class_indices}")

# ── Save class mapping ────────────────────────────────────────────────────────
class_map_path = os.path.join(MODEL_DIR, "class_mapping.json")
with open(class_map_path, 'w') as f:
    json.dump(train_gen.class_indices, f, indent=2)
print(f"\n  Class mapping saved to: {class_map_path}")

# ── Build model ───────────────────────────────────────────────────────────────
print(f"\nBuilding model...")

model = models.Sequential([
    # Block 1
    layers.Conv2D(32, (3,3), activation='relu',
                  input_shape=(CELL_SIZE, CELL_SIZE, 1), padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    # Block 2
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    # Block 3
    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    # Classifier head
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(len(CLASS_NAMES), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ── Callbacks ─────────────────────────────────────────────────────────────────
callbacks = [
    # Save best model automatically
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    # Stop early if no improvement for 8 epochs
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    # Reduce learning rate if stuck
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        verbose=1
    )
]

# ── Train ─────────────────────────────────────────────────────────────────────
print(f"\n🚀 Training for up to {EPOCHS} epochs...")
print(f"   (Early stopping will stop if no improvement)")
print(f"   Best model auto-saved to: {MODEL_PATH}\n")

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=1
)

# ── Results ───────────────────────────────────────────────────────────────────
best_val_acc = max(history.history['val_accuracy'])
best_val_loss = min(history.history['val_loss'])

print(f"\n{'='*50}")
print(f"  TRAINING COMPLETE")
print(f"{'='*50}")
print(f"  Best validation accuracy : {best_val_acc*100:.2f}%")
print(f"  Best validation loss     : {best_val_loss:.4f}")
print(f"  Model saved to           : {MODEL_PATH}")

if best_val_acc >= 0.85:
    print(f"\n  ✅ Good accuracy! Ready for testing.")
elif best_val_acc >= 0.70:
    print(f"\n  ⚠️  Decent accuracy. Collect more sheets to improve.")
else:
    print(f"\n  ❌ Low accuracy. Need more training data.")

print(f"\n  ✅ Run next: python3 3_test_model.py")