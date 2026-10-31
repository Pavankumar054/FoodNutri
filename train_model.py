import tensorflow as tf
from tensorflow.keras import layers, models
import os

# =====================================================
# 1. SETTINGS
# =====================================================

TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

MODEL_DIR = "model"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "food_model.keras"
)

CLASS_NAMES_PATH = os.path.join(
    MODEL_DIR,
    "class_names.txt"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# =====================================================
# 2. LOAD DATASET
# =====================================================

print("\nLoading training dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=123
)

print("\nLoading validation dataset...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=123
)

print("\nLoading test dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)


# =====================================================
# 3. CLASS NAMES
# =====================================================

class_names = train_dataset.class_names

print("\nFood Classes:")
print(class_names)

print("\nNumber of classes:", len(class_names))


# =====================================================
# 4. SAVE CLASS NAMES
# =====================================================

with open(
    CLASS_NAMES_PATH,
    "w",
    encoding="utf-8"
) as f:

    for name in class_names:
        f.write(name + "\n")

print("\nClass names saved to:")
print(CLASS_NAMES_PATH)


# =====================================================
# 5. PERFORMANCE
# =====================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    AUTOTUNE
)


# =====================================================
# 6. DATA AUGMENTATION
# =====================================================

data_augmentation = tf.keras.Sequential([

    layers.RandomFlip(
        "horizontal"
    ),

    layers.RandomRotation(
        0.1
    ),

    layers.RandomZoom(
        0.1
    )

])


# =====================================================
# 7. LOAD MOBILENETV2
# =====================================================

print("\nLoading MobileNetV2...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(
        IMG_SIZE,
        IMG_SIZE,
        3
    ),

    include_top=False,

    weights="imagenet"
)


# =====================================================
# 8. FREEZE BASE MODEL
# =====================================================

base_model.trainable = False


# =====================================================
# 9. BUILD MODEL
# =====================================================

model = models.Sequential([

    layers.Input(
        shape=(
            IMG_SIZE,
            IMG_SIZE,
            3
        )
    ),

    data_augmentation,

    # MobileNetV2 preprocessing
    layers.Rescaling(
        1.0/127.5,
        offset=-1
    ),

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.3),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.3),

    layers.Dense(
        len(class_names),
        activation="softmax"
    )

])


# =====================================================
# 10. COMPILE
# =====================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)


# =====================================================
# 11. MODEL SUMMARY
# =====================================================

print("\nModel Summary:\n")

model.summary()


# =====================================================
# 12. TRAIN
# =====================================================

print("\n======================================")
print("STARTING MOBILENETV2 TRAINING")
print("======================================\n")

history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS

)


# =====================================================
# 13. TEST
# =====================================================

print("\n======================================")
print("TESTING MODEL")
print("======================================\n")

test_loss, test_accuracy = model.evaluate(
    test_dataset
)


print(
    "\nTest Accuracy:",
    round(
        test_accuracy * 100,
        2
    ),
    "%"
)


# =====================================================
# 14. SAVE MODEL
# =====================================================

model.save(
    MODEL_PATH
)

print("\n======================================")
print("MODEL SAVED SUCCESSFULLY")
print("======================================")

print(
    "Model:",
    MODEL_PATH
)

print(
    "Classes:",
    CLASS_NAMES_PATH
)