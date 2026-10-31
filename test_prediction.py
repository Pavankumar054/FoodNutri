import tensorflow as tf
import numpy as np
import cv2
import os

# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "model/food_model.keras"
CLASS_NAMES_PATH = "model/class_names.txt"

IMAGE_PATH = "dataset/test/bread"


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully!")


# ==========================================
# LOAD CLASS NAMES
# ==========================================

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as f:

    class_names = [
        line.strip()
        for line in f
        if line.strip()
    ]


print("\nClasses:")
print(class_names)


# ==========================================
# FIND A BREAD IMAGE
# ==========================================

files = os.listdir(IMAGE_PATH)

image_file = None

for file in files:

    if file.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):

        image_file = file
        break


if image_file is None:

    print("No image found in bread folder.")
    exit()


image_path = os.path.join(
    IMAGE_PATH,
    image_file
)

print("\nTesting image:")
print(image_path)


# ==========================================
# READ IMAGE USING OPENCV
# ==========================================

image = cv2.imread(
    image_path
)

if image is None:

    print("Could not read image.")
    exit()


# ==========================================
# RESIZE
# ==========================================

image = cv2.resize(
    image,
    (224, 224)
)


# ==========================================
# BGR → RGB
# ==========================================

image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


# ==========================================
# PREPARE IMAGE
# ==========================================

image = np.array(
    image,
    dtype=np.float32
)

image = np.expand_dims(
    image,
    axis=0
)


# ==========================================
# PREDICTION
# ==========================================

prediction = model.predict(
    image,
    verbose=0
)


# ==========================================
# GET RESULT
# ==========================================

predicted_index = np.argmax(
    prediction[0]
)

predicted_food = class_names[
    predicted_index
]

confidence = (
    prediction[0][predicted_index]
    * 100
)


# ==========================================
# DISPLAY
# ==========================================

print("\n==============================")
print("EXPECTED FOOD : bread")
print("PREDICTED FOOD:", predicted_food)
print(
    "CONFIDENCE    :",
    round(float(confidence), 2),
    "%"
)
print("==============================")