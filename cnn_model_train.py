import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Function to get image size
def get_image_size():
    sample_path = 'gestures/1'
    sample_images = os.listdir(sample_path)
    if sample_images:
        img_path = os.path.join(sample_path, sample_images[0])
        img = cv2.imread(img_path)
        if img is not None:
            return img.shape[1], img.shape[0]
    raise ValueError("No valid images found in 'gestures/1'.")

# Create synthetic images for class 0 if not present
def generate_synthetic_images(class_path, count=100):
    os.makedirs(class_path, exist_ok=True)
    for i in range(count):
        img = np.random.randint(0, 256, (image_y, image_x, 3), dtype=np.uint8)
        cv2.imwrite(f"{class_path}/{i}.jpg", img)

# Load dataset
def load_data():
    data = []
    labels = []

    # Load gesture images (class 1)
    for img_name in os.listdir('gestures/1'):
        img_path = os.path.join('gestures/1', img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (image_x, image_y))
            data.append(img)
            labels.append(1)

    # Load non-gesture images (class 0)
    for img_name in os.listdir('gestures/0'):
        img_path = os.path.join('gestures/0', img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (image_x, image_y))
            data.append(img)
            labels.append(0)

    data = np.array(data, dtype="float32") / 255.0  # Normalize pixel values
    labels = np.array(labels)

    return data, labels

# Build the model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(image_y, image_x, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')  # Binary classification: 0 and 1
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Main code
if __name__ == "__main__":
    print("Loading dataset...")

    # Get image dimensions
    try:
        image_x, image_y = get_image_size()
    except ValueError as e:
        print(e)
        exit()

    # Generate synthetic images for 'gestures/0' if the folder is missing or empty
    if not os.path.exists('gestures/0') or len(os.listdir('gestures/0')) == 0:
        print("Generating synthetic images for class 0...")
        generate_synthetic_images('gestures/0')

    # Load dataset
    data, labels = load_data()
    print(f"Loaded {len(data)} images and {len(labels)} labels.")
    print(f"Unique labels found: {np.unique(labels)}")

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Build and train the model
    model = build_model()
    print("Training the model...")
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

    # Save the trained model
    model.save("cnn_model_keras2.h5")
    print("Model saved as 'cnn_model_keras2.h5'.")
