import cv2
import os

def flip_images():
    gest_folder = "gestures"  # Path to the gestures folder
    # Iterate through each gesture folder (for example, 0, 1, 2, etc.)
    for g_id in os.listdir(gest_folder):
        gesture_folder_path = os.path.join(gest_folder, g_id)

        # Check if the folder exists
        if not os.path.isdir(gesture_folder_path):
            continue
        
        # Loop over each image in the gesture folder
        for i in range(1200):  # Assuming there are 1200 images for each gesture
            img_path = os.path.join(gesture_folder_path, f"{i+1}.jpg")  # Original image path
            new_path = os.path.join(gesture_folder_path, f"{i+1+1200}.jpg")  # New flipped image path

            # Check if the image exists
            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                continue

            # Read the image
            img = cv2.imread(img_path, 0)  # Read as grayscale

            # Check if image is loaded correctly
            if img is None:
                print(f"Error reading image {img_path}")
                continue

            # Flip the image horizontally
            img = cv2.flip(img, 1)

            # Save the flipped image with a new filename
            cv2.imwrite(new_path, img)
            print(f"Flipped image saved to: {new_path}")

# Run the function
flip_images()
