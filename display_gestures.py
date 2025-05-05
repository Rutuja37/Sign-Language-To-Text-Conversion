import cv2
import os

# Function to get the image size from a valid file
def get_image_size():
    folder_path = "D:\\Deep\\Code\\gestures\\1"  # Updated path
    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)
        if img is not None:  # If the image is successfully read
            return img.shape[:2]  # Return height and width only
    print("Error: No valid images found in the folder.")
    exit(1)

# Display gestures
def display_gestures():
    folder_path = "D:\\Deep\\Code\\gestures\\1"  # Updated path
    image_x, image_y = get_image_size()  # Get height and width only

    for file in sorted(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)
        if img is None:  # Skip unreadable or missing files
            print(f"Warning: Could not read file {img_path}. Skipping...")
            continue

        img = cv2.resize(img, (image_y, image_x))  # Resize using width (x) and height (y)
        cv2.imshow("Gesture", img)
        keypress = cv2.waitKey(0)  # Wait for a key press to show the next image
        if keypress == 27:  # Press ESC to exit
            break

    cv2.destroyAllWindows()


# Main script
if __name__ == "__main__":
    display_gestures()
