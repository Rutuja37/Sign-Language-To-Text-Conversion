import cv2
import numpy as np
import pickle
import os
import sqlite3

image_x, image_y = 50, 50

# Function to load the pre-saved hand histogram
def get_hand_hist():
    try:
        with open("hist", "rb") as f:
            hist = pickle.load(f)
        return hist
    except FileNotFoundError:
        print("Error: Histogram file 'hist' not found!")
        exit(1)

# Function to initialize the database and create folder structure
def init_create_folder_database():
    db_path = "D:\\Sign Language To Text Conversion\\Code\\gesture_db.db"
    if not os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        create_table_cmd = "CREATE TABLE gesture ( g_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, g_name TEXT NOT NULL )"
        conn.execute(create_table_cmd)
        conn.commit()
        conn.close()

# Function to create a folder
def create_folder(folder_name):
    os.makedirs(folder_name, exist_ok=True)  # Ensure all directories exist

# Function to store gesture ID and name in the database
def store_in_db(g_id, g_name):
    db_path = "D:\\Sign Language To Text Conversion\\Code\\gesture_db.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("INSERT INTO gesture (g_id, g_name) VALUES (?, ?)", (g_id, g_name))
    except sqlite3.IntegrityError:
        choice = input("g_id already exists. Want to change the record? (y/n): ")
        if choice.lower() == 'y':
            cursor.execute("UPDATE gesture SET g_name = ? WHERE g_id = ?", (g_name, g_id))
        else:
            print("Doing nothing...")
            return
    conn.commit()
    conn.close()

# Function to capture and save hand gesture images
def store_images(g_id):
    total_pics = 1200
    hist = get_hand_hist()

    cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: No camera detected. Please check your camera connection.")
        return

    x, y, w, h = 300, 100, 300, 300
    folder_path = f"D:\\Sign Language To Text Conversion\\Code\\gestures\\{g_id}"
    create_folder(folder_path)

    pic_no = 0
    flag_start_capturing = True  # Automatically start capturing
    frames = 0

    while True:
        ret, img = cam.read()
        if not ret:
            print("Error: Unable to capture image from camera.")
            break

        img = cv2.flip(img, 1)
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)

        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        cv2.filter2D(dst, -1, disc, dst)
        blur = cv2.GaussianBlur(dst, (11, 11), 0)
        blur = cv2.medianBlur(blur, 15)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        thresh = thresh[y:y + h, x:x + w]
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        if contours:
            print("Contours detected.")
            contour = max(contours, key=cv2.contourArea)
            print("Contour area:", cv2.contourArea(contour))
            if cv2.contourArea(contour) > 3000 and flag_start_capturing:
                x1, y1, w1, h1 = cv2.boundingRect(contour)
                save_img = thresh[y1:y1 + h1, x1:x1 + w1]

                if save_img.shape[0] > 0 and save_img.shape[1] > 0:
                    save_img = cv2.resize(save_img, (image_x, image_y))
                    img_path = os.path.join(folder_path, f"{pic_no}.jpg")
                    cv2.imwrite(img_path, save_img)
                    print(f"Image {pic_no} saved: {img_path}")
                    pic_no += 1
                else:
                    print(f"Skipping image {pic_no}, invalid dimensions: {save_img.shape}")

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, str(pic_no), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
        cv2.imshow("Capturing gesture", img)
        cv2.imshow("Threshold", thresh)

        keypress = cv2.waitKey(1)
        if keypress == 27:  # Press 'ESC' to exit
            break
        if pic_no == total_pics:
            break

    cam.release()
    cv2.destroyAllWindows()

# Main script starts here
init_create_folder_database()

# Input gesture ID and name
g_id = input("Enter gesture ID (choose existing gesture ID): ")
g_name = input("Enter gesture name/text: ")
store_in_db(g_id, g_name)
store_images(g_id)
