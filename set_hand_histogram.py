import cv2
import numpy as np
import pickle


def build_squares(img):
    x, y, w, h = 420, 140, 10, 10
    d = 10
    imgCrop = None
    crop = None
    for i in range(10):
        for j in range(5):
            if np.any(imgCrop == None):
                imgCrop = img[y:y + h, x:x + w]
            else:
                imgCrop = np.hstack((imgCrop, img[y:y + h, x:x + w]))
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            x += w + d
        if np.any(crop == None):
            crop = imgCrop
        else:
            crop = np.vstack((crop, imgCrop))
        imgCrop = None
        x = 420
        y += h + d
    return crop


def get_hand_hist_from_file(hist_file_path):
    with open(hist_file_path, "rb") as f:
        hist = pickle.load(f)
    return hist


def process_video_with_hist():
    cam = cv2.VideoCapture(0)  # Use webcam for real-time processing
    x, y, w, h = 300, 100, 300, 300
    hist = get_hand_hist_from_file(r"D:\Sign Language To Text Conversion\Code\hist")  # Load existing histogram
    flagPressedC, flagPressedS = False, False
    imgCrop = None

    while True:
        img = cam.read()[1]
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        keypress = cv2.waitKey(1)
        if keypress == ord('s'):
            flagPressedS = True
            break

        if flagPressedC:
            dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
            dst1 = dst.copy()
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            cv2.filter2D(dst, -1, disc, dst)
            blur = cv2.GaussianBlur(dst, (11, 11), 0)
            blur = cv2.medianBlur(blur, 15)
            ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh = cv2.merge((thresh, thresh, thresh))
            cv2.imshow("Thresh", thresh)

        if not flagPressedS:
            imgCrop = build_squares(img)

        cv2.imshow("Hand Tracking", img)

    cam.release()
    cv2.destroyAllWindows()


process_video_with_hist()
