import cv2
import numpy as np
import pickle

def get_hand_hist_from_file(hist_file_path):
    try:
        with open(hist_file_path, "rb") as f:
            hist = pickle.load(f)
            if isinstance(hist, np.ndarray):
                print("Histogram loaded successfully.")
            else:
                print("The loaded data is not of the expected type.")
            return hist
    except (pickle.UnpicklingError, FileNotFoundError) as e:
        print(f"Error loading the histogram: {e}")
        return None

def process_video_with_hist():
    cam = cv2.VideoCapture(0)  # Use webcam for real-time processing
    x, y, w, h = 300, 100, 300, 300
    hist = get_hand_hist_from_file(r"D:\Deep\Code\hist")  # Load existing histogram
    
    if hist is None:
        print("Histogram is not available, exiting.")
        return

    flagPressedC, flagPressedS = False, False
    imgCrop = None

    while True:
        img = cam.read()[1]
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        keypress = cv2.waitKey(1)
        if keypress == ord('s'):  # Press 's' to stop the video capture
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

        cv2.imshow("Hand Tracking", img)

    cam.release()
    cv2.destroyAllWindows()

# Start the hand tracking process
process_video_with_hist()
