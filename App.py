import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image
import os

# Initialize variables
current_word = ""
detected_words = []
last_detected_time = time.time()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

DEMO_VIDEO = 'demo.mp4'
DEMO_IMAGE = 'demo.jpg'

my_list = []

# Custom CSS with blurry background + animations
st.markdown("""
<style>
/* Blurred Futuristic Background with Custom Image */
.stApp {
    background: url('/mnt/data/A_digital_illustration_depicts_a_person\'s_left_han.png');
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
    filter: blur(40%);
    color: #ffffff;
    position: relative;
}

/* Glass effect overlay */
[data-testid="stSidebar"] {
    background: rgba(15, 15, 35, 0.7);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    color: white;
    border-right: 2px solid rgba(255, 255, 255, 0.1);
}

/* Sidebar width */
[data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
    width: 400px;
}
[data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
    width: 400px;
    margin-left: -400px;
}

/* Buttons */
.stButton button {
    background: linear-gradient(to right, #8e2de2, #4a00e0);
    color: white;
    font-weight: bold;
    border-radius: 12px;
    padding: 12px 24px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    animation: fadeInButton 1.5s ease-in-out;
}

/* Button hover */
.stButton button:hover {
    background: linear-gradient(to right, #fbc2eb, #a18cd1);
    transform: scale(1.05);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
}

/* Button click animation (pulse) */
.stButton button:active {
    animation: pulse 0.4s;
}

/* Fade in elements */
.fade-in {
    animation: fadeInUp 1.2s ease-in-out;
}

/* Animations */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes fadeInButton {
    from {
        opacity: 0;
        transform: scale(0.95);
    }
    to {
        opacity: 1;
        transform: scale(1);
    }
}

@keyframes pulse {
    0% {
        transform: scale(0.98);
        box-shadow: 0 0 0 0 rgba(142, 45, 226, 0.7);
    }
    70% {
        transform: scale(1);
        box-shadow: 0 0 0 10px rgba(142, 45, 226, 0);
    }
    100% {
        transform: scale(0.98);
        box-shadow: 0 0 0 0 rgba(142, 45, 226, 0);
    }
}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title('🧠 Sign Language Detection')
st.sidebar.subheader('➤ Parameter')

@st.cache_resource()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

# App Mode
app_mode = st.sidebar.selectbox('📂 Choose the App Mode', 
                                ['Select Mode', 'Sign Language to Text'])

if app_mode == 'Select Mode':
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.title('Sign Language Detection System')
    st.markdown("""
        In this application, we use **MediaPipe** to detect and interpret hand gestures.
        Our system transforms those gestures into text to assist real-time communication.
    """)
    st.markdown("""
        ### 💡 About Us  
        Our AI-based sign detection system ensures seamless communication for individuals 
        with hearing and speech challenges.  
        We're committed to **accessibility** and **innovation**.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

elif app_mode == 'Sign Language to Text':
    st.title('✋ Sign Language to Text')

    use_webcam = st.sidebar.button('🎥 Use Webcam')
    record = st.sidebar.checkbox("🔴 Record Video")
    if record:
        st.checkbox("Recording", value=True)

    st.sidebar.markdown('---')
    sld = ""
    #st.markdown(' ## 📝 Output')
    #st.markdown(sld)
    # Glassmorphism Output Display
    st.markdown("""
    <style>
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.3);
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="glass-card"><h4>📝 Detected Output</h4><p style="font-size:22px; color:white;">{}</p></div>'.format(sld), unsafe_allow_html=True)


    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("📁 Upload a video", type=["mp4", "mov", 'avi', 'asf', 'm4v'])
    tfflie = tempfile.NamedTemporaryFile(delete=False)

    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tfflie.name = DEMO_VIDEO
    else:
        tfflie.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tfflie.name)
    

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))

    codec = cv2.VideoWriter_fourcc('V', 'P', '0', '9')
    out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    st.markdown("<hr/>", unsafe_allow_html=True)

    finger_tips = [8, 12, 16, 20]
    thumb_tip = 4
    threshold = 0.1  
    while True:
        ret, img = vid.read()
        img = cv2.flip(img, 1)
        h, w, c = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img)

        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                lm_list = []
                for id, lm in enumerate(hand_landmark.landmark):
                    lm_list.append(lm)
                finger_fold_status = []
                for tip in finger_tips:
                    x, y = int(lm_list[tip].x * w), int(lm_list[tip].y * h)
                   

                    if lm_list[tip].x < lm_list[tip - 2].x:
                       
                        finger_fold_status.append(True)
                    else:
                        finger_fold_status.append(False)

                print(finger_fold_status)
                x, y = int(lm_list[8].x * w), int(lm_list[8].y * h)
                print(x, y)
               

                # one
                if lm_list[3].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y > lm_list[10].y and \
                        lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y and lm_list[4].y < lm_list[
                    12].y:
                    cv2.putText(img, "ONE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("1")
                    current_word += "1"
                    sld="One"
                    last_detected_time = time.time()

                # two
                if lm_list[3].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y:
                    cv2.putText(img, "TWO", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("2")
                    current_word += "2"
                    sld="Two"
                    last_detected_time = time.time()

                # three
                if lm_list[2].x < lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y:
                    cv2.putText(img, "THREE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("3")
                    current_word += "3"
                    sld="Three"
                    last_detected_time = time.time()

                # four
                if lm_list[2].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y and lm_list[2].x < lm_list[8].x:
                    cv2.putText(img, "FOUR", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("4")
                    current_word += "4"
                    sld="Four"
                    last_detected_time = time.time()

                # five
                if lm_list[2].x < lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y and lm_list[17].x < lm_list[
                    0].x < \
                        lm_list[5].x:
                    cv2.putText(img, "FIVE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("5")
                    current_word += "5"
                    sld="Five"
                    last_detected_time = time.time()

                    # six
                if lm_list[2].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y < lm_list[14].y and lm_list[20].y > lm_list[18].y and lm_list[17].x < lm_list[
                    0].x < \
                        lm_list[5].x:
                    cv2.putText(img, "SIX", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("6")
                    current_word += "6"
                    sld="Six"
                    last_detected_time = time.time()

                # SEVEN
                if lm_list[2].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y > lm_list[14].y and lm_list[20].y < lm_list[18].y and lm_list[17].x < lm_list[
                    0].x < \
                        lm_list[5].x:
                    cv2.putText(img, "SEVEN", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("7")
                    current_word += "7"
                    sld="Seven"
                    last_detected_time = time.time()

                # EIGHT
                if lm_list[2].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y > lm_list[10].y and \
                        lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y and lm_list[17].x < lm_list[
                    0].x < \
                        lm_list[5].x:
                    cv2.putText(img, "EIGHT", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("8")
                    current_word += "8"
                    sld="Eight"
                    last_detected_time = time.time()

                # NINE
                if lm_list[2].x > lm_list[4].x and lm_list[8].y > lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y and lm_list[17].x < lm_list[
                    0].x < \
                        lm_list[5].x:
                    cv2.putText(img, "NINE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("9")
                    current_word += "9"
                    sld="Nine"
                    last_detected_time = time.time()

                # A
                if lm_list[2].y > lm_list[4].y and lm_list[8].y > lm_list[6].y and lm_list[12].y > lm_list[10].y and \
                        lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y and lm_list[17].x < lm_list[
                    0].x < \
                        lm_list[5].x and lm_list[4].y < lm_list[6].y:
                    cv2.putText(img, "A", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("A")
                    current_word += "A"
                    sld="A"
                    last_detected_time = time.time()

                # B
                if lm_list[2].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y and lm_list[2].x > lm_list[8].x:
                    cv2.putText(img, "B", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("B")
                    current_word += "B"
                    sld="B"
                    last_detected_time = time.time()

                # C
                if lm_list[2].x < lm_list[4].x and lm_list[8].x > lm_list[6].x and lm_list[12].x > lm_list[10].x and \
                        lm_list[16].x > lm_list[14].x and lm_list[20].x > lm_list[18].x:
                    cv2.putText(img, "C", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("C")
                    current_word += "C"
                    sld="C"
                    last_detected_time = time.time()

                # D
                if lm_list[3].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y > lm_list[10].y and \
                        lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y and lm_list[4].y > lm_list[8].y:
                    cv2.putText(img, "D", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("D")
                    current_word += "D"
                    sld="D"
                    last_detected_time = time.time()

                # E
                if lm_list[2].x > lm_list[4].x and lm_list[8].y > lm_list[6].y and lm_list[12].y > lm_list[10].y and \
                        lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y and lm_list[17].x < lm_list[
                    0].x < \
                        lm_list[5].x and lm_list[4].y > lm_list[6].y:
                    cv2.putText(img, "E", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("E")
                    current_word += "E"
                    sld="E"
                    last_detected_time = time.time()

               
              
                #F
                if lm_list[2].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and \
                        lm_list[12].y > lm_list[10].y and lm_list[16].y > lm_list[14].y and \
                        lm_list[20].y < lm_list[18].y and lm_list[8].x > lm_list[4].x:
                    cv2.putText(img, "F", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("F")
                    current_word += "F"
                    sld="F"
                    last_detected_time = time.time()

                # G
                if lm_list[2].x > lm_list[4].x and lm_list[8].y > lm_list[6].y and \
                        lm_list[8].x < lm_list[6].x and lm_list[12].y > lm_list[10].y and \
                        lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y:
                    cv2.putText(img, "G", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("G")
                    current_word += "G"
                    sld="G"
                    last_detected_time = time.time()

                # H
                if lm_list[2].x < lm_list[4].x and lm_list[8].y < lm_list[6].y and \
                        lm_list[12].y < lm_list[10].y and lm_list[16].y > lm_list[14].y and \
                        lm_list[20].y > lm_list[18].y:
                    cv2.putText(img, "H", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("H")
                    current_word += "H"
                    sld="H"
                    last_detected_time = time.time()

                # I
                if lm_list[2].x < lm_list[4].x and lm_list[8].y > lm_list[6].y and \
                        lm_list[12].y > lm_list[10].y and lm_list[16].y > lm_list[14].y and \
                        lm_list[20].y < lm_list[18].y:
                    cv2.putText(img, "I", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("I")
                    current_word += "I"
                    sld="I"
                    last_detected_time = time.time()

                # J
                if lm_list[2].x > lm_list[4].x and lm_list[8].y > lm_list[6].y and \
                        lm_list[12].y > lm_list[10].y and lm_list[16].y > lm_list[14].y and \
                        lm_list[20].y < lm_list[18].y:
                    cv2.putText(img, "J", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("J")
                    current_word += "J"
                    sld="J"
                    last_detected_time = time.time()

                # K
                if lm_list[2].x < lm_list[4].x and lm_list[8].y < lm_list[6].y and \
                        lm_list[12].y < lm_list[10].y and lm_list[16].y > lm_list[14].y and \
                        lm_list[20].y < lm_list[18].y:
                    cv2.putText(img, "K", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("K")
                    current_word += "K"
                    sld="K"
                    last_detected_time = time.time()

                # L
                if lm_list[2].x < lm_list[4].x and lm_list[8].y < lm_list[6].y and \
                        lm_list[12].y > lm_list[10].y and lm_list[16].y > lm_list[14].y and \
                        lm_list[20].y > lm_list[18].y:
                    cv2.putText(img, "L", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("L")
                    current_word += "L"
                    sld="L"
                    last_detected_time = time.time()
                # M
                    threshold = 20 
                if (lm_list[2].x < lm_list[4].x - threshold and lm_list[8].y > lm_list[6].y + threshold and
                    lm_list[12].y > lm_list[10].y + threshold and lm_list[16].y > lm_list[14].y + threshold and
                    lm_list[20].y > lm_list[18].y + threshold and lm_list[4].y < lm_list[8].y - threshold and
                    lm_list[2].x > 0 and lm_list[2].x < img.shape[1] and lm_list[4].x > 0 and lm_list[4].x < img.shape[1] and
                    lm_list[6].y > 0 and lm_list[6].y < img.shape[0] and lm_list[8].y > 0 and lm_list[8].y < img.shape[0]):
                    cv2.putText(img, "M", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("M")
                    current_word += "M"
                    sld="M" 
                    last_detected_time = time.time() 

                # N
                if lm_list[2].x < lm_list[4].x and lm_list[8].y > lm_list[6].y and \
                        lm_list[12].y > lm_list[10].y and lm_list[16].y > lm_list[14].y and \
                        lm_list[20].y > lm_list[18].y:
                    cv2.putText(img, "N", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("N")
                    current_word += "N"
                    sld="N"
                    last_detected_time = time.time()

                # O
                if lm_list[2].x < lm_list[4].x and lm_list[8].y < lm_list[6].y and \
                        lm_list[8].x < lm_list[6].x and lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y:
                    cv2.putText(img, "O", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("O")
                    current_word += "O"
                    sld="O"
                    last_detected_time = time.time()

                # P
                if lm_list[2].x < lm_list[4].x and lm_list[8].y < lm_list[6].y and \
                        lm_list[12].y < lm_list[10].y and lm_list[16].y > lm_list[14].y and \
                        lm_list[20].y < lm_list[18].y:
                    cv2.putText(img, "P", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("P")
                    current_word += "P"
                    sld="P"
                    last_detected_time = time.time()

                # Q
                if lm_list[8].y < lm_list[6].y and lm_list[8].x > lm_list[6].x and \
                        lm_list[4].y > lm_list[3].y and lm_list[4].x < lm_list[3].x and \
                        lm_list[12].y > lm_list[10].y and lm_list[16].y > lm_list[14].y and \
                        lm_list[20].y > lm_list[18].y:
                    cv2.putText(img, "Q", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("Q")
                    current_word += "Q"
                    sld = "Q"
                    last_detected_time = time.time()

                # R
                if lm_list[2].x < lm_list[4].x and lm_list[8].y < lm_list[6].y and \
                        lm_list[12].y > lm_list[10].y and lm_list[16].y > lm_list[14].y and \
                        lm_list[20].y > lm_list[18].y:
                    cv2.putText(img, "R", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("R")
                    current_word += "R"
                    sld="R"
                    last_detected_time = time.time()

                # S
                if lm_list[2].x < lm_list[4].x and lm_list[8].y > lm_list[6].y and \
                        lm_list[12].y > lm_list[10].y and lm_list[16].y > lm_list[14].y and \
                        lm_list[20].y > lm_list[18].y:
                    cv2.putText(img, "S", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("S")
                    current_word += "S"
                    sld="S"
                    last_detected_time = time.time()

                # T
                if lm_list[2].x < lm_list[4].x and lm_list[8].y > lm_list[6].y and \
                        lm_list[12].y > lm_list[10].y and lm_list[16].y > lm_list[14].y and \
                        lm_list[20].y > lm_list[18].y and lm_list[4].y < lm_list[10].y:
                    cv2.putText(img, "T", (20, 30), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 3)
                    my_list.append("T")
                    current_word += "T"
                    sld="T"
                    last_detected_time = time.time()

                # U
                if lm_list[2].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and \
                        lm_list[12].y < lm_list[10].y and lm_list[16].y < lm_list[14].y and \
                        lm_list[20].y < lm_list[18].y and lm_list[4].y > lm_list[10].y:
                    cv2.putText(img, "U", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("U")
                    current_word += "U"
                    sld="U"
                    last_detected_time = time.time()

                # V
                if lm_list[2].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and \
                        lm_list[12].y < lm_list[10].y and lm_list[16].y < lm_list[14].y and \
                        lm_list[20].y < lm_list[18].y and lm_list[4].y > lm_list[10].y and \
                        lm_list[8].y > lm_list[14].y:
                    cv2.putText(img, "V", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("V")
                    current_word += "V"
                    sld="V"
                    last_detected_time = time.time()

                # W
                if lm_list[2].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and \
                        lm_list[12].y < lm_list[10].y and lm_list[16].y < lm_list[14].y and \
                        lm_list[20].y < lm_list[18].y and lm_list[4].y > lm_list[10].y and \
                        lm_list[8].y > lm_list[14].y and lm_list[12].y > lm_list[18].y:
                    cv2.putText(img, "W", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("W")
                    current_word += "W"
                    sld="W"
                    last_detected_time = time.time()

                # X
                if lm_list[2].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and \
                        lm_list[12].y < lm_list[10].y and lm_list[16].y > lm_list[14].y and \
                        lm_list[20].y < lm_list[18].y and lm_list[8].x < lm_list[6].x:
                    cv2.putText(img, "X", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("X")
                    current_word += "X"
                    sld="X"
                    last_detected_time = time.time()

                # Y
                if lm_list[2].x < lm_list[4].x and lm_list[8].y < lm_list[6].y and \
                        lm_list[12].y < lm_list[10].y and lm_list[16].y > lm_list[14].y and \
                        lm_list[20].y < lm_list[18].y and lm_list[8].x < lm_list[6].x:
                    cv2.putText(img, "Y", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("Y")
                    current_word += "Y"
                    sld="Y"
                    last_detected_time = time.time()
                # Z
                if lm_list[2].x < lm_list[4].x and lm_list[8].y < lm_list[6].y and \
                        lm_list[12].y > lm_list[10].y and lm_list[16].y > lm_list[14].y and \
                        lm_list[20].y < lm_list[18].y:
                    cv2.putText(img, "Z", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("Z")
                    current_word += "Z"
                    sld="Z"
                    last_detected_time = time.time()

                # Best of Luck Gesture Detection (Thumbs-Up Style for Right Hand)
                if lm_list[4].y < lm_list[3].y and \
                        lm_list[8].y > lm_list[6].y and \
                        lm_list[12].y > lm_list[10].y and \
                        lm_list[16].y > lm_list[14].y and \
                        lm_list[20].y > lm_list[18].y and \
                        lm_list[4].x < lm_list[5].x:
                    cv2.putText(img, "Best of Luck", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    my_list.append("Best of Luck")
                    current_word += "Best of Luck"
                    sld = "Best of Luck"
                    last_detected_time = time.time()   
                    
                # "I'm Happy" Gesture Detection
                if lm_list[4].y < lm_list[3].y and lm_list[8].y < lm_list[7].y and \
                        lm_list[12].y < lm_list[11].y and lm_list[16].y < lm_list[15].y and \
                        lm_list[20].y < lm_list[19].y and lm_list[4].x > lm_list[5].x and \
                        lm_list[4].x < lm_list[9].x and lm_list[8].x > lm_list[9].x and \
                        lm_list[12].x > lm_list[9].x and lm_list[16].x > lm_list[9].x:
                    cv2.putText(img, "I'm Happy", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    my_list.append("I'm Happy")
                    current_word += "I'm Happy"
                    sld = "I'm Happy"
                    last_detected_time = time.time()
                    
                # "Bye" Gesture Detection
                if lm_list[4].y < lm_list[3].y and lm_list[8].y < lm_list[7].y and \
                        lm_list[12].y < lm_list[11].y and lm_list[16].y < lm_list[15].y and \
                        lm_list[20].y < lm_list[19].y and lm_list[4].x < lm_list[5].x and \
                        lm_list[8].x < lm_list[9].x and lm_list[12].x < lm_list[9].x and \
                        lm_list[16].x < lm_list[9].x and lm_list[20].x < lm_list[19].x:  # Horizontal motion to the left
                    cv2.putText(img, "Bye", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    my_list.append("Bye")
                    current_word += "Bye"
                    sld = "Bye"
                    last_detected_time = time.time()
                
                # "Have a Good Day" Gesture Detection – Unique right-hand pose
                if lm_list[4].y > lm_list[3].y and \
                        lm_list[8].y < lm_list[7].y and \
                        lm_list[12].y < lm_list[11].y and \
                        lm_list[16].y > lm_list[15].y and \
                        lm_list[20].y > lm_list[19].y and \
                        abs(lm_list[8].x - lm_list[12].x) > 0.05:
                    cv2.putText(img, "Have a Good Day", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0), 3)
                    my_list.append("Have a Good Day")
                    current_word += "Have a Good Day"
                    sld = "Have a Good Day"
                    last_detected_time = time.time()
                    
                # "OK" Gesture Detection – Right hand only
                if lm_list[4].x > lm_list[3].x and \
                        lm_list[8].x > lm_list[6].x and \
                        abs(lm_list[4].x - lm_list[8].x) < 0.04 and \
                        abs(lm_list[4].y - lm_list[8].y) < 0.04 and \
                        lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y < lm_list[14].y and \
                        lm_list[20].y < lm_list[18].y:
                    cv2.putText(img, "OK", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 3)
                    my_list.append("OK")
                    current_word += "OK"
                    sld = "OK"
                    last_detected_time = time.time()

            mp_draw.draw_landmarks(img, hand_landmark,
                                   mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec((0, 0, 255), 6, 3),
                                   mp_draw.DrawingSpec((0, 255, 0), 4, 2)
                                   )
            if record:

                out.write(img)


            frame = cv2.resize(img, (0, 0), fx=0.8, fy=0.8)
            frame = image_resize(image=frame, width=640)
            stframe.image(frame, channels='BGR', use_container_width=True)




    st.text('Video Processed')

    output_video = open('output1.mp4', 'rb')
    out_bytes = output_video.read()
    st.video(out_bytes)

    vid.release()
    out.release()
