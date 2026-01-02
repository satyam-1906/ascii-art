import numpy as np
import cv2
import mediapipe as mp
import time

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

x_coord = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
y_coord = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
flag = True
colour = (0, 0, 0)
c = 0
let_click = True


def print_result(result, output_image, timestamp_ms):
    global x_coord, y_coord
    # set coords when a hand is found, otherwise reset and mark absent
    if result.hand_landmarks and len(result.hand_landmarks) > 0:
        lm_list = result.hand_landmarks[0]
        for i, lm in enumerate(lm_list):
            x_coord[i] = lm.x
            y_coord[i] = lm.y

        present = True
    else:
        x_coord = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        y_coord = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        present = False
    videoFeed(output_image, x_coord, y_coord, present)

def videoFeed(img, Xc, Yc, present):
    global flag, colour, let_click
    n_frame = img.numpy_view()
    new_frame = np.copy(n_frame)
    new_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2BGR)
    flip = cv2.flip(new_frame, 1)
    gray_fil = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    resize_fil = cv2.resize(gray_fil, (64, 64))
    text1 = 'O'
    text2 = '_'
    if Yc[4]-Yc[8] > 0.1:
        let_click = True
    if present:
        if Xc[4]-Xc[8] <= 0.005 and Yc[4]-Yc[8] <= 0.1 and flag == True:
            if let_click:
                colour = (0, 0, 0)
                flag = False
                let_click = False
        elif Xc[4]-Xc[8] <= 0.005 and Yc[4]-Yc[8] <= 0.1 and flag == False:
            if let_click:
                colour = (255, 0, 0)
                flag = True
                let_click = False
    for y in range(64):
        for x in range(64):
            pixel = resize_fil[y, x]
            if pixel > 127:
                cv2.putText(flip, text1, (x*6, y*6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=colour, thickness=2)
            else:
                cv2.putText(flip, text2, (x*6, y*6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=colour, thickness=2)
    cv2.imshow('Live Video Feed', flip)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()


cap = cv2.VideoCapture(0)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM, min_hand_detection_confidence=0.5, min_tracking_confidence=0.5, num_hands=1,
    result_callback=print_result)

with HandLandmarker.create_from_options(options) as landmarker:
    
    last_timestamp_ms = 0
    

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        now_ms = int(time.monotonic() * 1000)
        if now_ms <= last_timestamp_ms:
            now_ms = last_timestamp_ms + 1
        last_timestamp_ms = now_ms
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=gray)
        landmarker.detect_async(mp_image, last_timestamp_ms)