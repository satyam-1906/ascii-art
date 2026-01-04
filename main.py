from matplotlib import text
import numpy as np
import cv2
import mediapipe as mp
import time

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

x_coord1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
y_coord1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
z_coord1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
x_coord2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
y_coord2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
z_coord2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
flag = True
colour = (0, 0, 0)
c = 0
let_click1 = True
let_click2 = True
distance1 = 1.0
distance2 = 1.0
hand1 = 0
hand2 = 0

def print_result(result, output_image, timestamp_ms):
    global x_coord1, y_coord1, z_coord1, hand1, hand2, x_coord2, y_coord2, z_coord2
    # set coords when a hand is found, otherwise reset and mark absent
    if result.hand_landmarks and len(result.hand_landmarks) > 0:
        if len(result.handedness) == 2:
            hand1 = 1
            hand2 = 1
        elif len(result.handedness) == 1:
            if result.handedness[0][0].category_name == 'Left':
                hand1 = 0
                hand2 = 1
            else:
                hand1 = 1
                hand2 = 0
        else:
            hand1 = 0
            hand2 = 0
        if len(result.handedness) == 1:
            if hand1 == 1:
                lm_list = result.hand_landmarks[0]
                for i, lm in enumerate(lm_list):
                    x_coord1[i] = lm.x
                    y_coord1[i] = lm.y
                    z_coord1[i] = lm.z
                x_coord2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                y_coord2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                z_coord2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            elif hand2 == 1:
                lm_list = result.hand_landmarks[0]
                for i, lm in enumerate(lm_list):
                    x_coord2[i] = lm.x
                    y_coord2[i] = lm.y
                    z_coord2[i] = lm.z
                x_coord1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                y_coord1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                z_coord1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif len(result.handedness) == 2:
            if result.handedness[0][0].category_name == 'Right':
                lm_list = result.hand_landmarks[0]
                for i, lm in enumerate(lm_list):
                    x_coord1[i] = lm.x
                    y_coord1[i] = lm.y
                    z_coord1[i] = lm.z
                lm_list = result.hand_landmarks[1]
                for i, lm in enumerate(lm_list):
                    x_coord2[i] = lm.x
                    y_coord2[i] = lm.y
                    z_coord2[i] = lm.z
            else:
                lm_list = result.hand_landmarks[1]
                for i, lm in enumerate(lm_list):
                    x_coord1[i] = lm.x
                    y_coord1[i] = lm.y
                    z_coord1[i] = lm.z
                lm_list = result.hand_landmarks[0]
                for i, lm in enumerate(lm_list):
                    x_coord2[i] = lm.x
                    y_coord2[i] = lm.y
                    z_coord2[i] = lm.z
        present = True
    else:
        x_coord1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        y_coord1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        z_coord1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        x_coord2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        y_coord2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        z_coord2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        present = False
    videoFeed(output_image, x_coord1, y_coord1, present)

def videoFeed(img, Xc1, Yc1, present):
    global flag, colour, let_click1, distance1, z_coord1, hand1, hand2, let_click2, c, distance2
    resolutions = [640, 480, 320, 160]
    n_frame = img.numpy_view()
    new_frame = np.copy(n_frame)
    new_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2BGR)
    gray_fil = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    sizes = [0.3, 0.5, 0.7, 1]
    thicknesses = [2, 3, 4, 5]
    if distance1 > 0.1:
        let_click1 = True
    if present and hand1 == 1:
        distance1 = ((Xc1[4]-Xc1[8])**2 + (Yc1[4]-Yc1[8])**2 + (z_coord1[4]-z_coord1[8])**2)**0.5
        if distance1 <= 0.05 and flag == True:
            if let_click1:
                colour = (0, 0, 0)
                flag = False
                let_click1 = False
        elif distance1 <= 0.05 and flag == False:
            if let_click1:
                colour = (255, 0, 0)
                flag = True
                let_click1 = False
    if distance2 > 0.1:
        let_click2 = True
    if present and hand2 == 1:
        distance2 = ((x_coord2[4]-x_coord2[8])**2 + (y_coord2[4]-y_coord2[8])**2 + (z_coord2[4]-z_coord2[8])**2)**0.5
        if distance2 <= 0.05:
            if let_click2:
                if c < 3:
                    c += 1
                else:
                    c = 0
                let_click2 = False
    resize_fil = cv2.resize(gray_fil, (resolutions[c]//10, resolutions[c]//10))
    text = ['$','@','B','%','8','&','W','M','#','*','o','a','h','k','b','d','p','q','w','m','Z','O','Q','L','C','J','U','Y','X','z','c','v','u','n','x','r','j','f','t', '/ ','| ','(',') ','1 ','{ ','} ','[ ','] ','?','- ','_ ','+','=','~ ','< ', '> ','i ','! ','l ','I ',': ',';','. ']
    canvas = np.zeros((640, 640, 3), dtype="uint8")
    canvas.fill(255)
    for y in range(resolutions[c]//10):
        for x in range(resolutions[c]//10):
            pixel = resize_fil[y, x]
            index = int((pixel / 255) * (len(text) - 1))
            cv2.putText(canvas, text[index], (x*(6400//resolutions[c]), y*(6400//resolutions[c])), cv2.FONT_HERSHEY_SIMPLEX, sizes[c], colour, thicknesses[c])
    if present and hand1 == 1:
        cv2.circle(new_frame, (int(Xc1[4]*new_frame.shape[1]), int(Yc1[4]*new_frame.shape[0])), 5, (255, 255, 255), -1)
        cv2.circle(new_frame, (int(Xc1[8]*new_frame.shape[1]), int(Yc1[8]*new_frame.shape[0])), 5, (255, 255, 255), -1)
        cv2.line(new_frame, (int(Xc1[4]*new_frame.shape[1]), int(Yc1[4]*new_frame.shape[0])), (int(Xc1[8]*new_frame.shape[1]), int(Yc1[8]*new_frame.shape[0])), (255, 255, 255), 2)
    if present and hand2 == 1:
        cv2.circle(new_frame, (int(x_coord2[4]*new_frame.shape[1]), int(y_coord2[4]*new_frame.shape[0])), 5, (255, 255, 255), -1)
        cv2.circle(new_frame, (int(x_coord2[8]*new_frame.shape[1]), int(y_coord2[8]*new_frame.shape[0])), 5, (255, 255, 255), -1)
        cv2.line(new_frame, (int(x_coord2[4]*new_frame.shape[1]), int(y_coord2[4]*new_frame.shape[0])), (int(x_coord2[8]*new_frame.shape[1]), int(y_coord2[8]*new_frame.shape[0])), (255, 255, 255), 2)
    flip = cv2.flip(new_frame, 1)
    if present and hand1 == 1:
        cv2.putText(flip, f'D1: {distance1:.4f}', (int((1-Xc1[8]+(Xc1[4]-Xc1[8])/2)*new_frame.shape[1]), int((Yc1[8]+(Yc1[4]-Yc1[8])/2)*new_frame.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) 
    if present and hand2 == 1:
        cv2.putText(flip, f'D2: {distance2:.4f}', (int((1-x_coord2[8]+(x_coord2[4]-x_coord2[8])/2)*new_frame.shape[1]), int((y_coord2[8]+(y_coord2[4]-y_coord2[8])/2)*new_frame.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)   
    cv2.imshow('Live Video Feed', flip)
    cv2.imshow('Drawing Canvas', canvas)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()


cap = cv2.VideoCapture(0)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM, min_hand_detection_confidence=0.5, min_tracking_confidence=0.5, num_hands=2,
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