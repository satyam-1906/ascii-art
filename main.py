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

x_coord = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
y_coord = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
z_coord = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
flag = True
colour = (0, 0, 0)
c = 0
let_click = True
distance = 1.0

def print_result(result, output_image, timestamp_ms):
    global x_coord, y_coord, z_coord
    # set coords when a hand is found, otherwise reset and mark absent
    if result.hand_landmarks and len(result.hand_landmarks) > 0:
        lm_list = result.hand_landmarks[0]
        for i, lm in enumerate(lm_list):
            x_coord[i] = lm.x
            y_coord[i] = lm.y
            z_coord[i] = lm.z
        present = True
    else:
        x_coord = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        y_coord = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        z_coord = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        present = False
    videoFeed(output_image, x_coord, y_coord, present)

def videoFeed(img, Xc, Yc, present):
    global flag, colour, let_click, distance, z_coord
    n_frame = img.numpy_view()
    new_frame = np.copy(n_frame)
    new_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2BGR)
    gray_fil = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    resize_fil = cv2.resize(gray_fil, (64, 64))
    text = ['$','@','B','%','8','&','W','M','#','*','o','a','h','k','b','d','p','q','w','m','Z','O','Q','L','C','J','U','Y','X','z','c','v','u','n','x','r','j','f','t','/','|','(',')','1','{','}','[',']','?','-','_','+','=','~','<','>','i','!','l','I',':',';','.',' ']
    canvas = np.zeros((640, 640, 3), dtype="uint8")
    canvas.fill(255)
    if distance > 0.1:
        let_click = True
    if present:
        distance = ((Xc[4]-Xc[8])**2 + (Yc[4]-Yc[8])**2 + (z_coord[4]-z_coord[8])**2)**0.5
        if distance <= 0.05 and flag == True:
            if let_click:
                colour = (0, 0, 0)
                flag = False
                let_click = False
        elif distance <= 0.05 and flag == False:
            if let_click:
                colour = (255, 0, 0)
                flag = True
                let_click = False
    for y in range(64):
        for x in range(64):
            pixel = resize_fil[y, x]
            index = int((pixel / 255) * (len(text) - 1))
            cv2.putText(canvas, text[index], (x*10, y*10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color=colour, thickness=2)
    cv2.circle(new_frame, (int(Xc[4]*new_frame.shape[1]), int(Yc[4]*new_frame.shape[0])), 5, (255, 255, 255), -1)
    cv2.circle(new_frame, (int(Xc[8]*new_frame.shape[1]), int(Yc[8]*new_frame.shape[0])), 5, (255, 255, 255), -1)
    cv2.line(new_frame, (int(Xc[4]*new_frame.shape[1]), int(Yc[4]*new_frame.shape[0])), (int(Xc[8]*new_frame.shape[1]), int(Yc[8]*new_frame.shape[0])), (255, 255, 255), 2)
    flip = cv2.flip(new_frame, 1)
    cv2.putText(flip, f'D: {distance:.4f}', (int((1-Xc[8]+(Xc[4]-Xc[8])/2)*new_frame.shape[1]), int((Yc[8]+(Yc[4]-Yc[8])/2)*new_frame.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)    
    cv2.imshow('Live Video Feed', flip)
    cv2.imshow('Drawing Canvas', canvas)
    
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