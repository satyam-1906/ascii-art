# ✋ Gesture-Controlled ASCII Art (Computer Vision Project)

A **real-time computer vision project** where I use **both hands** to control the appearance of **ASCII art** generated from a live camera feed.

Using **finger tap gestures**, I can interact with the ASCII output in real-time:
- ✅ **Right-hand finger taps** control the **color modes** of the ASCII art  
- ✅ **Left-hand finger taps** control the **resolution/detail level** of the ASCII art  

Built using **OpenCV**, **Google MediaPipe**, and **text-based ASCII rendering techniques (TextTags)**.

---

## 🎥 Demo
📌 A demo video is included in this repository or available on my LinkedIn post.

---

## 🚀 Features
- 📷 Live camera input using OpenCV
- 🖐️ Real-time hand tracking using MediaPipe Hands
- 🧠 Gesture recognition using fingertip landmark geometry
- 🎨 Multiple ASCII color modes
- 🔍 Dynamic ASCII resolution control
- ⚡ Low-latency real-time interaction

---

## 🛠️ Tech Stack
- Python
- OpenCV
- Google MediaPipe (Hands)
- Text-based ASCII rendering (TextTags)
- Real-time gesture-based HCI

---

## 🧩 How It Works

### 1. Live Video Capture
Frames are captured in real time using OpenCV and passed through the vision pipeline.

### 2. Hand Tracking
MediaPipe Hands detects and tracks hands, providing **21 landmarks per hand** with normalized coordinates.

### 3. Gesture Recognition
Finger tap gestures are detected by calculating distances between fingertip landmarks (e.g., thumb–index finger).

### 4. Two-Hand Control Mapping
- **Right Hand**: Controls ASCII color modes  
- **Left Hand**: Controls ASCII resolution by adjusting sampling density or block size

### 5. ASCII Art Generation
1. Convert frame to grayscale or RGB  
2. Resize based on resolution level  
3. Map pixel intensities to ASCII characters  
4. Render text using OpenCV  

All steps execute frame-by-frame in real time.

---

## 📂 Project Structure
```
Gesture-Controlled-ASCII-Art/
├── main.py
├── requirements.txt
├── README.md
└── demo.mp4
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
```

---

## ▶️ Run
```bash
python main.py
```

---

## 📌 Requirements
```
opencv-python
mediapipe
numpy
```

---

## 🧠 Key Learnings
- Real-time hand landmark tracking
- Gesture-based interaction design
- Bimanual control systems
- ASCII rendering techniques
- Computer vision performance optimization

---

## 🌟 Future Improvements
- Add more gesture mappings
- Improve gesture smoothing and stability
- Support multiple ASCII character sets
- Add on-screen UI indicators

---

## 🙌 Acknowledgements
- Google MediaPipe
- OpenCV
- ASCII art rendering inspirations
