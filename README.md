# 🚗😴 Driver Drowsiness Detection Agent

An **AI-powered real-time driver monitoring system** that detects **eye closure, prolonged drowsiness, and yawning** using computer vision.
The system alerts the driver through **beep alarms, voice alerts, and Tamil songs** to prevent accidents caused by fatigue.

---

# 🎯 Project Objective

Driver fatigue is one of the major causes of road accidents.
This project builds an **intelligent monitoring agent** that continuously analyzes the driver's face through a webcam and detects signs of drowsiness in real time.

When fatigue is detected, the system immediately triggers alerts to wake the driver.

---

# ⚡ Key Features

👁 **Eye Closure Detection**
Detects blinking and eye closure using **Eye Aspect Ratio (EAR)**.

🔔 **High Beep Alarm**
Triggers a beep when eyes stay closed for several frames.

🔊 **Voice Alert**
Plays **“Please stay awake buddy!”** if eyes remain closed for a longer duration.

🥱 **Yawn Detection**
Detects yawning using **Mouth Aspect Ratio (MAR)**.

🎵 **Tamil Song Activation**
Plays Tamil songs when yawning is detected to keep the driver active.

📊 **Session Monitoring Dashboard**
Shows real-time metrics:

* EAR value
* MAR value
* Eye status
* Drowsiness level

📄 **PDF Session Report**
Exports a **one-page report of driver alert events**.

📱 **Mobile Monitoring**
Dashboard can be opened on a mobile device for car mounting.

---

# 🧠 System Architecture

```
            ┌─────────────────────┐
            │   Driver Webcam     │
            │   (Video Input)     │
            └──────────┬──────────┘
                       │
                       ▼
             ┌───────────────────┐
             │   OpenCV Capture   │
             │   Frame Processing │
             └──────────┬─────────┘
                        │
                        ▼
          ┌──────────────────────────┐
          │ MediaPipe Face Mesh Model│
          │ 468 Facial Landmarks     │
          └──────────┬───────────────┘
                     │
                     ▼
       ┌──────────────────────────────┐
       │ Drowsiness Detection Engine  │
       │                              │
       │ • Eye Aspect Ratio (EAR)     │
       │ • Mouth Aspect Ratio (MAR)   │
       │ • Blink Detection            │
       │ • Yawn Detection             │
       └───────────┬──────────────────┘
                   │
                   ▼
        ┌────────────────────────────┐
        │ Alert Management System     │
        │                             │
        │ 🔔 Beep Alarm               │
        │ 🔊 Voice Warning            │
        │ 🎵 Tamil Song Activation    │
        └───────────┬────────────────┘
                    │
                    ▼
        ┌────────────────────────────┐
        │ Web Dashboard (Frontend)   │
        │ Real-time Monitoring UI    │
        └────────────────────────────┘
```

---

# 🛠️ Technologies Used

| Layer                   | Technology               |
| ----------------------- | ------------------------ |
| Programming             | Python                   |
| Computer Vision         | OpenCV                   |
| AI Face Detection       | MediaPipe Face Mesh      |
| Backend Framework       | Flask                    |
| Real-time Communication | Flask-SocketIO           |
| Frontend                | HTML5, CSS3, JavaScript  |
| Audio Alerts            | Web Audio API            |
| Data Processing         | NumPy                    |
| Report Generation       | ReportLab                |
| Web Camera Access       | Browser getUserMedia API |

---

# 📂 Project Structure

```
driver-drowsiness-detection-agent/
│
├── app.py
├── requirements.txt
│
├── detector/
│   ├── __init__.py
│   └── eye_detector.py
│
├── templates/
│   └── index.html
│
├── static/
│   └── audio/
│       ├── song1.mp3
│       ├── song2.mp3
│       └── song3.mp3
│
└── README.md
```

---

# 🚀 Installation & Usage

### 1️⃣ Clone the Repository

```
git clone https://github.com/yourusername/driver-drowsiness-detection-agent.git
cd driver-drowsiness-detection-agent
```

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Run the Application

```
python app.py
```

### 4️⃣ Open in Browser

```
http://localhost:5000
```

Allow **camera access** and click **Start Monitoring**.

---

# 📊 Detection Logic

**Eye Aspect Ratio (EAR)**
Detects eye closure by measuring the ratio between vertical and horizontal eye landmarks.

**Mouth Aspect Ratio (MAR)**
Detects yawning based on mouth opening distance.

If thresholds exceed limits → **alert system triggers**.

---

# 🌟 Future Improvements

🚘 Night vision detection
📱 Android dashboard app
🧠 Deep learning fatigue prediction
📡 Cloud monitoring for fleet drivers

---

# 👨‍💻 Author

**Kanagaraj Raj**

AI / Machine Learning Enthusiast
Passionate about building **real-world AI safety systems**.

---

⭐ If you like this project, consider **starring the repository!**
