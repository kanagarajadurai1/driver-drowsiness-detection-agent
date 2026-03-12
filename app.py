"""
DrowseGuard AI - Fixed Version
"""
import cv2, numpy as np, base64, time, threading, os
import mediapipe as mp
from flask import Flask, render_template, jsonify, Response
from flask_socketio import SocketIO, emit
from detector import DrowsinessDetector, eye_aspect_ratio, mouth_aspect_ratio, LEFT_EYE, RIGHT_EYE, MOUTH_OUTER

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dg2024'
# KEY FIX: use threading, not eventlet
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', logger=False, engineio_logger=False)

# ── MediaPipe ─────────────────────────────────────────────────
face_mesh = None
USE_TASKS = False

try:
    from mediapipe.tasks.python import vision as mpv
    from mediapipe.tasks import python as mpp
    MODEL = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
    if not os.path.exists(MODEL):
        import urllib.request
        print("[*] Downloading model...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            MODEL)
        print("[OK] Downloaded!")
    face_mesh = mpv.FaceLandmarker.create_from_options(mpv.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=MODEL),
        num_faces=1, min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5, min_tracking_confidence=0.5))
    USE_TASKS = True
    print("[OK] MediaPipe Tasks API ready")
except Exception as e:
    print(f"[~] Tasks API failed ({e}), trying legacy...")
    try:
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        USE_TASKS = False
        print("[OK] MediaPipe Legacy API ready")
    except Exception as e2:
        print(f"[!!] MediaPipe failed completely: {e2}")

detector   = DrowsinessDetector()
monitoring = False
start_t    = None
cap        = None

def get_color(lv):
    return {"SAFE":(0,255,150),"WARNING":(0,180,255),"DANGER":(0,80,255),"CRITICAL":(0,0,255)}.get(lv,(0,255,150))

def run_detection(frame):
    h, w = frame.shape[:2]
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ear, mar, found = 0.30, 0.0, False
    try:
        if USE_TASKS:
            res = face_mesh.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            lms = res.face_landmarks
            if lms:
                found = True
                lm = lms[0]
                lp = [(int(lm[i].x*w),int(lm[i].y*h)) for i in LEFT_EYE]
                rp = [(int(lm[i].x*w),int(lm[i].y*h)) for i in RIGHT_EYE]
                mp2= [(int(lm[i].x*w),int(lm[i].y*h)) for i in MOUTH_OUTER]
                ear = (eye_aspect_ratio(lp)+eye_aspect_ratio(rp))/2
                mar = mouth_aspect_ratio(mp2)
                c = get_color(detector.alert_level)
                mc= (255,200,0) if mar>0.35 else (100,255,200)
                cv2.polylines(frame,[np.array(lp,np.int32)],True,c,1)
                cv2.polylines(frame,[np.array(rp,np.int32)],True,c,1)
                cv2.polylines(frame,[np.array(mp2[:4],np.int32)],True,mc,1)
                for lmk in lm: cv2.circle(frame,(int(lmk.x*w),int(lmk.y*h)),1,(20,50,50),-1)
        else:
            res = face_mesh.process(rgb)
            if res.multi_face_landmarks:
                found = True
                lm = res.multi_face_landmarks[0]
                def pts(idx): return [(int(lm.landmark[i].x*w),int(lm.landmark[i].y*h)) for i in idx]
                lp,rp,mp2 = pts(LEFT_EYE),pts(RIGHT_EYE),pts(MOUTH_OUTER)
                ear = (eye_aspect_ratio(lp)+eye_aspect_ratio(rp))/2
                mar = mouth_aspect_ratio(mp2)
                c = get_color(detector.alert_level)
                mc= (255,200,0) if mar>0.35 else (100,255,200)
                cv2.polylines(frame,[np.array(lp,np.int32)],True,c,1)
                cv2.polylines(frame,[np.array(rp,np.int32)],True,c,1)
                cv2.polylines(frame,[np.array(mp2[:4],np.int32)],True,mc,1)
                for lmk in lm.landmark: cv2.circle(frame,(int(lmk.x*w),int(lmk.y*h)),1,(20,50,50),-1)
    except Exception as e:
        print(f"[DETECT ERR] {e}")

    # HUD
    lv = detector.alert_level
    c  = get_color(lv)
    cv2.putText(frame,f"EAR:{ear:.2f}",(8,h-50),cv2.FONT_HERSHEY_SIMPLEX,0.5,c,1)
    cv2.putText(frame,f"MAR:{mar:.2f}",(8,h-30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(100,255,180),1)
    if not found:
        cv2.putText(frame,"NO FACE",(w//2-60,h//2),cv2.FONT_HERSHEY_SIMPLEX,1,(0,80,255),2)
    bg={"SAFE":(0,30,15),"WARNING":(30,20,0),"DANGER":(40,0,0),"CRITICAL":(60,0,0)}.get(lv,(0,30,15))
    cv2.rectangle(frame,(w-115,5),(w-5,27),bg,-1)
    cv2.putText(frame,f"[{lv}]",(w-112,21),cv2.FONT_HERSHEY_SIMPLEX,0.42,c,1)
    return ear, mar, found, frame

def camera_loop():
    global monitoring, start_t, cap
    print("[CAM] Starting camera...")
    for cam_idx in [0, 1, 2]:
        cap = cv2.VideoCapture(cam_idx)
        if cap.isOpened():
            print(f"[CAM] Opened camera index {cam_idx}")
            break
        cap.release()
    
    if not cap or not cap.isOpened():
        print("[CAM ERR] No camera found!")
        socketio.emit('cam_error', {'msg': 'No camera found! Check your webcam.'})
        monitoring = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    start_t = time.time()
    detector.reset()
    fc = 0

    while monitoring:
        ok, frame = cap.read()
        if not ok:
            print("[CAM] Read failed, retrying...")
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)
        ts    = time.time() - start_t
        ear, mar, found, frame = run_detection(frame)
        alerts = detector.update(ear, mar, ts)
        stats  = detector.get_stats()

        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        b64 = base64.b64encode(buf).decode()

        socketio.emit('frame_data', {
            'frame': b64, 'stats': stats,
            'alerts': alerts, 'face': found, 'ts': round(ts)
        })
        fc += 1
        if fc % 60 == 0:
            print(f"[CAM] {fc} frames | EAR={ear:.2f} MAR={mar:.2f} LEVEL={detector.alert_level}")
        time.sleep(0.033)

    cap.release()
    print("[CAM] Camera released.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ping')
def ping():
    return jsonify({'status': 'ok', 'monitoring': monitoring})

@app.route('/songs')
def list_songs():
    audio_dir = os.path.join(app.static_folder, 'audio')
    ALLOWED = {'.mp3','.mp4','.wav','.ogg','.m4a','.MP3','.MP4'}
    songs = []
    if os.path.exists(audio_dir):
        for f in sorted(os.listdir(audio_dir)):
            if os.path.splitext(f)[1] in ALLOWED:
                songs.append({'name': os.path.splitext(f)[0], 'url': '/static/audio/'+f})
    print(f"[SONGS] {len(songs)}: {[s['name'] for s in songs]}")
    return jsonify(songs)

@socketio.on('connect')
def on_connect():
    print(f"[WS] Client connected!")
    emit('status', {'monitoring': monitoring, 'message': 'Connected!'})

@socketio.on('disconnect')
def on_disconnect():
    print("[WS] Client disconnected")

@socketio.on('start_monitoring')
def on_start():
    global monitoring
    print("[WS] START received")
    if not monitoring:
        monitoring = True
        threading.Thread(target=camera_loop, daemon=True).start()
    emit('status', {'monitoring': True, 'message': 'Camera starting...'})

@socketio.on('stop_monitoring')
def on_stop():
    global monitoring
    monitoring = False
    print("[WS] STOP received")
    emit('status', {'monitoring': False, 'message': 'Stopped.'})

@socketio.on('update_sensitivity')
def on_sens(data):
    lv = data.get('level','medium')
    cfg = {'low':(0.20,20,0.45),'medium':(0.22,15,0.35),'high':(0.26,10,0.28)}
    e,f,m = cfg.get(lv,cfg['medium'])
    detector.EAR_THRESHOLD=e; detector.EYE_CLOSED_FRAMES=f; detector.MAR_THRESHOLD=m
    emit('sensitivity_updated',{'level':lv})

if __name__ == '__main__':
    print("""
╔══════════════════════════════════╗
║   DrowseGuard AI  — READY        ║
║   http://localhost:5000          ║
╚══════════════════════════════════╝
""")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)