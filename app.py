import os
import cv2
import sqlite3
import pickle
import threading
import time
import datetime as dt
from flask import Flask, render_template, jsonify, request, send_file, redirect, url_for, Response
import numpy as np
import pandas as pd
from io import BytesIO

app = Flask(__name__)
DB_PATH = os.path.join(os.path.dirname(__file__), 'attendance.db')
KNOWN_FACES_DIR = os.path.join(os.path.dirname(__file__), 'known_faces')

@app.context_processor
def inject_globals():
    """Make user_count available in all templates via the navbar."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM users')
        count = c.fetchone()[0]
        conn.close()
        return {'user_count': count}
    except Exception:
        return {'user_count': 0}


STOP_CAMERA = False
CAMERA_THREAD = None
ATTENDANCE_LOG = []
SCAN_INTERVAL = 2                # how often (seconds) we scan a frame for faces
ATTENDANCE_COOLDOWN = 1800       # 30 minutes — each student marked at most once per 30 min
LAST_MARKED = {}                 # {user_id: timestamp} — persists across scans
RAW_FRAME = None
FRAME_LOCK = threading.Lock()
CAP = None
LAST_SCAN_RESULT = {"results": [], "status": "System Standby", "timestamp": 0}

haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        reg_num TEXT UNIQUE NOT NULL,
        encoding BLOB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        name TEXT NOT NULL,
        reg_num TEXT,
        status TEXT DEFAULT 'Present',
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        image_path TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )''')
    # Migration: Ensure columns exist if table already existed
    try:
        c.execute('ALTER TABLE users ADD COLUMN reg_num TEXT')
    except sqlite3.OperationalError:
        pass
    
    try:
        c.execute('ALTER TABLE attendance ADD COLUMN reg_num TEXT')
    except sqlite3.OperationalError:
        pass

    try:
        c.execute('ALTER TABLE attendance ADD COLUMN image_path TEXT')
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()

def get_all_users():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, name, reg_num, encoding FROM users')
    users = c.fetchall()
    conn.close()
    return users

def extract_face_histogram(face_roi):
    """
    Advanced Spatial LBP (Local Binary Pattern).
    Divides the face into an 8x8 grid to capture spatial information 
    (where specific features are), making it much harder to confuse 
    different people.
    """
    face_roi = cv2.resize(face_roi, (128, 128))
    if len(face_roi.shape) == 3:
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    
    # Lighting normalization (CLAHE is better than equalizeHist for faces)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    face_roi = clahe.apply(face_roi)
    
    # Vectorized LBP calculation
    center = face_roi[1:-1, 1:-1].astype(np.int16)
    lbp = np.zeros((126, 126), dtype=np.uint8)
    shifts = [
        face_roi[0:-2, 0:-2], face_roi[0:-2, 1:-1], face_roi[0:-2, 2:],
        face_roi[1:-1, 2:],   face_roi[2:,   2:],   face_roi[2:,   1:-1],
        face_roi[2:,   0:-2], face_roi[1:-1, 0:-2],
    ]
    for k, neighbor in enumerate(shifts):
        lbp |= ((neighbor.astype(np.int16) >= center).astype(np.uint8) << k)
    
    # Spatial Grid: Divide into 8x8 blocks
    grid_size = 8
    h, w = lbp.shape
    bh, bw = h // grid_size, w // grid_size
    
    all_hists = []
    for i in range(grid_size):
        for j in range(grid_size):
            block = lbp[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
            # 32 bins per block provides high detail without being too slow
            hist = cv2.calcHist([block], [0], None, [32], [0, 256]).flatten()
            all_hists.append(hist)
            
    full_hist = np.concatenate(all_hists)
    full_hist = full_hist / (full_hist.sum() + 1e-7)
    return full_hist.astype(np.float32)

def compare_histograms(h1, h2):
    """
    Safely compare two histograms. 
    Ensures they are the same size and float32 type to avoid OpenCV crashes.
    """
    if h1 is None or h2 is None or h1.shape != h2.shape:
        return 0
    return cv2.compareHist(h1.astype(np.float32), h2.astype(np.float32), cv2.HISTCMP_CORREL)

def load_last_marked():
    """Populate LAST_MARKED and ATTENDANCE_LOG from the database on startup."""
    global LAST_MARKED, ATTENDANCE_LOG
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # 1. Populate Cooldowns
        c.execute('SELECT user_id, MAX(timestamp) FROM attendance GROUP BY user_id')
        rows = c.fetchall()
        for user_id, ts_str in rows:
            if user_id:
                # Convert SQLite timestamp string to unix epoch
                dt_obj = dt.datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
                LAST_MARKED[user_id] = dt_obj.timestamp()
        
        # 2. Populate Recent Log for UI
        c.execute('SELECT id, name, reg_num, status, timestamp, image_path FROM attendance ORDER BY timestamp DESC LIMIT 20')
        log_rows = c.fetchall()
        ATTENDANCE_LOG = [
            {
                'record_id': r[0], 
                'name': r[1], 
                'reg_num': r[2],
                'status': r[3], 
                'timestamp': r[4],
                'image_url': f"/static/attendance_pics/{r[5]}" if r[5] else None
            }
            for r in log_rows
        ]
        
        conn.close()
        print(f"Loaded cooldown state and {len(ATTENDANCE_LOG)} recent entries with images.")
    except Exception as e:
        print(f"Error loading startup data: {e}")

def load_known_faces():
    """Load face encodings from DB. Does NOT reset LAST_MARKED."""
    users = get_all_users()
    known_faces = []
    for u in users:
        uid, name, reg_num, enc = u
        try:
            hist = pickle.loads(enc)
            known_faces.append((uid, name, reg_num, hist))
        except:
            pass
    return known_faces

def mark_attendance(user_id, name, reg_num, face_roi, status='Present'):
    # Save the face photo
    timestamp_str = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{name}_{timestamp_str}.jpg"
    save_dir = os.path.join('static', 'attendance_pics')
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    cv2.imwrite(filepath, face_roi)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO attendance (user_id, name, reg_num, status, image_path) VALUES (?, ?, ?, ?, ?)', 
              (user_id, name, reg_num, status, filename))
    record_id = c.lastrowid
    conn.commit()
    conn.close()
    
    entry = {
        'record_id': record_id,
        'name': name, 
        'reg_num': reg_num,
        'status': status, 
        'timestamp': dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'image_url': f"/static/attendance_pics/{filename}"
    }
    ATTENDANCE_LOG.insert(0, entry)
    if len(ATTENDANCE_LOG) > 50:
        ATTENDANCE_LOG.pop()

def process_frame(frame):
    global LAST_MARKED
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))
    
    if len(faces) == 0:
        return []

    known_faces = load_known_faces()
    if not known_faces:
        return [{"name": "No users registered", "status": "Unknown"}]

    results = []
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_hist = extract_face_histogram(face_roi)
        best_score = 0
        best_id = None
        best_name = None
        best_reg = None
        
        for uid, name, reg_num, known_hist in known_faces:
            score = compare_histograms(face_hist, known_hist)
            if score > best_score and score > 0.75:
                best_score = score
                best_id = uid
                best_name = name
                best_reg = reg_num
        
        if best_id is not None:
            now = time.time()
            last_time = LAST_MARKED.get(best_id, 0)
            elapsed = now - last_time
            if elapsed >= ATTENDANCE_COOLDOWN:
                LAST_MARKED[best_id] = now
                face_roi_color = frame[y:y+h, x:x+w]
                mark_attendance(best_id, best_name, best_reg, face_roi_color)
                results.append({"name": best_name, "reg_num": best_reg, "status": "Present"})
            else:
                results.append({"name": best_name, "reg_num": best_reg, "status": "Already Marked"})
        else:
            results.append({"name": "Unknown", "status": "Please Register"})
            
    return results

def camera_loop():
    global RAW_FRAME, CAP, STOP_CAMERA
    if CAP is None or not CAP.isOpened():
        print("Searching for a working camera...")
        found = False
        for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
            for index in [0, 1, 2]:
                print(f"Testing index {index} with backend {backend}...", flush=True)
                CAP = cv2.VideoCapture(index, backend)
                if CAP.isOpened():
                    ret, test_frame = CAP.read()
                    if ret:
                        print(f"Success! Found working camera at index {index} with backend {backend}", flush=True)
                        found = True
                        break
                    else:
                        print(f"Camera opened but failed to read frame at index {index}", flush=True)
                        CAP.release()
                else:
                    print(f"Failed to open camera at index {index}", flush=True)
            if found: break
            
        if not found:
            print("CRITICAL: No working camera found. Using index 0 default.")
            CAP = cv2.VideoCapture(0)
            
        CAP.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        CAP.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    last_scan = time.time()
    while not STOP_CAMERA:
        if CAP is None or not CAP.isOpened():
            time.sleep(1)
            continue
            
        ret, frame = CAP.read()
        if not ret or frame is None or np.sum(frame) == 0:
            # print("Grab failed or black frame received")
            time.sleep(0.1)
            continue
        
        with FRAME_LOCK:
            RAW_FRAME = frame.copy()
            # print("Frame updated")
            
        if time.time() - last_scan >= SCAN_INTERVAL:
            last_scan = time.time()
            results = process_frame(frame)
            with FRAME_LOCK:
                LAST_SCAN_RESULT = {
                    "results": results, 
                    "status": "Scanning Complete" if results else "No Faces Detected",
                    "timestamp": time.time()
                }
            if results:
                summary = ", ".join([f"{r['name']}({r['status']})" for r in results])
                print(f"Recognized: {summary}")
        time.sleep(0.01)
    
    if CAP:
        CAP.release()
        CAP = None

def gen_frames():
    global RAW_FRAME
    while True:
        with FRAME_LOCK:
            if RAW_FRAME is None:
                # Create a black placeholder frame with text
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Waiting for Camera...", (150, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(1.0)
                continue
            frame = RAW_FRAME.copy()
        
        # Draw rectangles on the frame for the feed
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)  # ~30 FPS cap — prevents CPU spin

def start_camera():
    global CAMERA_THREAD, STOP_CAMERA
    if CAMERA_THREAD is None or not CAMERA_THREAD.is_alive():
        STOP_CAMERA = False
        CAMERA_THREAD = threading.Thread(target=camera_loop, daemon=True)
        CAMERA_THREAD.start()

def stop_camera():
    global STOP_CAMERA
    STOP_CAMERA = True

@app.route('/')
def index():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM users')
    user_count = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM attendance WHERE DATE(timestamp) = DATE('now')")
    today_count = c.fetchone()[0]
    conn.close()
    return render_template('index.html', user_count=user_count, today_count=today_count)

@app.route('/api/log')
def get_log():
    return jsonify(ATTENDANCE_LOG[:20])

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        reg_num = request.form.get('reg_num', '').strip()
        if not name or not reg_num:
            return render_template('register.html', error='Name and Register Number are required')
        if request.form.get('capture'):
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            # ── Multi-sample registration ──────────────────────────────────
            # Collect up to 8 good face histograms and average them.
            # A single frame can be blurry or poorly lit; averaging gives
            # a much more stable and accurate face encoding.
            histograms = []
            for _ in range(40):          # try for ~8 seconds
                with FRAME_LOCK:
                    frame = RAW_FRAME.copy() if RAW_FRAME is not None else None
                if frame is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = haar_cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60)
                    )
                    for (x, y, w, h) in faces:
                        face_roi = gray[y:y+h, x:x+w]
                        histograms.append(extract_face_histogram(face_roi))
                        break          # one face per frame is enough
                if len(histograms) >= 8:
                    break
                time.sleep(0.2)

            if histograms:
                # Average all samples → smoother, more representative encoding
                avg_hist = np.mean(histograms, axis=0)
                avg_hist = (avg_hist / (avg_hist.sum() + 1e-7)).astype(np.float32)
                enc_blob = pickle.dumps(avg_hist)
                try:
                    c.execute('INSERT INTO users (name, reg_num, encoding) VALUES (?, ?, ?)', (name, reg_num, enc_blob))
                    conn.commit()
                    conn.close()
                    samples = len(histograms)
                    return render_template('register.html',
                        success=f'User {name} ({reg_num}) registered successfully! ({samples} samples captured)')
                except sqlite3.IntegrityError:
                    conn.close()
                    return render_template('register.html', error='Register Number already registered')
            conn.close()
            return render_template('register.html', error='No face detected. Please look directly at the camera and try again.')
        return render_template('register.html')
    return render_template('register.html')

@app.route('/reports')
def reports():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, name, reg_num, status, timestamp, image_path FROM attendance ORDER BY timestamp DESC LIMIT 100')
    records = c.fetchall()
    conn.close()
    return render_template('reports.html', records=records)

@app.route('/api/export')
def export_excel():
    conn = sqlite3.connect(DB_PATH)
    # Filter for 'Present' status as requested
    query = "SELECT name, reg_num, status, timestamp, image_path FROM attendance WHERE status='Present' ORDER BY timestamp DESC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Add full URL for images for easier reference in Excel
    if not df.empty and 'image_path' in df.columns:
        df['photo_link'] = df['image_path'].apply(lambda x: f"{request.host_url}static/attendance_pics/{x}" if x else "")
        df = df.drop(columns=['image_path'])

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Present_Attendance')
    output.seek(0)
    return send_file(output, download_name='present_attendance.xlsx', mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

@app.route('/analytics')
def analytics():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT DATE(timestamp) as date, COUNT(*) as total,
        SUM(CASE WHEN status='Present' THEN 1 ELSE 0 END) as present,
        SUM(CASE WHEN status != 'Present' THEN 1 ELSE 0 END) as unknown
        FROM attendance GROUP BY DATE(timestamp) ORDER BY date DESC LIMIT 30''')
    daily_stats = c.fetchall()
    c.execute('''SELECT name, COUNT(*) as count FROM attendance
        WHERE status='Present' GROUP BY name ORDER BY count DESC LIMIT 10''')
    top_users = c.fetchall()
    conn.close()
    return render_template('analytics.html', daily_stats=daily_stats, top_users=top_users)

@app.route('/api/users_list')
def get_users_list():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, name, reg_num, created_at FROM users ORDER BY name')
    users = c.fetchall()
    conn.close()
    return render_template('users.html', users=users)

@app.route('/api/delete_user', methods=['POST'])
def delete_user():
    user_id = request.form.get('user_id')
    redirect_to = request.form.get('redirect', 'get_users_list')
    if user_id:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('DELETE FROM users WHERE id=?', (user_id,))
        c.execute('DELETE FROM attendance WHERE user_id=?', (user_id,))
        conn.commit()
        conn.close()
    return redirect(url_for(redirect_to))

@app.route('/api/delete_attendance', methods=['POST'])
def delete_attendance_record():
    record_id = request.form.get('record_id')
    if record_id:
        record_id = int(record_id)
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('DELETE FROM attendance WHERE id=?', (record_id,))
        conn.commit()
        conn.close()
        # Also remove from the in-memory log for immediate UI update
        global ATTENDANCE_LOG
        ATTENDANCE_LOG = [entry for entry in ATTENDANCE_LOG if entry.get('record_id') != record_id]
    return redirect(url_for('reports'))

@app.route('/api/clear_attendance', methods=['POST'])
def clear_attendance():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM attendance')
    conn.commit()
    conn.close()
    global ATTENDANCE_LOG
    ATTENDANCE_LOG = []
    return redirect(url_for('reports'))

@app.route('/api/scan_status')
def get_scan_status():
    with FRAME_LOCK:
        return jsonify(LAST_SCAN_RESULT)

@app.route('/api/start_camera', methods=['POST'])
def api_start_camera():
    start_camera()
    return jsonify({'status': 'started'})

@app.route('/api/stop_camera', methods=['POST'])
def api_stop_camera():
    stop_camera()
    return jsonify({'status': 'stopped'})

if __name__ == '__main__':
    init_db()
    load_last_marked()
    start_camera()
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)