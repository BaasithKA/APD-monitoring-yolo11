import os
import sqlite3
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, redirect, url_for, request
import cv2
from ultralytics import YOLO
import atexit
from collections import Counter
import threading
import time
import io
import csv
import torch
from torch.nn import Sequential, Conv2d, BatchNorm2d, SiLU
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C3k2
import math

# --- Konfigurasi Aplikasi ---
app = Flask(__name__)
DB_NAME = 'deteksi.db'
PROCESSED_FOLDER = 'static/processed'
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
RECORDS_PER_PAGE = 100

# --- Variabel Global & Peta Warna ---
WARNA_KELAS = { 'helmet': (0, 255, 0), 'mask': (255, 0, 0), 'vest': (0, 165, 255) }
WARNA_DEFAULT = (255, 255, 255)
latest_counts = {}
lock = threading.Lock()
last_save_time = 0
SAVE_COOLDOWN = 10

# --- Inisialisasi Model & Kamera ---
torch.serialization.add_safe_globals([DetectionModel, Sequential, Conv, Conv2d, BatchNorm2d, SiLU, C3k2])
model = YOLO("best.pt")
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError("❌ Tidak bisa membuka webcam!")

# --- Fungsi Bantuan & Cleanup ---
@atexit.register
def cleanup():
    print("Melepaskan sumber daya kamera...")
    camera.release()

def draw_boxes(frame, result):
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf, cls_id = round(float(box.conf[0]), 2), int(box.cls[0])
        cls_name = model.names[cls_id]
        box_color = WARNA_KELAS.get(cls_name.lower(), WARNA_DEFAULT)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 1)
        label = f"{cls_name} {conf}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(frame, (x1, y1 - h - 5), (x1 + w, y1), box_color, -1)
        cv2.putText(frame, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return frame

# --- Generator Frame untuk Live Feed ---
def generate_frames():
    global latest_counts, last_save_time
    while True:
        success, frame = camera.read()
        if not success: break
        
        results = model(frame, stream=True, imgsz=320, conf=0.5, verbose=False)
        annotated_frame = frame.copy()

        for result in results:
            detected_classes_ids = result.boxes.cls.tolist()
            class_names = [model.names[int(cls_id)] for cls_id in detected_classes_ids]
            counts = Counter(class_names)
            with lock: latest_counts = dict(counts)
            annotated_frame = draw_boxes(annotated_frame, result)
            
            current_time = time.time()
            if counts and (current_time - last_save_time) > SAVE_COOLDOWN:
                timestamp_dt = datetime.now()
                is_complete = counts.get('helmet', 0) > 0 and counts.get('mask', 0) > 0 and counts.get('vest', 0) > 0
                status = "Lengkap" if is_complete else "Tidak Lengkap"
                
                print(f"Deteksi '{status}' terdeteksi, menyimpan data...")
                
                frame_to_save = annotated_frame.copy()
                timestamp_text_save = timestamp_dt.strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame_to_save, timestamp_text_save, (10, frame_to_save.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

                filename = f"{status.replace(' ', '_')}_{timestamp_dt.strftime('%Y%m%d-%H%M%S')}.jpg"
                filepath_relative = f"processed/{filename}"
                filepath_full = os.path.join(PROCESSED_FOLDER, filename)
                cv2.imwrite(filepath_full, frame_to_save)
                last_save_time = current_time
                
                try:
                    conn = sqlite3.connect(DB_NAME)
                    cursor = conn.cursor()
                    detected_str = ", ".join([f"{k}: {v}" for k, v in counts.items()])
                    cursor.execute(
                        "INSERT INTO deteksi (timestamp, detected_objects, image_path, status) VALUES (?, ?, ?, ?)",
                        (timestamp_dt, detected_str, filepath_relative, status)
                    )
                    conn.commit()
                    conn.close()
                except Exception as e:
                    print(f"Gagal menyimpan ke database: {e}")

        timestamp_text_stream = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        cv2.putText(annotated_frame, timestamp_text_stream, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- Fungsi Bantuan untuk Filter ---
def build_filter_query_parts(args):
    search_query = args.get('search', '')
    status_filter = args.get('status', '')
    start_date = args.get('start_date', '')
    end_date = args.get('end_date', '')
    
    conditions, params = [], []
    if search_query:
        conditions.append("detected_objects LIKE ?")
        params.append(f"%{search_query}%")
    if status_filter:
        conditions.append("status = ?")
        params.append(status_filter)
    if start_date and start_date.strip():
        conditions.append("DATE(timestamp) >= ?")
        params.append(start_date)
    if end_date and end_date.strip():
        conditions.append("DATE(timestamp) <= ?")
        params.append(end_date)
    
    where_clause = ""
    if conditions:
        where_clause = " WHERE " + " AND ".join(conditions)
        
    return where_clause, params

# --- Rute-Rute Aplikasi Flask ---
@app.route('/')
def index(): 
    return render_template('index.html')

@app.route('/video')
def video(): 
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data')
def data():
    with lock: 
        return jsonify(latest_counts)

@app.route('/history')
def history():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    page = request.args.get('page', 1, type=int)
    offset = (page - 1) * RECORDS_PER_PAGE
    where_clause, params = build_filter_query_parts(request.args)
    total_records = cursor.execute(f"SELECT COUNT(*) FROM deteksi{where_clause}", params).fetchone()[0]
    total_pages = math.ceil(total_records / RECORDS_PER_PAGE) if total_records > 0 else 1
    query = f"SELECT * FROM deteksi{where_clause} ORDER BY timestamp DESC LIMIT ? OFFSET ?"
    cursor.execute(query, (*params, RECORDS_PER_PAGE, offset))
    detections = cursor.fetchall()
    stats = {}
    stats['total_records'] = total_records
    today_where = (" AND " if where_clause else " WHERE ") + "DATE(timestamp) = DATE('now', 'localtime')"
    stats['today_detections'] = cursor.execute(f"SELECT COUNT(*) FROM deteksi{where_clause}{today_where}", params).fetchone()[0]
    complete_where = (" AND " if where_clause else " WHERE ") + "status = 'Lengkap'"
    stats['complete_detections'] = cursor.execute(f"SELECT COUNT(*) FROM deteksi{where_clause}{complete_where}", params).fetchone()[0]
    incomplete_where = (" AND " if where_clause else " WHERE ") + "status = 'Tidak Lengkap'"
    stats['incomplete_detections'] = cursor.execute(f"SELECT COUNT(*) FROM deteksi{where_clause}{incomplete_where}", params).fetchone()[0]
    conn.close()
    return render_template('history.html', 
                           detections=detections, 
                           stats=stats,
                           search_query=request.args.get('search', ''),
                           status_filter=request.args.get('status', ''),
                           start_date=request.args.get('start_date', ''),
                           end_date=request.args.get('end_date', ''),
                           current_page=page,
                           total_pages=total_pages)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/export')
def export_csv():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    where_clause, params = build_filter_query_parts(request.args)
    query = f"SELECT id, timestamp, detected_objects, status, image_path FROM deteksi{where_clause} ORDER BY timestamp DESC"
    cursor.execute(query, params)
    data = cursor.fetchall()
    conn.close()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Timestamp', 'Detected Objects', 'Status', 'Image Path'])
    for row in data:
        writer.writerow(row)
    output.seek(0)
    return Response(output,
                    mimetype="text/csv",
                    headers={"Content-Disposition": "attachment;filename=detection_history.csv"})

# --- API Endpoints for Dashboard ---
@app.route('/api/dashboard_stats')
def dashboard_stats_api():
    where_clause, params = build_filter_query_parts(request.args)
    stats = {}
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    stats['total_records'] = cursor.execute(f"SELECT COUNT(*) FROM deteksi{where_clause}", params).fetchone()[0]
    today_where = (" AND " if where_clause else " WHERE ") + "DATE(timestamp) = DATE('now', 'localtime')"
    stats['today_detections'] = cursor.execute(f"SELECT COUNT(*) FROM deteksi{where_clause}{today_where}", params).fetchone()[0]
    complete_where = (" AND " if where_clause else " WHERE ") + "status = 'Lengkap'"
    stats['complete_detections'] = cursor.execute(f"SELECT COUNT(*) FROM deteksi{where_clause}{complete_where}", params).fetchone()[0]
    incomplete_where = (" AND " if where_clause else " WHERE ") + "status = 'Tidak Lengkap'"
    stats['incomplete_detections'] = cursor.execute(f"SELECT COUNT(*) FROM deteksi{where_clause}{incomplete_where}", params).fetchone()[0]
    conn.close()
    return jsonify(stats)

@app.route('/api/bar_chart_data')
def bar_chart_data_api():
    timespan = request.args.get('timespan', 'month')
    where_clause, params = build_filter_query_parts(request.args)
    
    # Menambahkan "AND" jika where_clause sudah ada, atau "WHERE" jika belum
    base_connector = " AND " if where_clause else " WHERE "
    
    query = ""
    if timespan == 'today': # Last 30 days
        query = f"SELECT DATE(timestamp) as label, COUNT(*) as count FROM deteksi{where_clause}{base_connector}DATE(timestamp) >= DATE('now', 'localtime', '-29 days') GROUP BY label ORDER BY label"
    elif timespan == 'week': # Weeks of current month
        query = f"SELECT 'Minggu ' || STRFTIME('%W', timestamp) as label, COUNT(*) as count FROM deteksi{where_clause}{base_connector}STRFTIME('%Y-%m', timestamp) = STRFTIME('%Y-%m', 'now', 'localtime') GROUP BY label ORDER BY label"
    else: # Default 'month', months of current year
        query = f"SELECT STRFTIME('%Y-%m', timestamp) as label, COUNT(*) as count FROM deteksi{where_clause}{base_connector}STRFTIME('%Y', timestamp) = STRFTIME('%Y', 'now', 'localtime') GROUP BY label ORDER BY label"
    
    labels, data = [], []
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row
        rows = conn.cursor().execute(query, params).fetchall()
        for row in rows:
            labels.append(row['label'])
            data.append(row['count'])
        conn.close()
    except Exception as e: 
        print(f"Error in bar_chart_data API: {e}")
    return jsonify({'labels': labels, 'data': data})

@app.route('/api/status_pie_chart_data')
def status_pie_chart_data_api():
    where_clause, params = build_filter_query_parts(request.args)
    query = f"SELECT status, COUNT(*) as count FROM deteksi{where_clause} GROUP BY status"
    labels, data = [], []
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row
        rows = conn.cursor().execute(query, params).fetchall()
        for row in rows:
            labels.append(row['status'])
            data.append(row['count'])
        conn.close()
    except Exception as e: print(f"Error API status_pie_chart: {e}")
    return jsonify({'labels': labels, 'data': data})

@app.route('/api/ppe_pie_chart_data')
def ppe_pie_chart_data_api():
    where_clause, params = build_filter_query_parts(request.args)
    query = f"SELECT detected_objects FROM deteksi{where_clause}"
    pie_counts = Counter()
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row
        rows = conn.cursor().execute(query, params).fetchall()
        for row in rows:
            parts = row['detected_objects'].split(', ')
            for part in parts: pie_counts[part.split(':')[0].strip()] += 1
        conn.close()
    except Exception as e: print(f"Error API ppe_pie_chart: {e}")
    return jsonify({'labels': list(pie_counts.keys()), 'data': list(pie_counts.values())})

@app.route('/api/line_chart_data')
def line_chart_data_api():
    where_clause, params = build_filter_query_parts(request.args)
    line_where = (" AND " if where_clause else " WHERE ") + "DATE(timestamp) = DATE('now', 'localtime')"
    
    hourly_data = [0] * 24
    query = f"SELECT STRFTIME('%H', timestamp) as hour, COUNT(*) as count FROM deteksi{where_clause}{line_where} GROUP BY hour"
    
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row
        rows = conn.cursor().execute(query, params).fetchall()
        for row in rows:
            hour_index = int(row['hour'])
            hourly_data[hour_index] = row['count']
        conn.close()
    except Exception as e: 
        print(f"Error in line_chart_data API: {e}")
    
    labels = [f"{h:02d}:00" for h in range(24)]
    return jsonify({'labels': labels, 'data': hourly_data})


@app.route('/delete/<int:id>', methods=['POST'])
def delete_detection(id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT image_path FROM deteksi WHERE id = ?", (id,))
    data_to_delete = cursor.fetchone()
    if data_to_delete:
        try:
            image_path = os.path.join('static', data_to_delete[0])
            if os.path.exists(image_path): os.remove(image_path)
        except Exception as e: 
            print(f"Failed to delete image file: {e}")
        cursor.execute("DELETE FROM deteksi WHERE id = ?", (id,))
        conn.commit()
    conn.close()
    return redirect(url_for('history'))

# --- App Entry Point ---
if __name__ == '__main__':
    if not os.path.exists(DB_NAME):
        print(f"WARNING: Database file '{DB_NAME}' not found. Please run 'python init_db.py' first.")
    app.run(debug=True, use_reloader=False, threaded=True)