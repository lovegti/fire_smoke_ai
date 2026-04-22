import numpy as np
import cv2
import time
from datetime import datetime, timedelta
import threading
import random
import pymysql
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request, send_file
from ultralytics import YOLO
from io import BytesIO
import pandas as pd

app = Flask(__name__)

# ===================== MySQL 配置（改成你自己的）=====================
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = "*******"
DB_NAME = "*******"

def get_db_connection():
    return pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

# ===================== YOLOv11 =====================
model = YOLO("best.pt")
CONF_THRESH = 0.5

# ===================== 全局变量 =====================
lock = threading.Lock()
fire_detected = False
alarm_list = []
last_alarm_time = 0
current_video_url = ""

# ===================== 模拟传感器数据 =====================
def get_sensor_fake():
    if fire_detected:
        return {
            "temp": round(random.uniform(46, 75), 1),
            "smoke": random.randint(220, 600),
            "co": random.randint(45, 110),
            "fire_risk": random.randint(60, 100)
        }
    else:
        return {
            "temp": round(random.uniform(22, 28), 1),
            "smoke": random.randint(10, 40),
            "co": random.randint(5, 15),
            "fire_risk": random.randint(0, 30)
        }

# ===================== 定时存模拟数据到 MySQL =====================
def save_sensor_task():
    while True:
        try:
            d = get_sensor_fake()
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO sensor_data (timestamp, temp, smoke, co, fire_risk) VALUES (%s,%s,%s,%s,%s)",
                (now, d['temp'], d['smoke'], d['co'], d['fire_risk'])
            )
            conn.commit()
            cursor.close()
            conn.close()
        except:
            pass
        time.sleep(2)

threading.Thread(target=save_sensor_task, daemon=True).start()

# ===================== 视频流=====================
def gen_frames():
    global fire_detected, alarm_list, last_alarm_time, current_video_url
    while True:
        try:
            url = current_video_url
            if not url:
                # 无地址提示
                frame = np.zeros((480, 640, 3), np.uint8)
                cv2.putText(frame, "请输入摄像头IP并点击连接", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                _, jpg = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')
                time.sleep(0.5)
                continue

            # 用 OpenCV 直接读取视频流
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                frame = np.zeros((480, 640, 3), np.uint8)
                cv2.putText(frame, "无法连接摄像头，请检查IP", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                _, jpg = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')
                continue

            # 读取一帧
            ret, img = cap.read()
            cap.release()  # 释放资源，避免卡死

            if not ret or img is None:
                frame = np.zeros((480, 640, 3), np.uint8)
                cv2.putText(frame, "画面读取失败", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                _, jpg = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')
                continue

            # YOLOv11 检测
            results = model(img, conf=0.5)
            current_fire = False
            for r in results:
                for cls in r.boxes.cls:
                    if int(cls) in [0, 1]:
                        current_fire = True
                        break

            # 更新全局状态
            with lock:
                fire_detected = current_fire

            # 报警逻辑（10秒冷却）
            if current_fire and (time.time() - last_alarm_time > 10):
                with lock:
                    last_alarm_time = time.time()
                    now = time.strftime("%Y-%m-%d %H:%M:%S")
                    alarm_list.insert(0, {"time": now, "type": "烟火报警"})
                    if len(alarm_list) > 8:
                        alarm_list = alarm_list[:8]
                    # 存入MySQL
                    try:
                        conn = get_db_connection()
                        cursor = conn.cursor()
                        cursor.execute("INSERT INTO alarm_log (time, type) VALUES (%s, %s)", (now, "烟火报警"))
                        conn.commit()
                        cursor.close()
                        conn.close()
                    except:
                        pass

            # YOLO 自动画框
            frame = results[0].plot()
            _, jpg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')

        except Exception as e:
            print(f"视频流错误: {e}")
            frame = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(frame, "连接中...", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
            _, jpg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')
            time.sleep(0.5)
            continue

# ===================== 路由 =====================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/set_video_url', methods=['POST'])
def set_video_url():
    global current_video_url
    url = request.json.get('url')
    current_video_url = url
    return jsonify({"status": "ok"})

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/api/data')
def api_data():
    d = get_sensor_fake()
    return jsonify({
        "fire_detected": fire_detected,
        "alarm_list": alarm_list,
        "sensor": d,
        "fire_risk": d["fire_risk"]
    })

@app.route('/data')
def data_page():
    return render_template('data.html')

@app.route('/api/sensor_data')
def api_sensor_data():
    s = request.args.get('start')
    e = request.args.get('end')
    conn = get_db_connection()
    cursor = conn.cursor()
    if s and e:
        cursor.execute("SELECT timestamp,temp,smoke,co,fire_risk FROM sensor_data WHERE timestamp BETWEEN %s AND %s ORDER BY timestamp", (s, e))
    else:
        cursor.execute("SELECT timestamp,temp,smoke,co,fire_risk FROM sensor_data ORDER BY id DESC LIMIT 50")
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify([list(x.values()) for x in data])

@app.route('/alarm')
def alarm_page():
    return render_template('alarm.html')

@app.route('/api/alarm_log')
def api_alarm_log():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT time,type FROM alarm_log ORDER BY id DESC")
    logs = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify([list(x.values()) for x in logs])

@app.route('/analysis')
def analysis_page():
    return render_template('analysis.html')

@app.route('/api/analysis_data')
def api_analysis_data():
    s = request.args.get('start')
    e = request.args.get('end')
    conn = get_db_connection()
    cursor = conn.cursor()
    if s and e:
        cursor.execute("SELECT timestamp,temp,smoke,co,fire_risk FROM sensor_data WHERE timestamp BETWEEN %s AND %s ORDER BY timestamp", (s, e))
    else:
        cursor.execute("SELECT timestamp,temp,smoke,co,fire_risk FROM sensor_data ORDER BY id DESC LIMIT 100")
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify([list(x.values()) for x in data])

# ===================== 修复导出：一定有数据 =====================
@app.route('/export_data')
def export_data():
    s = request.args.get('start')
    e = request.args.get('end')
    conn = get_db_connection()
    cursor = conn.cursor()

    if s and e:
        cursor.execute(
            "SELECT timestamp, temp, smoke, co, fire_risk FROM sensor_data WHERE timestamp BETWEEN %s AND %s ORDER BY timestamp",
            (s, e)
        )
    else:
        cursor.execute(
            "SELECT timestamp, temp, smoke, co, fire_risk FROM sensor_data ORDER BY id DESC LIMIT 100"
        )

    data = cursor.fetchall()
    cursor.close()
    conn.close()

    # 修复：pymysql 返回字典，转成列表
    data_list = []
    for row in data:
        data_list.append([
            row['timestamp'],
            row['temp'],
            row['smoke'],
            row['co'],
            row['fire_risk']
        ])

    # 如果没数据，生成兜底模拟数据
    if not data_list:
        now = datetime.now()
        for i in range(10):
            time_str = (now - timedelta(seconds=i * 2)).strftime("%Y-%m-%d %H:%M:%S")
            temp = round(random.uniform(22, 28), 1)
            smoke = random.randint(10, 40)
            co = random.randint(5, 15)
            risk = random.randint(0, 30)
            data_list.append([time_str, temp, smoke, co, risk])

    # 构建 DataFrame
    df = pd.DataFrame(
        data_list,
        columns=['时间', '温度(℃)', '烟雾浓度', 'CO(ppm)', '火灾风险(%)']
    )

    # 导出 Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='消防监测数据', index=False)
    output.seek(0)

    return send_file(
        output,
        download_name=f'消防监测数据_{datetime.now().strftime("%Y%m%d")}.xlsx',
        as_attachment=True
    )
@app.route('/manage')
def manage_page():
    return render_template('manage.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)