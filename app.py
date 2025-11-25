from flask import Flask, render_template, Response, jsonify, request, send_file
import cv2
import face_recognition
import pickle
import numpy as np
from datetime import datetime
import pandas as pd
import os
import base64
import time
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from flask import session
from functools import wraps
import secrets

def get_runtime_dir():
    """回傳用於儲存可變資料的目錄。"""
    if getattr(sys, 'frozen', False):  # 在 PyInstaller 裡面去執行
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def get_resource_dir():
    """回傳包含綁定資源（templates/static）的目錄。"""
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


RUNTIME_DIR = get_runtime_dir()
RESOURCE_DIR = get_resource_dir()

# 確保相對路徑（CSV、編碼等）解析到專案根目錄
os.chdir(RUNTIME_DIR)

def data_path(*parts: str) -> Path:
    """建立相對於執行時目錄的絕對路徑。"""
    return RUNTIME_DIR.joinpath(*parts)


def resource_path(*parts: str) -> Path:
    """建立相對於 resource 目錄的絕對路徑。"""
    return RESOURCE_DIR.joinpath(*parts)


CONFIG_PATH = data_path('config.json')
ENCODINGS_DIR = data_path('encodings')
STUDENT_INFO_PATH = ENCODINGS_DIR / 'student_info.json'
FACE_ENCODINGS_PATH = ENCODINGS_DIR / 'face_encodings.pkl'

app = Flask(
    __name__,
    template_folder=str(resource_path('templates')),
    static_folder=str(resource_path('static'))
)
app.secret_key = secrets.token_hex(32)  # session 加密

# 自定義 Jinja2 過濾器
@app.template_filter('date_to_weekday')
def date_to_weekday_filter(date_str):
    """將日期字串轉為星期幾的索引 (0=週日, 6=週六)"""
    try:
        from datetime import datetime
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        weekday = date_obj.weekday()
        return (weekday + 1) % 7
    except:
        return 0

def draw_chinese_text(img, text, position, font_size=24, color=(255, 255, 255)):
    """
    在圖片上繪製中文文字
    :param img: OpenCV 圖片 (numpy array)
    :param text: 要繪製的文字
    :param position: 文字位置 (x, y)
    :param font_size: 字體大小
    :param color: 文字顏色 (B, G, R) for OpenCV
    :return: 繪製後的圖片
    """
    # 將 OpenCV 圖片 (BGR) 轉換為 PIL 圖片 (RGB)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font_paths = [
            'C:/Windows/Fonts/msyh.ttc',      # 微軟雅黑
            'C:/Windows/Fonts/simhei.ttf',    # 黑體
            'C:/Windows/Fonts/simsun.ttc',    # 宋體
            'C:/Windows/Fonts/msgothic.ttc',  # MS Gothic
        ]
        
        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                break
        
        if font is None:
            font = ImageFont.load_default()
    except Exception as e:
        print(f"字體載入失敗：{e}，使用默認字體")
        font = ImageFont.load_default()
    
    # 轉換 OpenCV 顏色 (BGR) 為 PIL 顏色 (RGB)
    color_rgb = (color[2], color[1], color[0])
    
    # 繪製文字
    draw.text(position, text, font=font, fill=color_rgb)
    
    # 轉換回 OpenCV 格式 (RGB -> BGR)
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    return img_cv

import json

# 設定管理頁面
def load_config():
    """載入系統設定"""
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # 回傳預設設定
        return {
            "system": {
                "system_name": "NPTU 國立屏東大學 AI 人臉辨識點名系統",
                "version": "3.4.0",
                "language": "zh-TW"
            },
            "recognition": {
                "tolerance": 0.45,
                "display_confidence_threshold": 60,
                "attendance_confidence_threshold": 65,
                "process_every_n_frames": 60
            },
            "video": {
                "jpeg_quality": 85,
                "camera_index": 0
            },
            "admin": {
                "username": "admin",
                "password": "admin123",
                "session_timeout": 3600
            },
            "attendance": {
                "auto_record": True,
                "record_format": "csv",
                "backup_enabled": False
            },
            "course": {
                "course_name": "計算機概論",
                "current_session": 1,
                "total_sessions": 3,
                "instructor": "林義凱教授"
            }
        }

def save_config(config):
    """儲存系統設定"""
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

# 載入設定
system_config = load_config()

def load_student_info():
    """載入學生資訊"""
    try:
        with open(STUDENT_INFO_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_student_info(student_info):
    """儲存學生資訊"""
    os.makedirs(ENCODINGS_DIR, exist_ok=True)
    with open(STUDENT_INFO_PATH, 'w', encoding='utf-8') as f:
        json.dump(student_info, f, ensure_ascii=False, indent=2)

def add_student_info(name, student_id):
    """新增學生資訊"""
    student_info = load_student_info()
    student_info[name] = {
        'student_id': student_id,
        'registered_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    save_student_info(student_info)
    return True

def get_student_id(name):
    """根據姓名獲取學號"""
    student_info = load_student_info()
    if name in student_info:
        return student_info[name].get('student_id', '')
    return ''

# 載入訓練好的臉部編碼
def load_encodings():
    """載入臉部編碼"""
    try:
        with open(FACE_ENCODINGS_PATH, 'rb') as f:
            data = pickle.load(f)
            return data['encodings'], data['names']
    except FileNotFoundError:
        print("訓練資料不存在，請先註冊學生")
        return [], []

known_encodings, known_names = load_encodings()

# 載入學生資訊（姓名 -> 學號的 key-value mapping）
student_info_dict = load_student_info()

# 今日出席記錄
attendance_today = set()

def initialize_attendance_today():
    """系統啟動時，從 CSV 載入今天的出席記錄到 attendance_today"""
    global attendance_today
    today = datetime.now().strftime("%Y%m%d")
    filename = f"attendance_records/attendance_{today}.csv"
    
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename, encoding='utf-8-sig')
            config = load_config()
            course_info = config.get('course', {})
            single_attendance = course_info.get('single_attendance', False)
            
            # 獲取當前課程、教師和節次資訊
            current_course = course_info.get('course_name', '')
            current_instructor = course_info.get('instructor', '')
            current_session = course_info.get('current_session', 1)
            total_sessions = course_info.get('total_sessions', 1)
            current_session_str = f"{current_session}/{total_sessions}"
            
            if single_attendance:
                # 單次點名模式：只載入當前課程和教師的記錄（不區分節次）
                if 'Course' in df.columns and 'Instructor' in df.columns:
                    # 篩選當前課程和教師的記錄
                    df_filtered = df[
                        (df['Course'] == current_course) & 
                        (df['Instructor'] == current_instructor)
                    ]
                    attended_names = df_filtered['Name'].unique().tolist()
                else:
                    # 舊格式
                    attended_names = df['Name'].unique().tolist()
            else:
                # 非單次點名模式：只載入當前課程、教師和節次的記錄
                if 'Course' in df.columns and 'Instructor' in df.columns and 'Session' in df.columns:
                    # 篩選當前課程、教師和節次的記錄
                    df_filtered = df[
                        (df['Course'] == current_course) & 
                        (df['Instructor'] == current_instructor) &
                        (df['Session'] == current_session_str)
                    ]
                    attended_names = df_filtered['Name'].unique().tolist()
                else:
                    # 舊格式或沒有節次資訊，載入所有記錄（向後兼容）
                    attended_names = df['Name'].unique().tolist()
            
            attendance_today.update(attended_names)
            print(f"系統啟動：從 CSV 載入 {len(attended_names)} 位已出席學生")
            print(f"已出席名單: {attended_names}")
        except Exception as e:
            print(f"載入今日出席記錄失敗：{e}")
    else:
        print(f"系統啟動：今日尚無出席記錄檔案")

# 初始化今日出席記錄
initialize_attendance_today()

# 註冊過程用
registration_active = False

# 鏡頭管理
import threading
camera_lock = threading.Lock()
active_camera = None
video_stream_active = False

class VideoCamera:
    def __init__(self):
        global active_camera
        # 效能優化：幀計數器和快取
        self.frame_count = 0
        self.process_every_n_frames = 60  # 每60幀處理一次人臉識別
        self.last_face_locations = []
        self.last_face_names = []
        
        with camera_lock:
            # 如果已有啟用的鏡頭先釋放
            if active_camera is not None:
                try:
                    active_camera.release()
                except:
                    pass
            
            # 用超時機制開啟鏡頭
            self.video = cv2.VideoCapture(0)
            if not self.video.isOpened():
                # 嘗試釋放重開
                self.video.release()
                time.sleep(0.5)
                self.video = cv2.VideoCapture(0)
            
            active_camera = self
        
    def __del__(self):
        self.release()
    
    def release(self):
        """手動釋放鏡頭"""
        with camera_lock:
            if self.video is not None:
                try:
                    self.video.release()
                except:
                    pass
                self.video = None
    
    def get_frame(self):
        """擷取影格執行人臉識別"""
        global known_encodings, known_names, system_config
        
        if self.video is None or not self.video.isOpened():
            return None, None
            
        success, frame = self.video.read()
        if not success:
            return None, None
        
        # 讀取設定中的信心度閾值
        display_threshold = system_config.get('recognition', {}).get('display_confidence_threshold', 55)
        attendance_threshold = system_config.get('recognition', {}).get('attendance_confidence_threshold', 60)
        
        self.frame_count += 1
        recognized_names = []
        
        # 效能優化：每 N 幀才做一次人臉識別
        if self.frame_count % self.process_every_n_frames == 0:
            # 調整影格大小以提升處理速度
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # 偵測人臉位置和編碼
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            # 儲存結果以供快取使用
            self.last_face_locations = []
            self.last_face_names = []
            
            # 辨識每張臉
            for face_encoding, face_location in zip(face_encodings, face_locations):
                name = "Unknown"
                confidence = 0
                
                # 與已知臉部進行比對
                if len(known_encodings) > 0:
                    # 更嚴格的 tolerance 避免誤判
                    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.45)
                    
                    # 計算距離
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                    
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        best_distance = face_distances[best_match_index]
                        confidence = (1 - best_distance) * 100
                        
                        # 雙重檢查要同時滿足
                        # 1. matches[best_match_index] == True (在 tolerance 範圍內)
                        # 2. confidence >= display_threshold (信心度足夠高，可以顯示)
                        if matches[best_match_index] and confidence >= display_threshold:
                            name = known_names[best_match_index]
                            
                            # Record attendance (需要 >= attendance_threshold 才記錄出席)
                            if name not in attendance_today and confidence >= attendance_threshold:
                                print(f"記錄出席：{name}，信心度：{confidence:.2f}%（閾值：{attendance_threshold}%）")
                                mark_attendance(name, confidence)
                                recognized_names.append(name)
                            elif name in attendance_today:
                                # 學生已在出席名單中
                                pass
                            elif confidence < attendance_threshold:
                                print(f"{name} 信心度 {confidence:.2f}% 低於閾值 {attendance_threshold}%，不記錄出席")
                        else:
                            # 信心度不足或不匹配，標記為 Unknown
                            name = "Unknown"
                            confidence = 0
                
                # 儲存結果供快取使用
                self.last_face_locations.append((face_location, name, confidence))
                self.last_face_names.append(name)
        
        # 繪製識別結果（使用快取的結果）
        for face_location, name, confidence in self.last_face_locations:
            # 縮放回原始大小
            top, right, bottom, left = face_location
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # 繪製方框和姓名
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # 顯示姓名和學號
            student_id = get_student_id(name) if name != "Unknown" else ""
            if student_id:
                # 擴大標籤區域以容納兩行文字
                cv2.rectangle(frame, (left, bottom - 60), (right, bottom), color, cv2.FILLED)
                text_line1 = f"{name}"
                text_line2 = f"{student_id} ({confidence:.1f}%)"
            else:
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                text_line1 = f"{name} ({confidence:.1f}%)"
                text_line2 = None
            
            # 效能優化
            # 只有中文名字才使用 PIL
            # 英文用 cv2.putText
            if any('\u4e00' <= char <= '\u9fff' for char in name):  # 檢查是否包含中文字元
                # PIL 繪製中文
                if text_line2:
                    frame = draw_chinese_text(frame, text_line1, (left + 6, bottom - 52), font_size=18, color=(255, 255, 255))
                    frame = draw_chinese_text(frame, text_line2, (left + 6, bottom - 28), font_size=16, color=(255, 255, 255))
                else:
                    frame = draw_chinese_text(frame, text_line1, (left + 6, bottom - 28), font_size=18, color=(255, 255, 255))
            else:
                # OpenCV 繪製英文
                if text_line2:
                    cv2.putText(frame, text_line1, (left + 6, bottom - 38), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, text_line2, (left + 6, bottom - 12), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
                else:
                    cv2.putText(frame, text_line1, (left + 6, bottom - 6), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        return frame, recognized_names

def mark_attendance(name, confidence):
    """記錄出席到 CSV 檔案"""
    # 載入課程資訊
    config = load_config()
    course_info = config.get('course', {})
    course_name = course_info.get('course_name', '未設定課程')
    current_session = course_info.get('current_session', 1)
    total_sessions = course_info.get('total_sessions', 1)
    instructor = course_info.get('instructor', '未設定教師')
    single_attendance = course_info.get('single_attendance', False)
    
    # 建立今日的出席檔案
    today = datetime.now().strftime("%Y%m%d")
    filename = f"attendance_records/attendance_{today}.csv"
    
    # 檢查學生是否已簽到過（根據模式決定檢查範圍）
    if os.path.exists(filename):
        try:
            existing_df = pd.read_csv(filename, encoding='utf-8-sig')
            session_str = f"{current_session}/{total_sessions}"
            
            if single_attendance:
                # 單次點名模式：檢查是否已在當前課程和教師下簽到過（不限節次）
                if 'Course' in existing_df.columns and 'Instructor' in existing_df.columns:
                    matching_records = existing_df[
                        (existing_df['Name'] == name) & 
                        (existing_df['Course'] == course_name) & 
                        (existing_df['Instructor'] == instructor)
                    ]
                    
                    if not matching_records.empty:
                        print(f"{name} 已在今日的【{course_name} - {instructor}】課程簽到過（單次點名模式），跳過記錄")
                        attendance_today.add(name)
                        return
                else:
                    # 舊格式（沒有課程資訊），只檢查姓名
                    if name in existing_df['Name'].values:
                        print(f"{name} 已在今日簽到過（單次點名模式，舊格式），跳過記錄")
                        attendance_today.add(name)
                        return
            else:
                # 非單次點名模式：檢查是否已在當前課程、教師和節次下簽到過
                if 'Course' in existing_df.columns and 'Instructor' in existing_df.columns and 'Session' in existing_df.columns:
                    matching_records = existing_df[
                        (existing_df['Name'] == name) & 
                        (existing_df['Course'] == course_name) & 
                        (existing_df['Instructor'] == instructor) &
                        (existing_df['Session'] == session_str)
                    ]
                    
                    if not matching_records.empty:
                        print(f"{name} 已在今日的【{course_name} - {instructor} - 第 {current_session} 節】簽到過，跳過記錄")
                        attendance_today.add(name)
                        return
                else:
                    # 舊格式，只檢查姓名（向後兼容）
                    if name in existing_df['Name'].values:
                        print(f"{name} 已在今日簽到過（舊格式），跳過記錄")
                        attendance_today.add(name)
                        return
        except Exception as e:
            print(f"讀取出席記錄失敗: {e}")
    
    # 加入今日出席名單
    attendance_today.add(name)
    
    # 準備記錄資料
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record = {
        'Name': name,
        'Time': timestamp,
        'Course': course_name,
        'Session': f"{current_session}/{total_sessions}",
        'Instructor': instructor,
        'Confidence': f"{confidence:.2f}%",
        'Status': 'Present',
        'SingleAttendanceMode': 'Yes' if single_attendance else 'No'  # 標記是否為單次點名模式
    }
    
    # 寫入 CSV
    df = pd.DataFrame([record])
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False, encoding='utf-8-sig')
    else:
        df.to_csv(filename, mode='w', header=True, index=False, encoding='utf-8-sig')
    
    mode_text = "（單次點名模式）" if single_attendance else ""
    print(f"{name} 出席記錄成功 {mode_text}(課程: {course_name}, 節次: {current_session}/{total_sessions}, 信心度: {confidence:.2f}%)")

def generate_frames():
    """產生影像串流的影格"""
    global video_stream_active, known_encodings, known_names
    camera = None
    
    print(f"generate_frames() 開始 - video_stream_active: {video_stream_active}")
    print(f"當前已註冊: {len(set(known_names))} 個學生, {len(known_encodings)} 個編碼")
    if len(known_names) > 0:
        print(f"學生名單: {sorted(set(known_names))}")
    
    try:
        camera = VideoCamera()
        print(f"鏡頭已初始化")
        
        frame_count = 0
        while video_stream_active:
            frame, names = camera.get_frame()
            if frame is None:
                print(f"無法獲取影格，停止串流")
                break
            
            frame_count += 1
            if frame_count % 30 == 0:  # 每30幀打印一次
                print(f"影像串流執行中... (已處理 {frame_count} 幀)")
                
            # 效能優化：降低 JPEG 品質提升編碼速度
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            ret, buffer = cv2.imencode('.jpg', frame, encode_param)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except GeneratorExit:
        print(f"客戶端斷開連接")
        pass
    except Exception as e:
        print(f"影像串流錯誤: {e}")
    finally:
        if camera:
            camera.release()
            print(f"鏡頭已釋放")

# 管理員帳號密碼（從設定讀取）
def get_admin_credentials():
    """獲取管理員帳號密碼"""
    config = load_config()
    return config['admin']['username'], config['admin']['password']

def login_required(f):
    """要求登入才能存取"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return jsonify({'success': False, 'message': '請先登入'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    """管理員登入頁面"""
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        ADMIN_USERNAME, ADMIN_PASSWORD = get_admin_credentials()
        
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['logged_in'] = True
            session['username'] = username
            print(f"管理員登入成功: {username}")
            return jsonify({'success': True, 'message': '登入成功'})
        else:
            print(f"登入失敗: 帳號或密碼錯誤")
            return jsonify({'success': False, 'message': '帳號或密碼錯誤'})
    
    return render_template('admin_login.html')

@app.route('/admin_dashboard')
@login_required
def admin_dashboard():
    """管理頁面（需要登入）"""
    # 獲取系統統計資訊
    total_students = len(set(known_names))
    total_encodings = len(known_encodings)
    present_today = len(attendance_today)
    student_list = sorted(set(known_names))
    
    # 載入課程資訊
    config = load_config()
    course_info = config.get('course', {
        'course_name': '計算機概論',
        'current_session': 1,
        'total_sessions': 3,
        'instructor': '林義凱教授'
    })
    
    return render_template('admin_dashboard.html', 
                          username=session.get('username'),
                          total_students=total_students,
                          total_encodings=total_encodings,
                          present_today=present_today,
                          student_list=student_list,
                          course_info=course_info)

@app.route('/admin_dashboard/update_course', methods=['POST'])
@login_required
def update_course_info():
    """更新課程資訊"""
    try:
        data = request.get_json()
        
        # 載入當前設定
        config = load_config()
        
        # 確保 course 區塊存在
        if 'course' not in config:
            config['course'] = {}
        
        # 記錄舊的課程資訊
        old_session = config['course'].get('current_session', 0)
        old_course_name = config['course'].get('course_name', '')
        old_instructor = config['course'].get('instructor', '')
        
        # 更新課程資訊
        config['course']['course_name'] = data.get('course_name', '').strip()
        config['course']['current_session'] = int(data.get('current_session', 1))
        config['course']['total_sessions'] = int(data.get('total_sessions', 3))
        config['course']['instructor'] = data.get('instructor', '').strip()
        config['course']['single_attendance'] = data.get('single_attendance', False)
        
        # 驗證
        if not config['course']['course_name']:
            return jsonify({'success': False, 'message': '課程名稱不能為空'})
        
        if not config['course']['instructor']:
            return jsonify({'success': False, 'message': '任課教師不能為空'})
        
        if config['course']['current_session'] < 1:
            return jsonify({'success': False, 'message': '目前節次必須大於 0'})
        
        if config['course']['total_sessions'] < 1:
            return jsonify({'success': False, 'message': '總節次必須大於 0'})
        
        if config['course']['current_session'] > config['course']['total_sessions']:
            return jsonify({'success': False, 'message': '目前節次不能大於總節次'})
        
        # 檢查課程資訊是否改變
        new_session = config['course']['current_session']
        new_course_name = config['course']['course_name']
        new_instructor = config['course']['instructor']
        
        session_changed = (old_session != new_session)
        course_name_changed = (old_course_name != new_course_name)
        instructor_changed = (old_instructor != new_instructor)
        
        # 儲存設定
        save_config(config)
        
        # 重新載入全局設定
        global system_config, attendance_today
        system_config = config
        
        # 如果節次、課程名稱或教師改變，重新載入對應課程的出席名單
        should_reload_attendance = session_changed or course_name_changed or instructor_changed
        
        if should_reload_attendance:
            change_reasons = []
            if session_changed:
                change_reasons.append(f"節次變更（第 {old_session} 節 → 第 {new_session} 節）")
            if course_name_changed:
                change_reasons.append(f"課程變更（{old_course_name} → {new_course_name}）")
            if instructor_changed:
                change_reasons.append(f"教師變更（{old_instructor} → {new_instructor}）")
            
            print(f"{', '.join(change_reasons)}，重新載入出席名單")
            
            # 重新載入當前課程的出席記錄
            attendance_today.clear()
            initialize_attendance_today()
        
        print(f"課程資訊已更新: {config['course']['course_name']} - 第 {config['course']['current_session']} 節")
        
        message = '課程資訊已更新！'
        if should_reload_attendance:
            message += '課程資訊已變更，出席名單已重新載入。'
        else:
            message += '主頁會即時顯示最新資訊。'
        
        return jsonify({'success': True, 'message': message})
        
    except Exception as e:
        print(f"更新課程資訊失敗: {e}")
        return jsonify({'success': False, 'message': f'更新失敗: {str(e)}'})

@app.route('/admin_logout', methods=['POST'])
def admin_logout():
    """管理員登出"""
    username = session.get('username', 'Unknown')
    session.pop('logged_in', None)
    session.pop('username', None)
    print(f"管理員登出: {username}")
    return jsonify({'success': True, 'message': '已登出'})

@app.route('/admin_attendance')
@login_required
def admin_attendance():
    """查看出席記錄頁面"""
    import glob
    
    # 載入課程設定
    config = load_config()
    course_info = config.get('course', {})
    single_attendance = course_info.get('single_attendance', False)
    
    # 獲取所有出席記錄文件
    attendance_files = sorted(glob.glob('attendance_records/attendance_*.csv'), reverse=True)
    
    # 準備記錄數據
    records_by_date = []
    
    for file_path in attendance_files:
        try:
            # 從檔名提取日期
            filename = os.path.basename(file_path)
            date_str = filename.replace('attendance_', '').replace('.csv', '')
            # 格式化日期 YYYYMMDD -> YYYY-MM-DD
            formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            
            # 讀取 CSV 檔案
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            
            # 新增學號資訊
            df['StudentID'] = df['Name'].apply(lambda name: get_student_id(name))
            
            # 確保 SingleAttendanceMode 欄位存在（處理舊記錄）
            if 'SingleAttendanceMode' not in df.columns:
                df['SingleAttendanceMode'] = 'No'
            
            # 計算原始不重複學生數量（展開前）
            unique_students = df['Name'].nunique() if not df.empty else 0
            
            # 處理單次點名模式：只展開標記為單次點名模式的記錄
            if not df.empty and 'Session' in df.columns:
                expanded_records = []
                for _, row in df.iterrows():
                    # 檢查該記錄是否標記為單次點名模式
                    is_single_attendance = False
                    if pd.notna(row.get('SingleAttendanceMode')) and str(row['SingleAttendanceMode']).strip().upper() == 'YES':
                        is_single_attendance = True
                    
                    # 只展開標記為單次點名模式的記錄
                    if is_single_attendance and pd.notna(row.get('Session')) and '/' in str(row['Session']):
                        try:
                            session_parts = str(row['Session']).split('/')
                            if len(session_parts) == 2:
                                total = int(session_parts[1])
                                # 為每個節次生成記錄
                                for session_num in range(1, total + 1):
                                    new_row = row.copy()
                                    new_row['Session'] = f"{session_num}/{total}"
                                    expanded_records.append(new_row)
                            else:
                                expanded_records.append(row)
                        except:
                            expanded_records.append(row)
                    else:
                        # 非單次點名模式或舊記錄，保留原記錄
                        expanded_records.append(row)
                
                df = pd.DataFrame(expanded_records)
            
            records = df.to_dict('records')
            
            records_by_date.append({
                'date': formatted_date,
                'date_display': formatted_date,
                'total_count': unique_students,
                'records': records
            })
            
        except Exception as e:
            print(f"讀取出席記錄失敗 {file_path}: {e}")
            continue
    
    username = session.get('username', 'Admin')
    
    return render_template('admin_attendance.html', 
                          username=username,
                          records_by_date=records_by_date,
                          total_dates=len(records_by_date))

@app.route('/admin_attendance/clear_today', methods=['POST'])
@login_required
def clear_today_attendance():
    """清除今日出席記錄"""
    global attendance_today
    try:
        today = datetime.now().strftime("%Y%m%d")
        filename = f"attendance_records/attendance_{today}.csv"
        
        if os.path.exists(filename):
            # 刪除 CSV 檔案
            os.remove(filename)
            print(f"已刪除今日出席記錄檔案: {filename}")
        
        # 清空記憶體中的出席記錄
        attendance_today.clear()
        print(f"已清空記憶體中的出席記錄")
        
        return jsonify({
            'success': True,
            'message': '今日出席記錄已清除'
        })
    except Exception as e:
        print(f"清除出席記錄失敗: {e}")
        return jsonify({
            'success': False,
            'message': f'清除失敗: {str(e)}'
        })

@app.route('/admin_students')
@login_required
def admin_students():
    """學生管理頁面"""
    global known_names, known_encodings
    
    # 載入學生資訊
    student_info = load_student_info()
    
    # 準備學生列表
    students = []
    for name in set(known_names):
        info = student_info.get(name, {})
        student_id = info.get('student_id', 'N/A')
        registered_date = info.get('registered_date', 'Unknown')
        
        # 計算該學生的編碼數量
        encoding_count = known_names.count(name)
        
        students.append({
            'name': name,
            'student_id': student_id,
            'registered_date': registered_date,
            'encoding_count': encoding_count
        })
    
    # 按姓名排序
    students.sort(key=lambda x: x['name'])
    
    username = session.get('username', 'Admin')
    
    return render_template('admin_students.html',
                          username=username,
                          students=students,
                          total_students=len(students))

@app.route('/admin_students/edit/<name>', methods=['GET', 'POST'])
@login_required
def edit_student(name):
    """編輯學生資訊"""
    if request.method == 'POST':
        try:
            data = request.get_json()
            new_name = data.get('name', '').strip()
            new_student_id = data.get('student_id', '').strip()
            
            if not new_name or not new_student_id:
                return jsonify({'success': False, 'message': '姓名和學號不能為空'})
            
            # 載入學生資訊
            student_info = load_student_info()
            
            if new_name != name:
                # 檢查新名字是否已存在
                if new_name in student_info and new_name != name:
                    return jsonify({'success': False, 'message': '該姓名已存在'})
                
                # 更新學生資訊
                if name in student_info:
                    student_info[new_name] = student_info.pop(name)
                    student_info[new_name]['student_id'] = new_student_id
                else:
                    student_info[new_name] = {
                        'student_id': new_student_id,
                        'registered_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                
                global known_names
                known_names = [new_name if n == name else n for n in known_names]
                
            else:
                # 只更新學號
                if name in student_info:
                    student_info[name]['student_id'] = new_student_id
                else:
                    student_info[name] = {
                        'student_id': new_student_id,
                        'registered_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
            
            # 儲存更新
            save_student_info(student_info)
            
            print(f"學生資訊已更新: {name} -> {new_name} ({new_student_id})")
            
            return jsonify({'success': True, 'message': '學生資訊已更新'})
            
        except Exception as e:
            print(f"更新學生資訊失敗: {e}")
            return jsonify({'success': False, 'message': f'更新失敗: {str(e)}'})
    
    # GET 請求：回傳學生資訊
    student_info = load_student_info()
    info = student_info.get(name, {})
    
    return jsonify({
        'name': name,
        'student_id': info.get('student_id', ''),
        'registered_date': info.get('registered_date', '')
    })

@app.route('/admin_students/delete/<name>', methods=['POST'])
@login_required
def delete_student(name):
    """刪除學生"""
    try:
        global known_names, known_encodings
        
        # 從學生資訊中刪除
        student_info = load_student_info()
        if name in student_info:
            del student_info[name]
            save_student_info(student_info)
        
        # 從編碼中刪除
        indices_to_remove = [i for i, n in enumerate(known_names) if n == name]
        
        # 反向刪除以免 index 問題
        for index in sorted(indices_to_remove, reverse=True):
            del known_names[index]
            del known_encodings[index]
        
        # 重新儲存編碼檔案
        with open(FACE_ENCODINGS_PATH, 'wb') as f:
            pickle.dump({
                'encodings': known_encodings,
                'names': known_names
            }, f)
        
        # 刪除該學生的圖片檔案
        import glob
        student_images = glob.glob(f'dataset/{name}_*.jpg')
        for img_path in student_images:
            try:
                os.remove(img_path)
                print(f"已刪除圖片: {img_path}")
            except Exception as e:
                print(f"刪除圖片失敗 {img_path}: {e}")
        
        print(f"學生已刪除: {name} (共刪除 {len(indices_to_remove)} 個編碼)")
        
        return jsonify({'success': True, 'message': f'已成功刪除學生 {name}'})
        
    except Exception as e:
        print(f"刪除學生失敗: {e}")
        return jsonify({'success': False, 'message': f'刪除失敗: {str(e)}'})

@app.route('/admin_settings')
@login_required
def admin_settings():
    """系統設定頁面"""
    config = load_config()
    username = session.get('username', 'Admin')
    
    return render_template('admin_settings.html',
                          username=username,
                          config=config)

@app.route('/admin_settings/save', methods=['POST'])
@login_required
def save_settings():
    """儲存系統設定"""
    try:
        data = request.get_json()
        
        # 載入當前設定
        config = load_config()
        
        # 更新設定
        if 'system' in data:
            config['system'].update(data['system'])
        
        if 'recognition' in data:
            # 驗證數值範圍
            if 'tolerance' in data['recognition']:
                tolerance = float(data['recognition']['tolerance'])
                if 0.3 <= tolerance <= 0.6:
                    config['recognition']['tolerance'] = tolerance
                else:
                    return jsonify({'success': False, 'message': 'Tolerance 必須在 0.3-0.6 之間'})
            
            if 'display_confidence_threshold' in data['recognition']:
                threshold = int(data['recognition']['display_confidence_threshold'])
                if 50 <= threshold <= 90:
                    config['recognition']['display_confidence_threshold'] = threshold
                else:
                    return jsonify({'success': False, 'message': '顯示信心度閾值必須在 50-90 之間'})
            
            if 'attendance_confidence_threshold' in data['recognition']:
                threshold = int(data['recognition']['attendance_confidence_threshold'])
                if 50 <= threshold <= 90:
                    config['recognition']['attendance_confidence_threshold'] = threshold
                else:
                    return jsonify({'success': False, 'message': '出席信心度閾值必須在 50-90 之間'})
            
            if 'process_every_n_frames' in data['recognition']:
                frames = int(data['recognition']['process_every_n_frames'])
                if 1 <= frames <= 120:
                    config['recognition']['process_every_n_frames'] = frames
                else:
                    return jsonify({'success': False, 'message': '處理幀數間隔必須在 1-120 之間'})
        
        if 'video' in data:
            if 'jpeg_quality' in data['video']:
                quality = int(data['video']['jpeg_quality'])
                if 60 <= quality <= 100:
                    config['video']['jpeg_quality'] = quality
                else:
                    return jsonify({'success': False, 'message': 'JPEG 品質必須在 60-100 之間'})
            
            if 'camera_index' in data['video']:
                config['video']['camera_index'] = int(data['video']['camera_index'])
        
        if 'admin' in data:
            # 更新管理員帳號密碼
            if 'username' in data['admin'] and data['admin']['username']:
                config['admin']['username'] = data['admin']['username']
            
            if 'password' in data['admin'] and data['admin']['password']:
                config['admin']['password'] = data['admin']['password']
            
            if 'session_timeout' in data['admin']:
                timeout = int(data['admin']['session_timeout'])
                if 300 <= timeout <= 86400:
                    config['admin']['session_timeout'] = timeout
        
        if 'attendance' in data:
            config['attendance'].update(data['attendance'])
        
        if 'course' in data:
            # 更新課程資訊
            if not config.get('course'):
                config['course'] = {}
            
            if 'course_name' in data['course']:
                config['course']['course_name'] = data['course']['course_name']
            
            if 'current_session' in data['course']:
                session_num = int(data['course']['current_session'])
                if session_num >= 1:
                    config['course']['current_session'] = session_num
            
            if 'total_sessions' in data['course']:
                total = int(data['course']['total_sessions'])
                if total >= 1:
                    config['course']['total_sessions'] = total
            
            if 'instructor' in data['course']:
                config['course']['instructor'] = data['course']['instructor']
        
        save_config(config)
        
        global system_config
        system_config = config
        
        print(f"系統設定已更新")
        
        return jsonify({'success': True, 'message': '設定已儲存成功！請重新啟動系統以套用所有更改。'})
        
    except Exception as e:
        print(f"儲存設定失敗: {e}")
        return jsonify({'success': False, 'message': f'儲存失敗: {str(e)}'})

@app.route('/admin_settings/reset', methods=['POST'])
@login_required
def reset_settings():
    """重置為預設設定"""
    try:
        default_config = {
            "system": {
                "system_name": "NPTU 國立屏東大學 AI 人臉辨識點名系統",
                "version": "3.4.0",
                "language": "zh-TW"
            },
            "recognition": {
                "tolerance": 0.45,
                "display_confidence_threshold": 60,
                "attendance_confidence_threshold": 65,
                "process_every_n_frames": 60
            },
            "video": {
                "jpeg_quality": 85,
                "camera_index": 0
            },
            "admin": {
                "username": "admin",
                "password": "admin123",
                "session_timeout": 3600
            },
            "attendance": {
                "auto_record": True,
                "record_format": "csv",
                "backup_enabled": False
            },
            "course": {
                "course_name": "計算機概論",
                "current_session": 1,
                "total_sessions": 3,
                "instructor": "林義凱教授"
            }
        }
        
        save_config(default_config)
        
        global system_config
        system_config = default_config
        
        print(f"系統設定已重置為預設值")
        
        return jsonify({'success': True, 'message': '設定已重置為預設值！'})
        
    except Exception as e:
        print(f"重置設定失敗: {e}")
        return jsonify({'success': False, 'message': f'重置失敗: {str(e)}'})

@app.route('/admin_export')
@login_required
def admin_export():
    """資料匯出頁面"""
    username = session.get('username', 'Admin')
    
    # 統計資訊
    student_info = load_student_info()
    total_students = len(student_info)
    
    # 統計出席記錄檔案數量
    import glob
    attendance_files = glob.glob('attendance_records/attendance_*.csv')
    total_records = len(attendance_files)
    
    # 統計總出席次數
    total_attendance_count = 0
    for file_path in attendance_files:
        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            total_attendance_count += len(df)
        except:
            pass
    
    return render_template('admin_export.html',
                          username=username,
                          total_students=total_students,
                          total_records=total_records,
                          total_attendance_count=total_attendance_count)

@app.route('/admin_export/students', methods=['POST'])
@login_required
def export_students():
    """匯出學生資料"""
    try:
        from io import BytesIO
        
        if request.is_json:
            data = request.get_json()
        else:
            data_str = request.form.get('data', '{}')
            data = json.loads(data_str)
        
        format_type = data.get('format', 'csv')
        
        student_info = load_student_info()
        
        if not student_info:
            return jsonify({'success': False, 'message': '沒有學生資料可匯出'})
        
        # 準備資料
        students_data = []
        for name, info in student_info.items():
            encoding_count = known_names.count(name)
            
            students_data.append({
                '姓名': name,
                '學號': info.get('student_id', 'N/A'),
                '註冊日期': info.get('registered_date', 'Unknown'),
                '編碼數量': encoding_count
            })
        
        df = pd.DataFrame(students_data)
        
        # 生成檔案名稱
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 在記憶體中生成檔案
        output = BytesIO()
        
        if format_type == 'csv':
            filename = f'students_export_{timestamp}.csv'
            df.to_csv(output, index=False, encoding='utf-8-sig')
            mimetype = 'text/csv'
        elif format_type == 'excel':
            filename = f'students_export_{timestamp}.xlsx'
            df.to_excel(output, index=False, engine='openpyxl')
            mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        else:
            return jsonify({'success': False, 'message': '不支援的格式'})
        
        output.seek(0)
        
        print(f"學生資料已生成: {filename} ({len(students_data)} 筆)")
        
        return send_file(
            output,
            mimetype=mimetype,
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        print(f"匯出學生資料失敗: {e}")
        return jsonify({'success': False, 'message': f'匯出失敗: {str(e)}'})

@app.route('/admin_export/attendance', methods=['POST'])
@login_required
def export_attendance():
    """匯出出席記錄"""
    try:
        from io import BytesIO
        
        if request.is_json:
            data = request.get_json()
        else:
            data_str = request.form.get('data', '{}')
            data = json.loads(data_str)
        
        format_type = data.get('format', 'csv')
        date_range = data.get('date_range', 'all')
        
        import glob
        
        # 獲取出席記錄檔案
        attendance_files = sorted(glob.glob('attendance_records/attendance_*.csv'), reverse=True)
        
        if not attendance_files:
            return jsonify({'success': False, 'message': '沒有出席記錄可匯出'})
        
        # 根據日期範圍篩選
        if date_range == 'today':
            today = datetime.now().strftime('%Y%m%d')
            attendance_files = [f for f in attendance_files if today in f]
        elif date_range == 'week':
            from datetime import timedelta
            week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')
            attendance_files = [f for f in attendance_files if os.path.basename(f).replace('attendance_', '').replace('.csv', '') >= week_ago]
        elif date_range == 'month':
            from datetime import timedelta
            month_ago = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
            attendance_files = [f for f in attendance_files if os.path.basename(f).replace('attendance_', '').replace('.csv', '') >= month_ago]
        
        # 合併所有記錄
        all_records = []
        for file_path in attendance_files:
            try:
                df = pd.read_csv(file_path, encoding='utf-8-sig')
                
                # 從檔名提取日期
                filename = os.path.basename(file_path)
                date_str = filename.replace('attendance_', '').replace('.csv', '')
                formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                
                # 新增日期欄位
                df['日期'] = formatted_date
                
                # 新增學號欄位
                student_info = load_student_info()
                df['學號'] = df['Name'].apply(lambda name: student_info.get(name, {}).get('student_id', 'N/A'))
                
                # 確保課程欄位存在（處理舊記錄）
                if 'Course' not in df.columns:
                    df['Course'] = '未設定'
                if 'Session' not in df.columns:
                    df['Session'] = '-'
                if 'Instructor' not in df.columns:
                    df['Instructor'] = '未設定'
                if 'SingleAttendanceMode' not in df.columns:
                    df['SingleAttendanceMode'] = 'No'  # 舊記錄默認為非單次點名模式
                
                # 重新排序欄位（包含 SingleAttendanceMode）
                df = df[['日期', 'Name', '學號', 'Course', 'Session', 'Instructor', 'Time', 'Confidence', 'Status', 'SingleAttendanceMode']]
                df.columns = ['日期', '姓名', '學號', '課程', '節次', '任課教師', '時間', '信心度', '狀態', 'SingleAttendanceMode']
                
                all_records.append(df)
            except Exception as e:
                print(f"讀取出席記錄失敗 {file_path}: {e}")
                continue
        
        if not all_records:
            return jsonify({'success': False, 'message': '沒有符合條件的出席記錄'})
        
        # 合併所有資料
        combined_df = pd.concat(all_records, ignore_index=True)
        
        # 處理單次點名模式：只展開標記為單次點名模式的記錄
        if not combined_df.empty:
            expanded_records = []
            
            for _, row in combined_df.iterrows():
                # 檢查該記錄是否標記為單次點名模式（新欄位）
                # 舊記錄可能沒有此欄位，默認為 'No'
                is_single_attendance = False
                if 'SingleAttendanceMode' in row and str(row['SingleAttendanceMode']).strip().upper() == 'YES':
                    is_single_attendance = True
                
                # 只展開標記為單次點名模式的記錄
                if is_single_attendance and row['課程'] != '未設定' and '/' in str(row['節次']):
                    # 解析節次資訊
                    try:
                        session_parts = str(row['節次']).split('/')
                        if len(session_parts) == 2:
                            total = int(session_parts[1])
                            # 為每個節次生成記錄
                            for session_num in range(1, total + 1):
                                new_row = row.copy()
                                new_row['節次'] = f"{session_num}/{total}"
                                expanded_records.append(new_row)
                        else:
                            # 無法解析，保留原記錄
                            expanded_records.append(row)
                    except:
                        # 解析失敗，保留原記錄
                        expanded_records.append(row)
                else:
                    # 非單次點名模式或舊記錄，保留原記錄
                    expanded_records.append(row)
            
            # 更新為擴展後的資料
            combined_df = pd.DataFrame(expanded_records)
            
            # 統計展開的記錄數
            single_mode_count = len([r for r in expanded_records if 'SingleAttendanceMode' in r and str(r.get('SingleAttendanceMode', '')).strip().upper() == 'YES'])
            if single_mode_count > 0:
                print(f"已展開 {single_mode_count} 筆單次點名模式記錄為所有節次")
        
        # 生成檔案名稱
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 在記憶體中生成檔案
        output = BytesIO()
        
        if format_type == 'csv':
            filename = f'attendance_export_{timestamp}.csv'
            combined_df.to_csv(output, index=False, encoding='utf-8-sig')
            mimetype = 'text/csv'
        elif format_type == 'excel':
            filename = f'attendance_export_{timestamp}.xlsx'
            combined_df.to_excel(output, index=False, engine='openpyxl')
            mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        else:
            return jsonify({'success': False, 'message': '不支援的格式'})
        
        output.seek(0)
        
        print(f"出席記錄已生成: {filename} ({len(combined_df)} 筆)")
        
        return send_file(
            output,
            mimetype=mimetype,
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        print(f"匯出出席記錄失敗: {e}")
        return jsonify({'success': False, 'message': f'匯出失敗: {str(e)}'})

@app.route('/admin_export/backup', methods=['POST'])
@login_required
def export_backup():
    """匯出完整備份"""
    try:
        from io import BytesIO
        import zipfile
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        zip_filename = f'system_backup_{timestamp}.zip'
        
        # 在記憶體中建立 ZIP 檔案
        memory_file = BytesIO()
        
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # 備份學生資訊
            if STUDENT_INFO_PATH.exists():
                zipf.write(STUDENT_INFO_PATH, 'student_info.json')
            
            # 備份編碼檔案
            if FACE_ENCODINGS_PATH.exists():
                zipf.write(FACE_ENCODINGS_PATH, 'face_encodings.pkl')
            
            # 備份設定檔案
            if CONFIG_PATH.exists():
                zipf.write(CONFIG_PATH, 'config.json')
            
            # 備份出席記錄
            if os.path.exists('attendance_records'):
                for root, dirs, files in os.walk('attendance_records'):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, '.')
                        zipf.write(file_path, arcname)
            
            # 備份學生圖片
            # if os.path.exists('dataset'):
            #     for root, dirs, files in os.walk('dataset'):
            #         for file in files:
            #             file_path = os.path.join(root, file)
            #             arcname = os.path.relpath(file_path, '.')
            #             zipf.write(file_path, arcname)
        
        memory_file.seek(0)
        
        print(f"完整備份已建立: {zip_filename}")
        
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=zip_filename
        )
        
    except Exception as e:
        print(f"建立備份失敗: {e}")
        return jsonify({'success': False, 'message': f'備份失敗: {str(e)}'})

@app.route('/')
def index():
    """主頁面"""
    global video_stream_active, active_camera
    # 確保回到主頁時影像串流可重開
    video_stream_active = True
    print(f"主頁載入 - video_stream_active 設置為: {video_stream_active}")
    
    # 釋放任何遺留的攝影機（非阻塞）
    try:
        if active_camera is not None:
            active_camera.release()
            active_camera = None
    except:
        pass
    
    # 檢查是否已登入，傳遞登入狀態給前端
    is_logged_in = 'logged_in' in session
    username = session.get('username', '')
    
    # 載入課程資訊和系統設定
    config = load_config()
    course_info = config.get('course', {
        'course_name': '計算機概論',
        'current_session': 1,
        'total_sessions': 3,
        'instructor': '林義凱教授'
    })
    
    # 載入系統資訊
    system_info = config.get('system', {
        'system_name': 'NPTU 國立屏東大學 AI 人臉辨識點名系統',
        'version': '3.4.0'
    })
    
    return render_template('index.html', 
                          is_logged_in=is_logged_in, 
                          username=username,
                          course_info=course_info,
                          system_name=system_info.get('system_name', 'NPTU 國立屏東大學 AI 人臉辨識點名系統'))

@app.route('/video_feed')
def video_feed():
    """影像串流路由"""
    global video_stream_active, active_camera
    # 啟動影像串流時，始終設置為 True
    video_stream_active = True
    
    print(f"影像串流啟動請求 - video_stream_active 設置為: {video_stream_active}")
    print(f"當前已註冊學生數: {len(set(known_names))}")
    
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/attendance_status')
def attendance_status():
    """取得今日出席狀態"""
    global known_names, attendance_today
    
    total_registered = len(set(known_names))
    
    # 載入課程資訊
    config = load_config()
    course_info = config.get('course', {})
    single_attendance = course_info.get('single_attendance', False)
    
    # 檢查 CSV 檔案是否存在，如果不存在則清空 attendance_today
    today = datetime.now().strftime("%Y%m%d")
    filename = f"attendance_records/attendance_{today}.csv"
    
    if not os.path.exists(filename):
        # CSV 若被手動刪除則清空記憶體中的出席記錄
        if len(attendance_today) > 0:
            print(f"今日 CSV 檔案不存在，清空 {len(attendance_today)} 位出席記錄")
            attendance_today.clear()
    
    # 從 CSV 檔案同步當前節次或課程的簽到學生
    attendance_list = list(attendance_today)
    
    # 獲取當前課程、教師和節次資訊
    current_course = course_info.get('course_name', '')
    current_instructor = course_info.get('instructor', '')
    current_session = course_info.get('current_session', 1)
    total_sessions = course_info.get('total_sessions', 1)
    session_str = f"{current_session}/{total_sessions}"
    
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename, encoding='utf-8-sig')
            
            if single_attendance:
                # 單次點名模式：只載入當前課程和教師的記錄（不限節次）
                if 'Course' in df.columns and 'Instructor' in df.columns:
                    df_filtered = df[
                        (df['Course'] == current_course) & 
                        (df['Instructor'] == current_instructor)
                    ]
                    all_attended = df_filtered['Name'].unique().tolist()
                else:
                    # 舊記錄格式（沒有課程資訊），保留所有學生
                    all_attended = df['Name'].unique().tolist()
                
                attendance_list = all_attended
                # 同步更新 attendance_today
                attendance_today.clear()
                attendance_today.update(all_attended)
                
                print(f"單次點名模式：從 CSV 載入 {len(all_attended)} 位學生（課程：{current_course}，教師：{current_instructor}）")
            else:
                # 非單次點名模式：只載入當前課程、教師和節次的記錄
                if 'Course' in df.columns and 'Instructor' in df.columns and 'Session' in df.columns:
                    df_filtered = df[
                        (df['Course'] == current_course) & 
                        (df['Instructor'] == current_instructor) &
                        (df['Session'] == session_str)
                    ]
                    all_attended = df_filtered['Name'].unique().tolist()
                    
                    attendance_list = all_attended
                    # 同步更新 attendance_today
                    attendance_today.clear()
                    attendance_today.update(all_attended)
                    
                    print(f"非單次點名模式：從 CSV 載入 {len(all_attended)} 位學生（課程：{current_course}，教師：{current_instructor}，節次：{session_str}）")
                else:
                    # 舊記錄格式（向後兼容）
                    print(f"檢測到舊格式 CSV，使用記憶體中的出席記錄")
        except Exception as e:
            print(f"讀取當日出席記錄失敗: {e}")
    
    present_today = len(attendance_list)
    
    # 載入學生資訊，為每個學生新增學號
    student_info = load_student_info()
    attendance_details = []
    for name in attendance_list:
        student_id = student_info.get(name, {}).get('student_id', '未設定')
        attendance_details.append({
            'name': name,
            'student_id': student_id
        })
    
    mode_text = "（單次點名模式）" if single_attendance else ""
    print(f"出席狀態查詢 {mode_text}- 已註冊: {total_registered} 人, 今日出席: {present_today} 人")
    
    return jsonify({
        'total_registered': total_registered,
        'present_today': present_today,
        'attendance_list': attendance_list,
        'attendance_details': attendance_details
    })

@app.route('/register')
def register_page():
    """學生註冊頁面"""
    global video_stream_active, active_camera
    # 停止影像串流以進入註冊頁面
    video_stream_active = False
    
    # 釋放正在使用的攝影機（非阻塞）
    try:
        if active_camera is not None:
            active_camera.release()
    except:
        pass
    
    return render_template('register.html')

@app.route('/start_registration', methods=['POST'])
def start_registration():
    """註冊流程"""
    global registration_active
    data = request.get_json()
    student_name = data.get('name')
    student_id = data.get('student_id', '')
    photo_count = data.get('photo_count', 5)
    
    if not student_name:
        return jsonify({'success': False, 'message': '請輸入學生姓名'})
    
    # 儲存學生資訊（姓名和學號）
    if student_id:
        try:
            add_student_info(student_name, student_id)
            print(f"學生資訊已儲存: {student_name} ({student_id})")
        except Exception as e:
            print(f"儲存學生資訊失敗: {e}")
    
    # 建立學生資料夾
    student_folder = os.path.join('dataset', student_name)
    os.makedirs(student_folder, exist_ok=True)
    
    registration_active = True
    
    return jsonify({
        'success': True, 
        'message': f'開始註冊 {student_name} ({student_id})，將拍攝 {photo_count} 張照片',
        'folder': student_folder
    })

@app.route('/capture_photo', methods=['POST'])
def capture_photo():
    """從影像串流擷取照片"""
    data = request.get_json()
    student_name = data.get('name')
    photo_index = data.get('index')
    image_data = data.get('image')
    
    if not all([student_name, photo_index is not None, image_data]):
        return jsonify({'success': False, 'message': '參數不完整'})
    
    try:
        # 解碼 base64 影像
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # 轉換為 OpenCV 格式
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 偵測人臉
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_img)
        
        if len(face_locations) == 0:
            return jsonify({'success': False, 'message': '未偵測到人臉，請重新拍攝'})
        
        if len(face_locations) > 1:
            return jsonify({'success': False, 'message': '偵測到多張人臉，請確保畫面中只有一人'})
        
        # 儲存照片（使用 imencode 以支援中文路徑）
        student_folder = os.path.join('dataset', student_name)
        filename = f'photo_{photo_index}_{int(time.time())}.jpg'
        filepath = os.path.join(student_folder, filename)
        
        # 使用 imencode 和二進位寫入以支援中文路徑
        is_success, buffer = cv2.imencode('.jpg', img)
        if is_success:
            with open(filepath, 'wb') as f:
                f.write(buffer)
            print(f"照片已儲存: {filepath}")
        else:
            return jsonify({'success': False, 'message': '影像編碼失敗'})
        
        return jsonify({
            'success': True, 
            'message': f'照片 {photo_index} 已成功儲存',
            'filename': filename
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'儲存失敗: {str(e)}'})

@app.route('/finish_registration', methods=['POST'])
def finish_registration():
    """完成註冊、重新訓練模型"""
    global registration_active, known_encodings, known_names
    registration_active = False
    
    data = request.get_json()
    student_name = data.get('name')
    
    print("\n" + "=" * 60)
    print(f"準備完成 {student_name} 的註冊...")
    print("=" * 60)
    
    try:
        # 訓練前的輸出
        print(f"訓練前: {len(set(known_names))} 個學生, {len(known_encodings)} 個編碼")
        
        # 重新訓練人臉編碼
        print("\n開始重新訓練模型...")
        from train_faces import train_known_faces
        train_known_faces()
        
        # 重新載入編碼
        print("\n重新載入編碼...")
        global student_info_dict
        known_encodings, known_names = load_encodings()
        student_info_dict = load_student_info()  # 重新載入學生資訊
        
        # 訓練完成後的輸出
        print(f"\n訓練完成!")
        print(f"訓練後: {len(set(known_names))} 個學生, {len(known_encodings)} 個編碼")
        print(f"已註冊學生: {sorted(set(known_names))}")
        print("=" * 60 + "\n")
        
        return jsonify({
            'success': True, 
            'message': f'{student_name} 註冊完成！人臉資料庫已更新（共 {len(set(known_names))} 位學生）'
        })
        
    except Exception as e:
        print(f"\n訓練失敗: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'訓練失敗: {str(e)}'})

@app.route('/check_face_quality', methods=['POST'])
def check_face_quality():
    """檢查人臉品質（用於自動擷取）"""
    data = request.get_json()
    image_data = data.get('image')
    
    try:
        # 解碼影像
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 轉換為 RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 偵測人臉
        face_locations = face_recognition.face_locations(rgb_img)
        
        if len(face_locations) == 0:
            return jsonify({
                'success': False,
                'quality': 0,
                'message': '未偵測到人臉，請重新拍攝'
            })
        
        if len(face_locations) > 1:
            return jsonify({
                'success': False,
                'quality': 0,
                'message': '偵測到多張人臉，請確保畫面中只有一人'
            })
        
        # 計算人臉品質分數
        top, right, bottom, left = face_locations[0]
        face_height = bottom - top
        face_width = right - left
        img_height, img_width = img.shape[:2]
        
        # 評估人臉大小、中心位置
        size_score = min((face_height * face_width) / (img_height * img_width) * 100, 50)
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        center_score = 50 - (abs(center_x - img_width/2) / img_width * 25 + 
                             abs(center_y - img_height/2) / img_height * 25)
        
        quality_score = size_score + center_score
        
        # 分數超過 80 可自動擷取
        can_capture = quality_score >= 80
        
        return jsonify({
            'success': True,
            'quality': round(quality_score),
            'can_capture': can_capture,
            'message': '人臉品質良好' if can_capture else '請調整位置'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'quality': 0,
            'message': f'偵測失敗: {str(e)}'
        })

if __name__ == '__main__':
    os.makedirs('attendance_records', exist_ok=True)
    os.makedirs('encodings', exist_ok=True)
    os.makedirs('dataset', exist_ok=True)
    
    # 啟動 Flask 伺服器
    print("=" * 50)
    print("NPTU 人臉辨識系統")
    print("請於瀏覽器中開啟主頁：http://localhost:5000")
    print("=" * 50)
    app.run(debug=False, host='0.0.0.0', port=5000)
