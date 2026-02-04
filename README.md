# NPTUDAS - 國立屏東大學 AI 智慧即時點名系統

NPTUDAS (NPTU Deep Attendance System) 是一款基於 Python Flask 與 Deep Learning 的即時人臉辨識點名系統。

主要針對課堂環境而設計，能夠透過鏡頭做到即時偵測並辨識學生，自動完成點名記錄。

（2025/12/15）本作品於【114-1學期屏大AI創作應用競賽暨成果展】中斬獲【**第一名**】佳績，得獎名單網址：https://ugsi2usr.nptu.edu.tw/p/406-1043-190073,r11.php

## 主要功能

### 1. 即時人臉辨識

- Web 介面顯示即時影像串流。
- 自動偵測畫面中的人臉並與資料庫比對。

### 2. 學生註冊系統

- 自動拍攝模式：系統偵測到清晰人臉時自動擷取。
- 手動拍攝模式：使用者可手動控制拍攝時機。
- 自動訓練模型：註冊完成後立即更新人臉特徵編碼。

### 3. 管理員後台

- 後台儀表板（DashBoard）：檢視註冊人數、今日出席人數、系統狀態。
- 學生管理：編輯學生資料、刪除學生、重新訓練模型。
- 系統設定：
  - 調整辨識容忍度 (Tolerance)。
  - 設定信心閾值 (Confidence Threshold)。
  - 選擇攝影機來源。
  - 修改課程資訊（授課教師、課程名稱）等等。
- 資料匯出：支援匯出學生名單與出席記錄 (CSV / Excel)。

### 4. 資料保存

- 自動將出席記錄儲存為 CSV 檔案。
- 支援系統設定檔 (`config.json`) 自定義。
- 具備資料備份功能。

## 技術架構

- 後端框架: Python Flask
- 電腦視覺: OpenCV, dlib, face_recognition
- 資料處理: Pandas, NumPy
- 前端介面: HTML5, CSS3, JavaScript（RWD 設計）

## 網頁截圖

<img width="1918" height="1027" alt="main page" src="https://github.com/user-attachments/assets/b78d06ae-9660-45ef-bb1e-413d79785112" />

<img width="1918" height="1027" alt="dashboard page" src="https://github.com/user-attachments/assets/ff68b114-2bfc-42ac-bc9c-bcd84ea96ec9" />

<img width="1915" height="1027" alt="registry page" src="https://github.com/user-attachments/assets/85c133f4-c1c2-483f-8e30-c9f128317d74" />

## 安裝執行

### 環境需求

- Windows / macOS / Linux
- Python 3.8 或以上版本
- 網路攝影機 (Webcam)

### 1. 安裝依賴套件

建議使用虛擬環境 (Virtual Environment)：

```bash
python -m venv venv

venv\Scripts\activate

pip install flask opencv-python face_recognition pandas numpy openpyxl
```

### 2. 執行

```bash
python app.py
```
