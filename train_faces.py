import face_recognition
import os
import pickle
import cv2
import numpy as np

def train_known_faces(dataset_path='dataset'):
    """
    訓練在資料集中的所有臉部特徵
    生成128維的特徵向量編碼
    """
    known_encodings = []
    known_names = []
    
    print("=" * 50)
    print("開始臉部模型訓練...")
    print("=" * 50)
    
    if not os.path.exists(dataset_path):
        print(f"找不到資料集路徑: {dataset_path}")
        return None
    
    person_count = 0
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        
        if not os.path.isdir(person_folder):
            continue
        
        person_count += 1
        print(f"\n處理中: {person_name}")
        
        # 處理該學生的所有照片
        photo_count = 0
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            
            try:
                # 載入圖片並生成編碼（用 cv2 支援中文路徑）
                img = cv2.imread(image_path)
                if img is None:
                    # 嘗試以二進位模式讀取中文路徑
                    with open(image_path, 'rb') as f:
                        img_data = f.read()
                    nparr = np.frombuffer(img_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # 將 BGR 轉換為 RGB 以供 face_recognition 使用
                image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(image)
                
                if len(encodings) > 0:
                    known_encodings.append(encodings[0])
                    known_names.append(person_name)
                    photo_count += 1
                    print(f"已處理: {image_name}")
                else:
                    print(f"未偵測到臉部: {image_name}")
                    
            except Exception as e:
                print(f"處理失敗 {image_name}: {str(e)}")
        
        print(f"已處理照片總數: {photo_count}")
    
    if len(known_encodings) == 0:
        print("\n未成功處理任何臉部資料")
        return None
    
    # 儲存訓練結果
    os.makedirs('encodings', exist_ok=True)
    data = {"encodings": known_encodings, "names": known_names}
    
    with open('encodings/face_encodings.pkl', 'wb') as f:
        pickle.dump(data, f)
    
    print("\n" + "=" * 50)
    print(f"訓練完成!")
    print(f"已處理學生總數: {person_count} 人")
    print(f"已處理臉部照片總數: {len(known_encodings)} 張")
    print(f"編碼已儲存至: encodings/face_encodings.pkl")
    print("=" * 50)
    
    return data

if __name__ == "__main__":
    train_known_faces()
