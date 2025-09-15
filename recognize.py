import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
from collections import deque

# Cấu hình
MODEL_PATH = 'best_model.keras'
EMOTION_LABELS = ['angry', 'disgust', 'fear',
                  'happy', 'neutral', 'sad', 'surprise']
CONFIDENCE_THRESHOLD = 0.4
SMOOTHING_WINDOW = 5

# Màu sắc cho từng emotion (BGR format)
EMOTION_COLORS = {
    'happy': (0, 255, 0),      # Xanh lá
    'neutral': (255, 255, 0),   # Vàng
    'sad': (255, 0, 0),         # Xanh dương
    'angry': (0, 0, 255),       # Đỏ
    'surprise': (0, 165, 255),  # Cam
    'fear': (128, 0, 128),      # Tím
    'disgust': (42, 42, 165)    # Nâu
}


def load_model_and_detector():
    """Load model và face detector"""
    print("Đang load model...")
    try:
        model = load_model(MODEL_PATH)
        print("✓ Model đã được load thành công!")
    except Exception as e:
        print(f"✗ Lỗi khi load model: {e}")
        return None, None

    # Load face detector
    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            raise Exception("Không thể load Haar Cascade")
        print("✓ Face detector đã được load thành công!")
    except Exception as e:
        print(f"✗ Lỗi khi load face detector: {e}")
        return None, None

    return model, face_cascade


def preprocess_face(face):
    """Tiền xử lý khuôn mặt với các cải thiện"""
    # Resize về 48x48
    face = cv2.resize(face, (48, 48))

    # Convert to grayscale
    if len(face.shape) == 3:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    # Cải thiện contrast
    face = cv2.equalizeHist(face)

    # Giảm noise
    face = cv2.GaussianBlur(face, (3, 3), 0)

    # Normalize
    face = face.astype('float32') / 255.0

    # Reshape cho model
    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=-1)

    return face


def smooth_emotion(emotion_history, current_emotion):
    """Smooth emotion predictions để giảm flickering"""
    emotion_history.append(current_emotion)

    if len(emotion_history) >= 3:
        # Lấy emotion xuất hiện nhiều nhất
        unique, counts = np.unique(emotion_history, return_counts=True)
        return unique[np.argmax(counts)]

    return current_emotion


def draw_enhanced_ui(frame, x, y, w, h, emotion, confidence, fps):
    """Vẽ UI đẹp hơn cho emotion detection"""
    color = EMOTION_COLORS.get(emotion, (255, 255, 255))

    # Vẽ khung khuôn mặt với độ dày thay đổi theo confidence
    thickness = 3 if confidence > 0.7 else 2
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)

    # Tạo label
    label = f"{emotion.upper()}: {confidence*100:.1f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2

    # Tính kích thước text
    (text_w, text_h), _ = cv2.getTextSize(
        label, font, font_scale, font_thickness)

    # Vẽ background cho text
    cv2.rectangle(frame, (x, y-text_h-10), (x+text_w+10, y), color, -1)

    # Vẽ text
    cv2.putText(frame, label, (x+5, y-5), font, font_scale,
                (255, 255, 255), font_thickness)

    # Hiển thị FPS
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (frame.shape[1]-120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Hiển thị instructions
    instructions = "Press 'q' to quit, 's' to save image"
    cv2.putText(frame, instructions, (10, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def calculate_fps(fps_counter):
    """Tính FPS"""
    current_time = time.time()
    fps_counter.append(current_time)

    if len(fps_counter) > 1:
        time_diff = fps_counter[-1] - fps_counter[0]
        return len(fps_counter) / time_diff if time_diff > 0 else 0
    return 0


def main():
    """Hàm main chạy emotion recognition"""
    # Load model và detector
    model, face_cascade = load_model_and_detector()
    if model is None or face_cascade is None:
        return

    # Khởi tạo camera
    print("Đang khởi tạo camera...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("✗ Không thể mở camera")
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("✓ Camera đã sẵn sàng!")
    print("\nBắt đầu emotion recognition...")
    print("Nhấn 'q' để thoát")
    print("Nhấn 's' để lưu ảnh hiện tại")

    # Khởi tạo variables
    emotion_history = deque(maxlen=SMOOTHING_WINDOW)
    fps_counter = deque(maxlen=30)
    frame_count = 0

    # Warm up model
    dummy_input = np.random.random((1, 48, 48, 1))
    model.predict(dummy_input, verbose=0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("✗ Không thể đọc frame từ camera")
                break

            frame_count += 1
            start_time = time.time()

            # Flip frame horizontally để như mirror
            frame = cv2.flip(frame, 1)

            # Convert to grayscale cho face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50)  # Tăng kích thước tối thiểu
            )

            # Xử lý từng khuôn mặt
            for (x, y, w, h) in faces:
                # Extract face ROI
                face_roi = frame[y:y+h, x:x+w]

                # Preprocess
                processed_face = preprocess_face(face_roi)

                # Predict emotion
                predictions = model.predict(processed_face, verbose=0)
                emotion_idx = np.argmax(predictions[0])
                confidence = predictions[0][emotion_idx]

                # Chỉ hiển thị nếu confidence đủ cao
                if confidence > CONFIDENCE_THRESHOLD:
                    emotion = EMOTION_LABELS[emotion_idx]

                    # Smooth emotion
                    smoothed_emotion = smooth_emotion(emotion_history, emotion)

                    # Tính FPS
                    fps = calculate_fps(fps_counter)

                    # Vẽ UI
                    draw_enhanced_ui(frame, x, y, w, h,
                                     smoothed_emotion, confidence, fps)

            # Hiển thị frame
            cv2.imshow('Enhanced Emotion Recognition', frame)

            # Xử lý phím
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nĐang thoát...")
                break
            elif key == ord('s'):
                filename = f"emotion_capture_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"✓ Đã lưu ảnh: {filename}")

    except KeyboardInterrupt:
        print("\nDừng bởi người dùng")

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Đã dọn dẹp tài nguyên")
        print(f"Tổng cộng xử lý {frame_count} frames")


if __name__ == "__main__":
    main()
