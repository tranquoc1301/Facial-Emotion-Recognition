import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
from collections import deque

# ================== CONFIG ==================
MODEL_PATH = 'best_model.keras'
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

EMOTION_LABELS = ['angry', 'contempt', 'disgust', 'fear',
                  'happy', 'neutral', 'sad', 'surprise']
CONFIDENCE_THRESHOLD = 0.6   # chỉ hiển thị nếu > 60%
SMOOTHING_WINDOW = 5

# Màu sắc cho từng emotion (BGR)
EMOTION_COLORS = {
    'happy': (0, 255, 0),
    'neutral': (255, 255, 0),
    'sad': (255, 0, 0),
    'angry': (0, 0, 255),
    'surprise': (0, 165, 255),
    'fear': (128, 0, 128),
    'disgust': (42, 42, 165),
    'contempt': (192, 192, 192)
}
# ============================================


def load_model_and_detector():
    """Load model và Haar cascade detector"""
    print("Đang load model...")
    try:
        model = load_model(MODEL_PATH)
        print("✓ Model đã được load thành công!")
    except Exception as e:
        print(f"✗ Lỗi khi load model: {e}")
        return None, None

    try:
        face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        if face_cascade.empty():
            raise IOError("Không load được Haarcascade")
        print("✓ Haarcascade detector đã được load thành công!")
    except Exception as e:
        print(f"✗ Lỗi khi load Haarcascade: {e}")
        return None, None

    return model, face_cascade


def preprocess_face(face):
    """Tiền xử lý khuôn mặt giống training pipeline"""
    face = cv2.resize(face, (96, 96))

    if len(face.shape) == 3:  # BGR -> Grayscale
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=(0, -1))  # (1,96,96,1)
    return face


def smooth_emotion(emotion_history, current_emotion):
    """Làm mượt dự đoán"""
    emotion_history.append(current_emotion)
    if len(emotion_history) >= 3:
        unique, counts = np.unique(emotion_history, return_counts=True)
        return unique[np.argmax(counts)]
    return current_emotion


def draw_enhanced_ui(frame, x, y, w, h, emotion, confidence, fps):
    """Vẽ UI kết quả"""
    color = EMOTION_COLORS.get(emotion, (255, 255, 255))
    thickness = 3 if confidence > 0.7 else 2

    cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)

    label = f"{emotion.upper()}: {confidence*100:.1f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, font_thickness = 0.7, 2

    (text_w, text_h), _ = cv2.getTextSize(
        label, font, font_scale, font_thickness)
    cv2.rectangle(frame, (x, y-text_h-10), (x+text_w+10, y), color, -1)
    cv2.putText(frame, label, (x+5, y-5), font, font_scale,
                (255, 255, 255), font_thickness)

    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (frame.shape[1]-120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def calculate_fps(fps_counter):
    """Tính FPS"""
    current_time = time.time()
    fps_counter.append(current_time)
    if len(fps_counter) > 1:
        time_diff = fps_counter[-1] - fps_counter[0]
        return len(fps_counter) / time_diff if time_diff > 0 else 0
    return 0


def main():
    model, face_cascade = load_model_and_detector()
    if model is None or face_cascade is None:
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Không thể mở camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("✓ Camera đã sẵn sàng!")
    print("\nNhấn 'q' để thoát | 's' để lưu ảnh")

    emotion_history = deque(maxlen=SMOOTHING_WINDOW)
    fps_counter = deque(maxlen=30)
    frame_count = 0

    # Warm up model
    print("Đang warm up model...")
    dummy_input = np.random.random((1, 96, 96, 1))
    model.predict(dummy_input, verbose=0)
    print("✓ Model warm up hoàn tất!")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("✗ Không thể đọc frame từ camera")
                break

            frame_count += 1
            frame = cv2.flip(frame, 1)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40)  # tránh detect mặt quá nhỏ
            )

            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size == 0:
                    continue

                processed_face = preprocess_face(face_roi)
                predictions = model.predict(processed_face, verbose=0)[0]

                emotion_idx = np.argmax(predictions)
                confidence = predictions[emotion_idx]

                if confidence > CONFIDENCE_THRESHOLD:
                    emotion = EMOTION_LABELS[emotion_idx]
                    smoothed_emotion = smooth_emotion(emotion_history, emotion)
                    fps = calculate_fps(fps_counter)

                    draw_enhanced_ui(frame, x, y, w, h,
                                     smoothed_emotion, confidence, fps)

                    # Debug
                    print(
                        f"Softmax: {np.round(predictions, 3)} -> {emotion} ({confidence:.2f})")

            cv2.imshow('Emotion Recognition (Haarcascade)', frame)

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
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Đã dọn dẹp tài nguyên")
        print(f"Tổng cộng xử lý {frame_count} frames")


if __name__ == "__main__":
    main()
