import cv2
import torch
import time
import numpy as np

from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# CUDA 강제 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# YOLOv8 모델 로드 (사람 탐지)
yolo_person = YOLO('yolov8n.pt')  # yolov8n/s/m/l/x 중 선택

# DeepSORT 트래커
tracker = DeepSort(max_age=30, n_init=2)

# 체류시간 기록용 딕셔너리
id_start_time = {}

# 얼굴 블러링 함수
def anonymize_face_simple(frame, x1, y1, x2, y2, method='pixelate'):
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return frame

    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return frame

    if method == 'blur':
        face = cv2.GaussianBlur(face, (15, 15), 10)
    elif method == 'pixelate':
        temp = cv2.resize(face, (10, 10), interpolation=cv2.INTER_LINEAR)
        face = cv2.resize(temp, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

    frame[y1:y2, x1:x2] = face
    return frame

# 전체 감지 및 표시 함수
def detect_and_draw(frame):
    start_time = time.time()

    # YOLO 감지
    results_person = yolo_person(frame, conf=0.8, verbose=False)[0]  # 첫 결과

    boxes = results_person.boxes
    dets_person = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy()

    # 사람 감지된 박스 리스트 구성
    detections = []
    for i in range(len(dets_person)):
        if int(clss[i]) == 0:  # class 0: person
            x1, y1, x2, y2 = map(int, dets_person[i])
            conf = confs[i]
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

    # DeepSORT 추적 업데이트
    tracks = tracker.update_tracks(detections, frame=frame)
    h, w = frame.shape[:2]

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = int(track.track_id)
        l, t, r, b = map(int, track.to_ltrb())
        current_time = time.time()

        # 체류 시간 계산
        if track_id not in id_start_time:
            id_start_time[track_id] = current_time
        duration = current_time - id_start_time[track_id]
        duration_min = int(duration // 60)
        duration_sec = int(duration % 60)

        # 얼굴 블러 처리
        face_y2 = t + int((b - t) * 0.4)
        frame = anonymize_face_simple(frame, l, t, r, face_y2, method='blur')

        score = getattr(track, 'det_conf', 0.0) or 0.0
        cv2.rectangle(frame, (l, t), (r, b), (255, 150, 0), 2)

        # 정보 박스 위치
        box_width = 220
        box_height = 60
        info_box_x = l
        info_box_y = b + 10 if b + 10 + box_height < h else t - box_height - 10
        info_box_y = max(info_box_y, 0)

        # 반투명 박스 그리기
        overlay = frame.copy()
        cv2.rectangle(overlay, (info_box_x, info_box_y),
                      (info_box_x + box_width, info_box_y + box_height),
                      (50, 50, 50), -1)
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # 텍스트 그리기 - 가독성을 위해 밝은 글씨와 검은 외곽선 추가
        def draw_text_with_outline(img, text, org, font, font_scale, color, thickness):
            x, y = org
            cv2.putText(img, text, (x, y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)  # 외곽선
            cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

        draw_text_with_outline(frame, f'Visitor {track_id:03d}', (info_box_x + 10, info_box_y + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        draw_text_with_outline(frame, f'Confidence: {score:.2f}', (info_box_x + 10, info_box_y + 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200), 1)
        draw_text_with_outline(frame, f'Stay: {duration_min}m {duration_sec}s',
                               (info_box_x + 10, info_box_y + 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 255), 1)

    fps = 1.0 / (time.time() - start_time + 1e-5)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return frame

# 테스트 실행
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)  # 또는 "D:/Project_2/Sample/market.mp4"

    if not cap.isOpened():
        print("❌ 웹캠을 열 수 없습니다.")
        exit()

    print("🎥 실시간 분석 시작")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("🔁 영상 끝 - 다시 시작")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        result = detect_and_draw(frame)
        cv2.imshow("YOLOv8 + DeepSORT + Face Blur", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ 종료 완료")
