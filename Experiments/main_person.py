import os
import cv2
import torch
import numpy as np
from collections import defaultdict, deque
from deep_sort_realtime.deepsort_tracker import DeepSort
from Model.res18_LSTM import VAModel
import torch.nn as nn
import torch.optim as optim
import argparse

# --- 파서 정의 (main.py 방식 완전 반영) ---
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', default=r'D:/Project_2/DATA/train')
    parser.add_argument('--val-dir', default=r'D:/Project_2/DATA/val')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--window-size', default=2, type=int)
    parser.add_argument('--fps', default=3, type=int)
    parser.add_argument('--hidden-dim', default=256, type=int)
    parser.add_argument('--num-layers', default=1, type=int)
    parser.add_argument('--num-classes', default=7, type=int)
    parser.add_argument('--bidirectional', default=True)
    parser.add_argument('--freeze-backbone', default=False)
    parser.add_argument('--save-path', default=r'D:/Project_2/code/Experiments/res/win2_biLSTM_person_best_model.pth')
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--video-path', default=r'D:/Project_2/VIDEO/sample.mp4')
    return parser.parse_args()

args = parse_args()

# --- 설정 ---
seq_len = args.window_size * args.fps
image_size = 224

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 모델 초기화 ---
model = VAModel(args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()

# --- 라벨 매핑 (예: track_id → 라벨), 여기서는 0으로 고정 ---
track_id_to_label = defaultdict(lambda: 0)

# --- YOLOv5 로드 ---
yolo = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
yolo.conf = 0.6

# --- DeepSORT ---
tracker = DeepSort(max_age=30)

# --- 시퀀스 버퍼 ---
person_buffers = defaultdict(lambda: deque(maxlen=seq_len))

# --- 비디오 경로로 실행 ---
cap = cv2.VideoCapture(args.video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo(frame)
    dets = results.xyxy[0].cpu().numpy()
    detections = []

    for x1, y1, x2, y2, conf, cls in dets:
        if int(cls) == 0:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = int(track.track_id)
        l, t, r, b = map(int, track.to_ltrb())
        crop = frame[t:b, l:r]

        if crop.size == 0:
            continue

        crop = cv2.resize(crop, (image_size, image_size))
        crop = crop[:, :, ::-1]  # BGR → RGB
        person_buffers[track_id].append(crop)

        if len(person_buffers[track_id]) == seq_len:
            seq = np.stack(person_buffers[track_id])  # (T, H, W, C)
            seq = torch.from_numpy(seq).permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)
            seq = seq.unsqueeze(0).to(device)  # (1, T, C, H, W)

            label = torch.tensor([track_id_to_label[track_id]], dtype=torch.long).to(device)

            model.train()
            optimizer.zero_grad()
            output = model(seq)  # (1, num_classes)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            print(f"[Train] track_id={track_id}, loss={loss.item():.4f}")

    cv2.imshow("Live Tracking & Training", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
