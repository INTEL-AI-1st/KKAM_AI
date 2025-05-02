#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
server.py – 실내 이상행동 감지 및 기록 서버
----------------------------------------------------------------
• 웹캠 스트림에 YOLO+DeepSORT 트래킹 결과를 표시
• 2초(6프레임) window로 4class 이상행동 예측 (multi-label classification)
• 다음 중 하나라도 'Theft' 75%, 'Fall' 90%, 'Break' 80% 이상일 경우 Abnormal로 판정
• Supabase `class` 테이블에 이상 클래스 int 저장
• 나머지는 모두 Normal로 판별
• Supabase store 테이블에서 현재 서버 IP와 매핑된 uid, sid 사용

socketio.emit('class_update', {...})를 통해 Android에 실시간 전송

/video_feed는 MJPEG 영상 스트리밍 그대로 유지

socket_push_loop() 스레드를 통해 1초 주기로 클래스 푸시

latest_abnormal_class는 전역으로 관리되어 양쪽에서 공유
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import threading
import sys
import socket
from collections import deque
from flask import Flask, Response
import cv2
import numpy as np
import torch
from supabase import create_client
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import argparse
from flask_socketio import SocketIO, emit
import time

# 기존 YOLO+DeepSORT anonymize 로직
from YOLOv8_infoBox import detect_and_draw

# ────────────── Supabase 및 모델 세팅 ──────────────
SUPABASE_URL = "https://cyoaojdrmcecjhjrsjpl.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImN5b2FvamRybWNlY2poanJzanBsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDUxOTU2OTAsImV4cCI6MjA2MDc3MTY5MH0.cLnlc8eGbLlEv9x3aiEvGNg9jgTvuaWHNv1TdHRnEwg"
sb = create_client(SUPABASE_URL, SUPABASE_KEY)
# 현재 서버 IP로 store 조회
hostname = socket.gethostname()
LOCAL_IP = socket.gethostbyname(hostname)
resp = sb.table('store').select('sid','uid').eq('address', LOCAL_IP).execute()
if not resp.data:
    raise RuntimeError(f"Store not found for IP {LOCAL_IP}")
store_sid, store_uid = resp.data[0]['sid'], resp.data[0]['uid']

# VAModel import (single_video_infer.py와 동일)
try:
    sys.path.append(str(Path(__file__).resolve().parent / 'Experiments' / 'Model'))
    from res18_LSTM import VAModel
    
except ImportError:
    print("res18_LSTM.py not found"); sys.exit(1)

# 모델 체크포인트 및 디바이스
MODEL_PATH = Path(r"D:\Project_2\code\Experiments\res\BCE_4_best_model.pth")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 학습코드와 동일한 파라미터
FPS, WINDOW_SEC = 3, 2
SEQ_LEN  = FPS * WINDOW_SEC   # 6 frames
STRIDE   = SEQ_LEN * 2        # 12 frames
IMG_SIZE = 224
MEAN = np.array([0.485,0.456,0.406],np.float32)
STD  = np.array([0.229,0.224,0.225],np.float32)
IDX2LABEL = {0:"Normal", 1:"Fall", 2:"Break", 3: "Theft"}

# 모델 로드
arguments = argparse.Namespace(hidden_dim=256, num_layers=1, bidirectional=True,
                           freeze_backbone=False, num_classes=4,
                           window_size=WINDOW_SEC, fps=FPS)
model = VAModel(arguments).to(device)
model.load_state_dict(torch.load(str(MODEL_PATH), map_location=device))
model.eval()

# 전처리 함수
def preprocess(frame: np.ndarray) -> torch.Tensor:
    img = cv2.cvtColor(cv2.resize(frame, (IMG_SIZE, IMG_SIZE)),
                       cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return torch.from_numpy(img).permute(2, 0, 1)

# 한글 폰트
FONT = ImageFont.truetype(r"C:\Windows\Fonts\malgunbd.ttf", 18)

# ────────────────────────────────────────────────────
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')

# VIDEO_PATH = "D:\Project_2\Sample\market_CCTV_sample.mp4"  # 또는 상대경로 "./video/test_video.mp4"
# camera = cv2.VideoCapture(VIDEO_PATH)

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("❌ 웹캠을 열 수 없습니다.")
    sys.exit(1)

buffer = deque(maxlen=SEQ_LEN)
frame_cnt = 0
last_text = ''
latest_abnormal_class = None 



@app.route('/')
def index():
    return '''
    <h1 style="text-align: center;">아이스크림조아 신사홍대점 CAM1</h1>
    <img src="/video_feed" width="100%" height="auto">
    '''


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def gen_frames():
    global frame_cnt, latest_abnormal_class
    is_abnormal = False
    status = "분석 중"
    last_text = ""

    print("[INFO] gen_frames 시작됨")

    while True:
        ret, frame = camera.read()
        if not ret:
            print("⚠️ 프레임을 읽지 못했습니다.")
            break

        # 1) YOLO+DeepSORT 표시
        disp = detect_and_draw(frame.copy())

        # 2) 원본 프레임 버퍼 저장
        buffer.append(frame.copy())
        frame_cnt += 1

        # 3) 슬라이딩 윈도우 inference
        if frame_cnt % STRIDE == 0 and len(buffer) == SEQ_LEN:
            clip = list(buffer)
            tensor = torch.stack([preprocess(f) for f in clip])
            tensor = tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.sigmoid(logits).cpu().numpy()[0]


            # 확률 추출
            normal_prob, fall_prob, break_prob, theft_prob = probs

            # Abnormal 조건
            is_abnormal = (theft_prob >= 0.7 or break_prob >= 0.7 or fall_prob >= 0.7)
            
            if is_abnormal:
                status = "Abnormal 판정"
                abnormal_class = np.argmax([fall_prob, break_prob, theft_prob]) + 1
                latest_abnormal_class = int(abnormal_class)
                sb.table('abnormal').insert({
                    'uid': store_uid,
                    'sid': store_sid,
                    'class': latest_abnormal_class,
                }).execute()
            else:
                status = "Normal 판정"
                latest_abnormal_class = 0


            # 전체 클래스 확률 문자열 생성
            prob_str = ' | '.join(
                f"{IDX2LABEL[i]}: {probs[i]*100:.1f}%" for i in range(4)
            )
            last_text = f"{status} ({prob_str})"

            # Supabase 기록
            if status == "Abnormal 판정":
                abnormal_class = np.argmax([fall_prob, break_prob, theft_prob]) + 1
                latest_abnormal_class = int(abnormal_class)
                sb.table('abnormal').insert({
                    'uid': store_uid,
                    'sid': store_sid,
                    'class': latest_abnormal_class,
                }).execute()

            else:
                status = "Normal 판정"
                latest_abnormal_class = 0

            # 4) Top-4 overlay
            prob_str = ' | '.join(f"{IDX2LABEL[i]}: {probs[i]*100:.1f}%" for i in range(4))
            last_text = f"{status} ({prob_str})"

        pil = Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        w = disp.shape[1]

        if status == "Abnormal 판정":
            draw.rectangle([(5, 5), (w - 5, 35)], fill=(255, 200, 200))
            draw.text((10, 8), last_text, font=FONT, fill=(255, 0, 0))
        else:
            draw.rectangle([(5, 5), (w - 5, 35)], fill=(255, 255, 255))
            draw.text((10, 8), last_text, font=FONT, fill=(0, 0, 255))
        disp = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        cv2.imshow('Local Preview', disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        ret2, buf2 = cv2.imencode('.jpg', disp)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf2.tobytes() + b'\r\n')

    camera.release()
    cv2.destroyAllWindows()

@socketio.on('connect')
def on_connect():
    print("📡 Android client connected")

def socket_push_loop():
    global latest_abnormal_class
    while True:
        socketio.emit('class_update', {'class': latest_abnormal_class or 0})
        print("class update:", latest_abnormal_class or 0)
        time.sleep(1)

threading.Thread(target=socket_push_loop, daemon=True).start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
