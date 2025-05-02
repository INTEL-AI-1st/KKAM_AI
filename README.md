# KKAM_AI
### 데이터 전처리
- 동영상(3fps/sec) label 별 영상 분리 및 데이터파일 재구성
- 프레임 별 라벨 할당 (Normal, Fall, Theft, Break)
- 영상 처리 (224*224), 채널 단위 Z-score 정규화

### AI 모델 구조
- Resnet18 + Bidirectional LSTM

### 학습 전략
- Multi-label classification (4-class)
- BCE + sigmoid: 각 클래스에 대한 **독립적인 2진 분류(binary classification)** 수행
- Validation loss 기반 best model 가중치 업데이트
  
#### 하이퍼파라미터 세팅
```
    # 학습 파라미터
    parser.add_argument('--epochs', default=50, help='number of training epochs')
    parser.add_argument('--batch-size', default=4, help='batch size per GPU')
    parser.add_argument('--lr', default=1e-4, help='learning rate')

    # 데이터셋 파라미터
    parser.add_argument('--window-size', default=2, help='time window size for clip (sec)')
    parser.add_argument('--num-workers', default=4, help='number of DataLoader workers')
    parser.add_argument('--fps', default=3, help='number of frames per clip (3fps)') 

    # 모델 파라미터
    parser.add_argument('--hidden-dim', default=256, help='LSTM hidden dimension')
    parser.add_argument('--num-layers', default=1, help='number of LSTM layers')
    parser.add_argument('--num-classes', default=4, help='number of anomaly classes; default = auto from dataset')
    parser.add_argument('--bidirectional', default=True, help='use bidirectional LSTM')
    parser.add_argument('--freeze-backbone', default=False, help='freeze ResNet backbone weights')


```
