import torch
from torch.utils.data import DataLoader
from Model.res18_LSTM import VAModel
from Dataloader import VideoDataset  # 사용자 정의 데이터셋 클래스
import argparse
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Inference with trained model')
    parser.add_argument('--test-dir', default=r'D:\Project_2\DATA\깜빡catch 학습데이터\1\test', help='path to test data')
    parser.add_argument('--model-path', default=r'D:\Project_2\code\Experiments\res\win2_best_model.pth', help='path to trained model checkpoint')
    parser.add_argument('--batch-size', default=4) #32
    parser.add_argument('--num-workers', default=4)
    parser.add_argument('--fps', default=3)
    parser.add_argument('--window-size', default=2) # 1
    parser.add_argument('--hidden-dim', default=256)
    parser.add_argument('--num-layers', default=1)
    parser.add_argument('--num-classes', default=7)
    parser.add_argument('--bidirectional', default=True)
    parser.add_argument('--freeze-backbone', default=False, help='freeze ResNet backbone weights')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터 로딩
    test_dataset = VideoDataset(args, args.test_dir)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 모델 생성 및 가중치 로딩
    model = VAModel(args)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    
    all_labels = []
    all_preds = []
    all_probs = []

    model.eval()
    with torch.no_grad():
        for x, y in test_loader:  # or test_loader
            x, y = x.to(device), y.to(device)
            outputs = model(x)  # shape: (B, num_classes)
            probs = torch.softmax(outputs, dim=1)

            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # numpy array로 변환
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_score = np.array(all_probs)

    # 평가 지표 계산
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')  # 다중 클래스 weighted 방식
    top3 = top_k_accuracy_score(y_true, y_score, k=3)

    print(f"Accuracy     : {acc:.4f}")
    print(f"F1 Score     : {f1:.4f}")
    print(f"Top-3 Accuracy: {top3:.4f}")

if __name__ == '__main__':
    main()
