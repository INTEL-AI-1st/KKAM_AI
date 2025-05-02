import os
import argparse
import torch
from torch.utils.data import DataLoader
from Model.res18_LSTM import VAModel
from BCE_train import Trainer
from Dataloader import VideoDataset  # 사용자 정의 데이터셋 클래스
import torch
import time

"""
Normal, 전도, 절도, 파손
"""

def parse_args():
    parser = argparse.ArgumentParser(description='Video Anomaly Detection Training')
    # 경로 설정
    parser.add_argument('--train-dir', default=r'D:\Project_2\DATA\Anomaly_detection\1\train', help='path to training data directory')
    parser.add_argument('--val-dir', default=r'D:\Project_2\DATA\Anomaly_detection\1\val', help='path to validation data directory')
    parser.add_argument('--save-path', default=r'D:\Project_2\code\Experiments\res\BCE_4_best_model_1sec.pth', help='path to save the best model checkpoint')

    # 학습 파라미터
    parser.add_argument('--epochs', default=50, help='number of training epochs')
    parser.add_argument('--batch-size', default=4, help='batch size per GPU')
    parser.add_argument('--lr', default=1e-4, help='learning rate')

    # 데이터셋 파라미터
    parser.add_argument('--window-size', default=1, help='time window size for clip (sec)')
    parser.add_argument('--num-workers', default=4, help='number of DataLoader workers')
    parser.add_argument('--fps', default=3, help='number of frames per clip (3fps)') # 바꾸지 말기


    # 모델 파라미터
    parser.add_argument('--hidden-dim', default=256, help='LSTM hidden dimension')
    parser.add_argument('--num-layers', default=1, help='number of LSTM layers')
    parser.add_argument('--num-classes', default=4, help='number of anomaly classes; default = auto from dataset')
    parser.add_argument('--bidirectional', default=True, help='use bidirectional LSTM')
    parser.add_argument('--freeze-backbone', default=False, help='freeze ResNet backbone weights')



    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # 데이터셋 / DataLoader  
    train_dataset = VideoDataset(args, args.train_dir)
    val_dataset   = VideoDataset(args, args.val_dir)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"✅ CUDA 사용 중: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ CUDA 미사용: CPU로 대체")


    # 모델
    model = VAModel(args).to(device)

    # 옵티마이저 & 스케줄러
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        save_path=args.save_path
    )
    
    # 학습 시간 측정
    total_start = time.time()

    trainer.fit(epochs=args.epochs)

    total_elapsed = time.time() - total_start
    print(f"[⏱️] 전체 학습 소요 시간: {total_elapsed / 60:.2f} 분 ({total_elapsed:.1f}초)")



if __name__ == '__main__':
    main()
