import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader


class PersonVideoDataset(Dataset):
    def __init__(self, root_dir, seq_len=16, transform=None):
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.npy')]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        video = np.load(self.files[idx])  # (T, H, W, C)

        if video.shape[-1] == 3:
            video = np.transpose(video, (0, 3, 1, 2))  # (T, C, H, W)

        video_tensor = torch.tensor(video, dtype=torch.float32) / 255.0
        return video_tensor, 0  # 예시: 라벨 없음



class VideoDataset(Dataset):
    """
    Custom dataset for loading .npy data with class folders.
    Each class has subfolders X/ and y/ for features and labels.
    """
    def __init__(self, args, root_dir, transform=None):
        self.root_dir = root_dir
        self.args = args
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),  # (H, W, C) → (C, H, W)
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225])
        ])
        # 클래스 고정 정의
        LABEL_MAP = {
            'Normal': 0,
            '전도': 1,
            '파손': 2,
            '흡연': 3,
            '유기': 4,
            '절도': 5,
            '폭행': 6
        }


        self.class_to_idx = LABEL_MAP
        self.classes = list(LABEL_MAP.keys())

        self.samples = []

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        for cls_name in self.classes:
            x_dir = os.path.join(root_dir, cls_name, 'X')
            y_dir = os.path.join(root_dir, cls_name, 'y')

            for x_path in sorted(glob.glob(os.path.join(x_dir, '*.npy'))):
                filename = os.path.basename(x_path)
                y_path = os.path.join(y_dir, filename.replace('_X.npy', '_y.npy'))
                if os.path.exists(y_path):
                    self.samples.append((x_path, y_path, self.class_to_idx[cls_name]))

        if not self.samples:
            raise RuntimeError(f'No .npy pairs found in {root_dir}')

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x_path, y_path, label = self.samples[idx]
        x_data = np.load(x_path)  # (T, H, W, C)
        if x_data.shape[-1] == 3:
            x_data = np.transpose(x_data, (0, 3, 1, 2))  # (T, C, H, W)

        seq_len = self.args.window_size * self.args.fps  # 1 sec * 3 fps = 3 frames
        if x_data.shape[0] < seq_len:
            pad = seq_len - x_data.shape[0]
            x_data = np.pad(x_data, ((0, pad), (0, 0), (0, 0), (0, 0)), mode='edge')
        else:
            x_data = x_data[:seq_len]

        # Custom transform: OpenCV + normalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        frames = []
        for frame in x_data:
            frame = np.transpose(frame, (1, 2, 0))  # → (H, W, C)
            frame = cv2.resize(frame, (224, 224))
            frame = (frame / 255.0 - mean) / std
            frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1)
            frames.append(frame)

        x_tensor = torch.stack(frames)  # (seq_len, C, H, W)
        return x_tensor, label

    ### frame별 샘플로 구성
    # def __getitem__(self, idx):
    #     x_path, y_path, label = self.samples[idx]
    #     x_data = np.load(x_path)  # (T, H, W, C)

    #     # If needed, convert (T, H, W, C) → (T, C, H, W)
    #     if x_data.shape[-1] == 3:
    #         x_data = np.transpose(x_data, (0, 3, 1, 2))

    #     # Apply transform to each frame
    #     x_tensor = torch.stack([
    #         self.transform(np.transpose(frame, (1, 2, 0)))  # back to HWC for ToTensor
    #         for frame in x_data
    #     ])

    #     return x_tensor, label




def get_data_loaders(args):
    """
    Returns training and validation DataLoader for video anomaly detection.

    Args:
        train_dir: path to training data (with class subfolders)
        val_dir:   path to validation data (with class subfolders)
        seq_len:   number of frames per clip
        batch_size: batch size
        fps:       sampling FPS
        num_workers: number of loader workers

    Returns:
        train_loader, val_loader
    """
    train_dataset = VideoDataset(args.train_dir)
    val_dataset   = VideoDataset(args.val_dir)

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

    return train_loader, val_loader


if __name__ == '__main__':

    import argparse

    def parse_args():
        parser = argparse.ArgumentParser(description='Video Anomaly Detection Training')
        parser.add_argument('--train-dir', default='D:/Project_2/DATA/train.npy', help='path to training data directory')
        parser.add_argument('--val-dir', default='D:/Project_2/DATA/val.npy', help='path to validation data directory')
        parser.add_argument('--epochs', default=50, help='number of training epochs')
        parser.add_argument('--batch-size', default=8, help='batch size per GPU')
        parser.add_argument('--lr', default=1e-4, help='learning rate')

        parser.add_argument('--window-size', default=1, help='time window size for clip (sec)')

        parser.add_argument('--fps', default=3, help='number of frames per clip (3fps)') # 바꾸지 말기
        parser.add_argument('--hidden-dim', default=256, help='LSTM hidden dimension')
        parser.add_argument('--num-layers', default=1, help='number of LSTM layers')
        parser.add_argument('--num-classes', default=8, help='number of anomaly classes; default = auto from dataset')
        parser.add_argument('--bidirectional', action='store_true', help='use bidirectional LSTM')
        parser.add_argument('--freeze-backbone', action='store_true', help='freeze ResNet backbone weights')
        parser.add_argument('--save-path', default='best_model.pth', help='path to save the best model checkpoint')
        parser.add_argument('--num-workers', default=4, help='number of DataLoader workers')
        args = parser.parse_args()
        return args


    args = parse_args()


    # Example usage
    train_loader, val_loader = get_data_loaders(args)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val   samples: {len(val_loader.dataset)}")
