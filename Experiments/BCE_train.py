import torch
import torch.nn as nn
import torch.optim as optim
import time
import random
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple


class Trainer:
    def __init__(self, model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader = None,
                 criterion: nn.Module = nn.BCEWithLogitsLoss(),
                 optimizer: torch.optim.Optimizer = None,
                 scheduler=None,
                 device: torch.device = None,
                 save_path: str = 'BCE_4_best_model.pth'):

        # reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer or torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

        self.scheduler = scheduler
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_path = save_path
        self.best_val_loss = float('inf')
        self.model.to(self.device)

        # 로그 초기화
        with open("training_log.csv", "w") as f:
            f.write("epoch,train_loss,val_loss,train_acc,val_acc\n")

    def one_hot(self, y, num_classes=4):
        if y.ndim == 1:  # 정수형 라벨일 경우만 변환
            y = torch.nn.functional.one_hot(y.long(), num_classes=num_classes)
        return y.float()

    def compute_accuracy(self, logits, y):
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        acc = (preds == y).float().mean()
        return acc.item()

    def train_epoch(self) -> Tuple[float, float]:

        self.model.train()
        running_loss = 0.0
        running_acc = 0.0
        for x, y in tqdm(self.train_loader, desc='Train', leave=False):
            x, y = x.to(self.device), self.one_hot(y.to(self.device))
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            acc = self.compute_accuracy(logits, y)
            running_loss += loss.item() * x.size(0)
            running_acc += acc * x.size(0)

        total = len(self.train_loader.dataset)
        return running_loss / total, running_acc / total

    def validate_epoch(self) -> Tuple[float, float]:
        if self.val_loader is None:
            return 0.0, 0.0
        self.model.eval()
        running_loss = 0.0
        running_acc = 0.0
        with torch.no_grad():
            for x, y in tqdm(self.val_loader, desc='Val', leave=False):
                x, y = x.to(self.device), self.one_hot(y.to(self.device))
                logits = self.model(x)
                loss = self.criterion(logits, y)
                acc = self.compute_accuracy(logits, y)

                running_loss += loss.item() * x.size(0)
                running_acc += acc * x.size(0)

        total = len(self.val_loader.dataset)
        return running_loss / total, running_acc / total

    def fit(self, epochs: int):
        for epoch in range(1, epochs + 1):
            start = time.time()
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()

            # 스케줄러 업데이트
            if self.scheduler:
                self.scheduler.step(val_loss)

            # 베스트 모델 저장
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.save_path)
                print(f"✅ Best model saved at epoch {epoch} (val_loss: {val_loss:.4f})")

            elapsed = time.time() - start
            print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
                  f"| Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Time: {elapsed:.1f}s")

            # 로그 저장
            with open("training_log.csv", "a") as f:
                f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f},{train_acc:.4f},{val_acc:.4f}\n")

  