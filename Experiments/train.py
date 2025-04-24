import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms

class Trainer:
    """
    Trainer for VideoAnomalyModel with best-model saving based on validation loss.
    """
    def __init__(self, model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader = None,
                 criterion: nn.Module = nn.CrossEntropyLoss(),
                 optimizer: torch.optim.Optimizer = None,
                 scheduler=None,
                 device: torch.device = None,
                 save_path: str = 'best_model.pth'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer or torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

        self.scheduler = scheduler
        self.device = device or torch.device('cuda')
        self.save_path = save_path
        self.best_val_loss = float('inf')
        self.model.to(self.device)


    def train_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0
        for x, y in tqdm(self.train_loader, desc='Train', leave=False):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()
            torch.cuda.empty_cache()
            running_loss += loss.item() * x.size(0)
        return running_loss / len(self.train_loader.dataset)

    def validate_epoch(self) -> float:
        if self.val_loader is None:
            return 0.0
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for x, y in tqdm(self.val_loader, desc='Val', leave=False):
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.criterion(logits, y)
                running_loss += loss.item() * x.size(0)
        return running_loss / len(self.val_loader.dataset)

    def fit(self, epochs: int):
        for epoch in range(1, epochs+1):
            start = time.time()
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()
            # scheduler step based on validation
            if self.scheduler:
                self.scheduler.step(val_loss)
            # save best model if validation loss improved
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.save_path)
                print(f"Saved best model at epoch {epoch} with val_loss: {val_loss:.4f}")
            elapsed = time.time() - start
            print(f"Epoch {epoch}/{epochs} "
                  f"Train Loss: {train_loss:.4f} "
                  f"Val Loss: {val_loss:.4f} "
                  f"Time: {elapsed:.1f}s")
