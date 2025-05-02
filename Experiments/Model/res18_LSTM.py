import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18(nn.Module):
    """
    Spatial feature extractor using pretrained ResNet18 (ImageNet weights).
    Input: Tensor of shape (B, 3, H, W)
    Output: Tensor of shape (B, feature_dim)
    """
    def __init__(self):
        super(ResNet18, self).__init__()
        resnet = models.resnet18(pretrained=False)
        modules = list(resnet.children())[:-1]  # remove final FC layer
        self.backbone = nn.Sequential(*modules)
        self.feature_dim = resnet.fc.in_features  # typically 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        feats = self.backbone(x)               # (B, feature_dim, 1, 1)
        feats = feats.view(feats.size(0), -1) # (B, feature_dim)
        return feats


class LSTM(nn.Module):
    """
    Temporal model using LSTM.
    Input: Tensor of shape (B, seq_len, input_dim)
    Output: Tensor of shape (B, hidden_dim)
    """
    def __init__(self,
                 input_dim,
                 hidden_dim = 256,
                 num_layers = 1,
                 bidirectional = False):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F, input_dim)
        out, _ = self.lstm(x)               # out: (B, F, hidden_dim * num_directions)
        # take last time-step
        last = out[:, -1, :]                # (B, hidden_dim * num_directions)
        return last


class VAModel(nn.Module):
    """
    Complete model combining ResNet feature extractor + LSTM temporal head + classifier.
    Input: Tensor of shape (B, F, 3, H, W)
    Output: logits Tensor of shape (B, num_classes)
    """
    def __init__(self, args):
        
        super(VAModel, self).__init__()
        self.args = args
        # Spatial extractor
        self.feature_extractor = ResNet18()
        feat_dim = self.feature_extractor.feature_dim

        if args.freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # Temporal model
        self.temporal_model = LSTM(
            input_dim=feat_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            bidirectional=args.bidirectional
        )
        out_dim = args.hidden_dim * (2 if args.bidirectional else 1)

        # Classifier
        self.classifier = nn.Linear(out_dim, args.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F, 3, H, W)
        B, F, C, H, W = x.size()
        # merge batch and time dims for spatial feature extraction
        x_reshaped = x.view(B * F, C, H, W)
        feats = self.feature_extractor(x_reshaped)    # (B*F, feat_dim)
        # restore sequence shape
        feats = feats.view(B, F, -1)                  # (B, F, feat_dim)

        # temporal LSTM
        temporal_out = self.temporal_model(feats)     # (B, hidden_dim*directions)

        # classification
        logits = self.classifier(temporal_out)        # (B, num_classes)
        return logits

if __name__ == "__main__":

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
    model = VAModel(args)
    inp = torch.randn(4, 16, 3, 224, 224)  # batch of 4, seq_len=16
    out = model(inp)  # (4, 8) logits

