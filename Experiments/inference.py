import torch
from torch.utils.data import DataLoader
from Model.res18_LSTM import VAModel
from Dataloader import VideoDataset
import argparse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Inference with trained model')
    parser.add_argument('--test-dir', default=r'D:\Project_2\DATA\Anomaly_detection\1\test', help='path to test data')
    parser.add_argument('--model-path', default=r'D:\Project_2\code\Experiments\res\BCE_4_best_model.pth', help='path to trained model checkpoint')
    parser.add_argument('--batch-size', default=4)
    parser.add_argument('--num-workers', default=4)
    parser.add_argument('--fps', default=3)
    parser.add_argument('--window-size', default=2)
    parser.add_argument('--hidden-dim', default=256)
    parser.add_argument('--num-layers', default=1)
    parser.add_argument('--num-classes', default=4)
    parser.add_argument('--bidirectional', default=True)
    parser.add_argument('--freeze-backbone', default=False)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = VideoDataset(args, args.test_dir)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = VAModel(args)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for x, y in tqdm(test_loader):
            x, y = x.to(device), y.to(device).float()
            outputs = model(x)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int()

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())


        y_pred = np.array(all_preds)   # shape: (N, num_classes)
        y_true_indices = np.array(all_labels).astype(int)
        y_true = np.eye(args.num_classes)[y_true_indices]
          
        # multilabel metric 적용
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

        print(f"F1 Score    : {f1:.4f}")
        print(f"Precision   : {precision:.4f}")
        print(f"Recall      : {recall:.4f}")



    from sklearn.metrics import classification_report

    target_names = ['Normal', 'Fall', 'Theft', 'Break']
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))



if __name__ == '__main__':
    main()