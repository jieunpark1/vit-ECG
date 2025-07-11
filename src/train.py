import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
import torch.nn.functional as F
import numpy as np

from utils.dataset_loader import MIMICIVECGDataset
from src.config import load_config  # load_config 함수는 yaml 파싱하는 것으로 가정

from src.model import SimpleECGCNN

def train():
    cfg = load_config('configs/cnn_config.yaml')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleECGCNN(cfg['input_channels'], cfg['num_classes']).to(device)

    # Train/Validation 데이터셋 분리
    train_dataset = MIMICIVECGDataset(
        train=True,
        train_ratio=cfg.get('train_ratio', 0.8),
        shuffle=True
    )
    val_dataset = MIMICIVECGDataset(
        train=False,
        train_ratio=cfg.get('train_ratio', 0.8),
        shuffle=True
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])

    epoch_losses = []

    for epoch in range(cfg['epochs']):
        # --- Training ---
        model.train()
        total_loss = 0
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
        for signals, labels in train_iter:
            signals, labels = signals.to(device), labels.to(device)
            outputs = model(signals)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_iter.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{cfg['epochs']}, Train Loss: {avg_loss:.4f}")

        # --- Validation ---
        model.eval()
        all_labels = []
        all_probs = []
        val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation")
        with torch.no_grad():
            for val_signals, val_labels in val_iter:
                val_signals, val_labels = val_signals.to(device), val_labels.to(device)
                val_outputs = model(val_signals)

                probs = F.softmax(val_outputs, dim=1)[:, 1]  # 클래스 1 확률

                all_labels.extend(val_labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        val_auroc = roc_auc_score(all_labels, all_probs)
        val_auprc = average_precision_score(all_labels, all_probs)

        print(f"Val AUROC: {val_auroc:.4f}, Val AUPRC: {val_auprc:.4f}")

    # 모델 저장
    os.makedirs(os.path.dirname(cfg['model_save_path']), exist_ok=True)
    torch.save(model.state_dict(), cfg['model_save_path'])

    # Loss 그래프 저장
    plt.plot(range(1, cfg['epochs'] + 1), epoch_losses, marker='o')
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("outputs/loss_plot.png")
    print("✅ Saved loss plot at outputs/loss_plot.png")


if __name__ == '__main__':
    train()
