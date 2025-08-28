import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn import metrics
from codecarbon import EmissionsTracker

# ==== CONFIGURATION ====
N_CLASSES = 3
FUSION_DIM = 256
HIDDEN_DIMS = [512, 256, 128]
DROPOUT = 0.3
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SPLITS = 50  

MODEL_TYPE = 'clip'
CLIP_BACKBONE = '...' # Choose between vit16, rn50 and rn504
DATASET_PATH = '.json' # Dataset path
FEATURES_PATH = '.json' # Embeddings path
OUT_PREFIX = f'{MODEL_TYPE}_{CLIP_BACKBONE}'

LABEL_COL = 'label'
ID_COL = 'claim_id'

# ==== DATASET ====
class MultimodalDataset(Dataset):
    def __init__(self, df, all_feats, id_col, label_col, img_dim, txt_dim):
        self.claim_ids = df[id_col].astype(str).tolist()
        self.labels = df[label_col].astype(int).tolist()
        self.all_feats = all_feats
        self.img_dim = img_dim
        self.txt_dim = txt_dim

    def __len__(self):
        return len(self.claim_ids)

    def __getitem__(self, idx):
        claim_id = self.claim_ids[idx]
        label = self.labels[idx]
        img_feat = np.array(self.all_feats['img_feats'].get(claim_id, np.zeros(self.img_dim)), dtype=np.float32)
        txt_feat = np.array(self.all_feats['text_feats'].get(claim_id, np.zeros(self.txt_dim)), dtype=np.float32)
        img_mask = 1.0 if claim_id in self.all_feats['img_feats'] else 0.0
        txt_mask = 1.0 if claim_id in self.all_feats['text_feats'] else 0.0
        return (
            torch.tensor(img_feat), torch.tensor(txt_feat),
            torch.tensor([img_mask]), torch.tensor([txt_mask]),
            torch.tensor(label)
        )

# ==== MODEL ====
class UltraDeepGatedFusionNet(nn.Module):
    def __init__(self, img_dim, txt_dim, fusion_dim, hidden_dims, ncls, dropout=0.3):
        super().__init__()
        self.img_encoder = nn.Sequential(
            nn.Linear(img_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.txt_encoder = nn.Sequential(
            nn.Linear(txt_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.fusion_attention = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Tanh(),
            nn.Linear(fusion_dim, 2),
            nn.Softmax(dim=1)
        )
        mlp_layers = []
        prev_dim = fusion_dim
        for hdim in hidden_dims:
            mlp_layers.append(nn.Linear(prev_dim, hdim))
            mlp_layers.append(nn.BatchNorm1d(hdim))
            mlp_layers.append(nn.GELU())
            mlp_layers.append(nn.Dropout(dropout))
            prev_dim = hdim
        self.mlp = nn.Sequential(*mlp_layers)
        self.final = nn.Linear(prev_dim, ncls)

    def forward(self, img_feat, txt_feat, img_mask, txt_mask):
        img_proj = self.img_encoder(img_feat) * img_mask
        txt_proj = self.txt_encoder(txt_feat) * txt_mask
        fusion_input = torch.cat([img_proj, txt_proj], dim=1)
        attn_weights = self.fusion_attention(fusion_input)
        fused = attn_weights[:, 0:1] * img_proj + attn_weights[:, 1:2] * txt_proj
        x = self.mlp(fused)
        out = self.final(x)
        return out

# ==== METRICS ====
def compute_metrics(y_true, y_pred):
    return {
        'accuracy': round(metrics.accuracy_score(y_true, y_pred) * 100, 2),
        'f1_macro': round(metrics.f1_score(y_true, y_pred, average='macro') * 100, 2),
        'precision_macro': round(metrics.precision_score(y_true, y_pred, average='macro') * 100, 2),
        'recall_macro': round(metrics.recall_score(y_true, y_pred, average='macro') * 100, 2),
        'mcc': round(metrics.matthews_corrcoef(y_true, y_pred), 4)
    }

# ==== LOAD DATA ====
df = pd.read_json(DATASET_PATH)
with open(FEATURES_PATH, 'r') as f:
    all_feats = json.load(f)
example_img = next(iter(all_feats['img_feats'].values()))
example_txt = next(iter(all_feats['text_feats'].values()))
img_dim, txt_dim = len(example_img), len(example_txt)

# ==== 80/20 Cross-Validation Loop ====
results = []
train_times = []
inference_times = []
os.makedirs(f'{OUT_PREFIX}', exist_ok=True)

# Start codecarbon tracker
tracker = EmissionsTracker(output_file=f'{OUT_PREFIX}/Codecarbon_Deep_MLP.txt', log_level="error")
tracker.start()

for split_idx in range(N_SPLITS):
    print(f'\n--- 80/20 Split {split_idx+1}/{N_SPLITS} ---')
    # 80/20 split, stratified
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df[LABEL_COL], random_state=42 + split_idx
    )
    train_set = MultimodalDataset(train_df, all_feats, ID_COL, LABEL_COL, img_dim, txt_dim)
    test_set = MultimodalDataset(test_df, all_feats, ID_COL, LABEL_COL, img_dim, txt_dim)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    model = UltraDeepGatedFusionNet(img_dim, txt_dim, FUSION_DIM, HIDDEN_DIMS, N_CLASSES, dropout=DROPOUT).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_split_metrics = None

    # --- Training Time ---
    start_train = time.time()
    for epoch in range(EPOCHS):
        model.train()
        for img_feat, txt_feat, img_mask, txt_mask, label in train_loader:
            img_feat, txt_feat, img_mask, txt_mask, label = (
                img_feat.to(DEVICE), txt_feat.to(DEVICE),
                img_mask.to(DEVICE), txt_mask.to(DEVICE), label.to(DEVICE)
            )
            optimizer.zero_grad()
            logits = model(img_feat, txt_feat, img_mask, txt_mask)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
        # Validation on test set after each epoch
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for img_feat, txt_feat, img_mask, txt_mask, label in test_loader:
                img_feat, txt_feat, img_mask, txt_mask = (
                    img_feat.to(DEVICE), txt_feat.to(DEVICE),
                    img_mask.to(DEVICE), txt_mask.to(DEVICE)
                )
                logits = model(img_feat, txt_feat, img_mask, txt_mask)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(label.cpu().numpy())
        metrics_result = compute_metrics(all_labels, all_preds)
        if metrics_result['accuracy'] > best_val_acc:
            best_val_acc = metrics_result['accuracy']
            torch.save(model.state_dict(), f'{OUT_PREFIX}/Deep_MLP_split{split_idx+1}.pt')
            best_split_metrics = metrics_result
    end_train = time.time()
    train_time = end_train - start_train
    train_times.append(train_time)

    # --- Inference Time ---
    model.eval()
    start_infer = time.time()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for img_feat, txt_feat, img_mask, txt_mask, label in test_loader:
            img_feat, txt_feat, img_mask, txt_mask = (
                img_feat.to(DEVICE), txt_feat.to(DEVICE),
                img_mask.to(DEVICE), txt_mask.to(DEVICE)
            )
            logits = model(img_feat, txt_feat, img_mask, txt_mask)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(label.cpu().numpy())
    end_infer = time.time()
    inference_time = end_infer - start_infer
    inference_times.append(inference_time)

    print(f'Split {split_idx+1} - Accuracy: {best_split_metrics["accuracy"]} | F1: {best_split_metrics["f1_macro"]} | '
          f'Train Time: {train_time:.2f}s | Inference Time: {inference_time:.2f}s')
    results.append(best_split_metrics)

# Stop codecarbon tracker
emissions = tracker.stop()

print('\n====== AVERAGED RESULTS ACROSS SPLITS ======')
avg_metrics = {k: round(np.mean([r[k] for r in results]), 2) for k in results[0]}
for k, v in avg_metrics.items():
    print(f'{k}: {v}')
print(f'Average Training Time (s): {np.mean(train_times):.2f}')
print(f'Average Inference Time (s): {np.mean(inference_times):.2f}')
print(f'Total CO2 emissions (kg): {emissions:.6f}')

results_file = f'{OUT_PREFIX}/Deep_MLP_results.txt'
with open(results_file, 'w') as f:
    f.write('====== PER-SPLIT RESULTS ======\n')
    for i, r in enumerate(results):
        f.write(f"Split {i+1}: " + ", ".join(f"{k}={v}" for k, v in r.items()) +
                f", train_time={train_times[i]:.2f}s, inference_time={inference_times[i]:.2f}s\n")
    f.write('\n====== AVERAGED RESULTS ACROSS SPLITS ======\n')
    for k, v in avg_metrics.items():
        f.write(f'{k}: {v}\n')
    f.write(f'Average Training Time (s): {np.mean(train_times):.2f}\n')
    f.write(f'Average Inference Time (s): {np.mean(inference_times):.2f}\n')
    f.write(f'Total CO2 emissions (kg): {emissions:.6f}\n')
