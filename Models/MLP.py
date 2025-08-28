import time
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker

# Configuration
DATA_PATH = ".json" # Embeddings path

# MLP Model
class MLP(nn.Module):
    def __init__(self, input_size=2048, hidden1=1024, hidden2=512, hidden3=128, hidden4=32, num_classes=3):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden3, hidden4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden4, num_classes),
        )

    def forward(self, x):
        return self.model(x)

# Loss function
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        logp = torch.nn.functional.log_softmax(inputs, dim=1)
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=inputs.size(1)).float()
        p = torch.exp(logp)
        loss = -self.alpha * (1 - p) ** self.gamma * logp
        loss = (loss * targets_one_hot).sum(dim=1)

        return loss.mean() if self.reduction == "mean" else loss.sum()
    
# Training function
def train_model(model, train_loader, val_loader, device, epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} terminé.")

# Load data
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)   
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Extract ambeddings and labels
X = np.array([np.array(d["text_emb"] + d["image_emb"], dtype=np.float32) for d in data])
label = [d["label"] for d in data]
le = LabelEncoder()
y = le.fit_transform(label)


# Metrics
accs, f1s, recs, precs, mccs, train_t, pred_t = [], [], [], [], [], [], []
global_cm = np.zeros((3,3), dtype=int)

# Start codecarbon tracker
tracker = EmissionsTracker(output_file='JinaClipV2/MLP/MLP_co2.txt', log_level="error")
tracker.start()

for i in range(50):
    
    # Split train/test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        
    print(f"\n Entraînement {i+1}...")
    batch_size = 64
    lr = 0.001
    
    train_ds = TensorDataset(torch.tensor(X_temp), torch.tensor(y_temp))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    model = MLP()
    model.to(device)
    start_train = time.time()
    train_model(model, train_loader, val_loader=None, device=device, epochs=10, lr=lr)
    end_train = time.time()
    
    # Saving the model of the current split
    torch.save(model, f"JinaClipV2/MLP/MLP_split{i+1}.pt")

    # Final evaluation
    model.eval()
    all_preds, all_targets = [], []
    start_infer = time.time()
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = model(xb).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(yb.numpy())
    end_infer = time.time()
    
    accuracy = accuracy_score(all_targets, all_preds)
    f_one = f1_score(all_targets, all_preds, average='macro')
    precision = precision_score(all_targets, all_preds, average='macro')
    recall = recall_score(all_targets, all_preds, average='macro')
    cm = confusion_matrix(all_targets, all_preds)
    mcc = matthews_corrcoef(all_targets, all_preds)
    
    accs.append(accuracy)
    f1s.append(f_one)
    precs.append(precision)
    recs.append(recall)
    global_cm += cm
    mccs.append(mcc) 
    train_t.append(end_train-start_train)
    pred_t.append(end_infer-start_infer)
    
# Stop codecarbon tracker
emissions = tracker.stop()
    
# Display the mean metrics 
print("\n=== Average results on 50 folds ===")
print(f"Accuracy         : {np.mean(accs):.4f}")
print(f"Macro F1-score   : {np.mean(f1s):.4f}")
print(f"Macro Recall     : {np.mean(recs):.4f}")
print(f"Macro Precision  : {np.mean(precs):.4f}")
print(f"MCC              : {np.mean(mccs):.4f}")
print(f"Training time : {np.mean(train_t):.4f} sec")
print(f"Prediction time : {np.mean(pred_t):.4f} sec")
print(f'Total CO2 emissions (kg): {emissions:.6f}')

# Confusion matrix
class_names = ['Supports','Refutes','Irrelevant']
plt.figure(figsize=(8, 6))
sns.heatmap(global_cm, annot=True, fmt="d", cmap='Oranges', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predictions")
plt.ylabel("Ground truth")
plt.title("Confusion matrix on the 50 folds")
plt.tight_layout()
plt.show()

# Save metrics to file 
with open("JinaClipV2/MLP/MLP_results.txt", "w") as f:
    f.write("=== Results per fold ===\n")
    for i in range(len(accs)):
        f.write(
            f"Fold {i+1:02d} | "
            f"Acc: {accs[i]:.4f} | "
            f"F1: {f1s[i]:.4f} | "
            f"Recall: {recs[i]:.4f} | "
            f"Precision: {precs[i]:.4f} | "
            f"MCC: {mccs[i]:.4f} | "
            f"TrainT: {train_t[i]:.4f}s | "
            f"PredT: {pred_t[i]:.4f}s\n"
        )
    
    f.write("\n=== Average results on 50 folds ===\n")
    f.write(f"Accuracy         : {np.mean(accs):.4f}\n")
    f.write(f"Macro F1-score   : {np.mean(f1s):.4f}\n")
    f.write(f"Macro Recall     : {np.mean(recs):.4f}\n")
    f.write(f"Macro Precision  : {np.mean(precs):.4f}\n")
    f.write(f"MCC              : {np.mean(mccs):.4f}\n")
    f.write(f"Training time    : {np.mean(train_t):.4f} sec\n")
    f.write(f"Prediction time  : {np.mean(pred_t):.4f} sec\n")
    f.write(f"Total CO2 emissions (kg): {emissions:.6f}\n")

print("Detailed results written to MLP_results.txt")
