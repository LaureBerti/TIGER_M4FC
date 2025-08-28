import os
import json
import numpy as np
import time
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    matthews_corrcoef,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from codecarbon import EmissionsTracker

# === 1. Load data ===
with open(".json", "r") as f:  # Embeddings path
    data = json.load(f)

# === 2. Prepare X and y ===
X = []
y = []

for entry in data:
    text_emb = entry["text_emb"]
    image_emb = entry["image_emb"]
    label = entry["label"]
    combined_emb = text_emb + image_emb
    X.append(combined_emb)
    y.append(label)

X = np.array(X)
y = LabelEncoder().fit_transform(y)
num_classes = len(set(y))

# === 3. Define XGBoost model ===
model = XGBClassifier(
    objective="multi:softprob",
    num_class=num_classes,
    tree_method="hist",
    eval_metric="mlogloss",
    verbosity=1,
    n_jobs=-1,
    random_state=42,
    n_estimators=100,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.9
)

# === 4. Cross-validation ===
cv = StratifiedKFold(n_splits=50, shuffle=True, random_state=42)

# === 5. Initialise metrics ===
accs, f1s, recs, precs, mccs, train_t, pred_t = [], [], [], [], [], [], []
global_cm = np.zeros((num_classes, num_classes), dtype=int)

# Start codecarbon tracker
tracker = EmissionsTracker(output_file='JinaClipV2/Gboost/GB_co2.txt', log_level="error")
tracker.start()

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    start_train = time.time()
    model.fit(X_train, y_train)
    end_train = time.time()
    
    start_pred = time.time()
    y_pred = model.predict(X_test)
    end_pred = time.time()

    accs.append(accuracy_score(y_test, y_pred))
    f1s.append(f1_score(y_test, y_pred, average='macro'))
    recs.append(recall_score(y_test, y_pred, average='macro'))
    precs.append(precision_score(y_test, y_pred, average='macro'))
    mccs.append(matthews_corrcoef(y_test, y_pred))
    train_t.append(end_train-start_train)
    pred_t.append(end_pred-start_pred)
    global_cm += confusion_matrix(y_test, y_pred, labels=range(num_classes))
    
    print(f" Fold {fold}/50 finished — Accuracy: {accs[-1]:.4f}, F1: {f1s[-1]:.4f}")
    
    # Saving the model of the current split
    os.makedirs('JinaClipV2/Gboost', exist_ok=True)
    model_filename = f"JinaClipV2/Gboost/GB_split{fold}.joblib"
    joblib.dump(model, model_filename)
    print(f"Model saved in {model_filename}")

# Stop codecarbon tracker
emissions = tracker.stop()

# === 6. Display the mean metrics ===
print("\n=== Average results on 50 folds ===")
print(f"Accuracy         : {np.mean(accs):.4f}")
print(f"Macro F1-score   : {np.mean(f1s):.4f}")
print(f"Macro Recall     : {np.mean(recs):.4f}")
print(f"Macro Precision  : {np.mean(precs):.4f}")
print(f"MCC              : {np.mean(mccs):.4f}")
print(f"Training time : {np.mean(train_t):.4f} sec")
print(f"Prediction time : {np.mean(pred_t):.4f} sec")
print(f'Total CO2 emissions (kg): {emissions:.6f}')

# === 7. Confusion matrix ===
plt.figure(figsize=(8, 6))
sns.heatmap(global_cm, annot=True, fmt="d", cmap="Oranges", xticklabels=range(num_classes), yticklabels=range(num_classes))
plt.xlabel("Predictions")
plt.ylabel("Ground truth")
plt.title("Confusion matrix on the 50 folds")
plt.tight_layout()
plt.show()

# === 8. Save metrics to file ===
with open("JinaClipV2/Gboost/GB_results.txt", "w") as f:
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

print("Detailed results written to GB_results.txt")