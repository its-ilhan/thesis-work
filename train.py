import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    roc_auc_score, classification_report
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/content/thesis-work')

from model import build_model, BERT_DIM, NUMERIC_DIM

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
VECTORS_PATH = "/content/processed/phase4_vectors.npz"
OUTPUT_DIR   = "/content/processed"
MODEL_PATH   = "/content/processed/best_model.pt"

BATCH_SIZE   = 16
EPOCHS       = 50
LR           = 1e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT    = 0.2
SEED         = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

def set_all_seeds(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_all_seeds(SEED)


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class AudioChunkDataset(Dataset):
    """
    Wraps the Phase 4 numpy arrays into a PyTorch Dataset.
    Splits each 811-dim vector back into BERT (768) and numeric (43) parts.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.bert    = torch.tensor(X[:, :BERT_DIM],   dtype=torch.float32)
        self.numeric = torch.tensor(X[:, BERT_DIM:],   dtype=torch.float32)
        self.labels  = torch.tensor(y,                 dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.bert[idx], self.numeric[idx], self.labels[idx]


# ─────────────────────────────────────────────
# AUDIO AUGMENTATION (training only)
# ─────────────────────────────────────────────

def augment_numeric_features(numeric: torch.Tensor) -> torch.Tensor:
    """
    Applies light random noise to numeric features during training.
    This is the feature-space equivalent of audio augmentation —
    it prevents the model from memorizing exact feature values
    and forces it to learn robust patterns instead.
    Only applied during training, never during validation/testing.
    """
    noise = torch.randn_like(numeric) * 0.01
    return numeric + noise


# ─────────────────────────────────────────────
# TRAINING HELPERS
# ─────────────────────────────────────────────

def get_sampler(y_train: np.ndarray) -> WeightedRandomSampler:
    """
    Creates a weighted sampler so that real and fake samples are
    seen equally often during training regardless of class imbalance.
    """
    class_counts = np.bincount(y_train.astype(int))
    weights      = 1.0 / class_counts
    sample_weights = torch.tensor(
        [weights[int(label)] for label in y_train],
        dtype=torch.float32
    )
    return WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Computes Equal Error Rate (EER) — the point where false acceptance
    rate equals false rejection rate. Lower EER = better model.
    Standard evaluation metric for audio deepfake detection.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr         = 1 - tpr
    # Find the threshold where FPR and FNR are closest
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer     = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    return float(eer)


def train_one_epoch(model, loader, optimizer, criterion, device, augment=True):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for bert, numeric, labels in loader:
        bert    = bert.to(device)
        numeric = numeric.to(device)
        labels  = labels.to(device)

        if augment:
            numeric = augment_numeric_features(numeric)

        optimizer.zero_grad()
        logits = model(bert, numeric).squeeze(1)
        loss   = criterion(logits, labels)
        loss.backward()

        # Gradient clipping — prevents exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).long().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.long().cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []

    with torch.no_grad():
        for bert, numeric, labels in loader:
            bert    = bert.to(device)
            numeric = numeric.to(device)
            labels  = labels.to(device)

            logits = model(bert, numeric).squeeze(1)
            loss   = criterion(logits, labels)
            probs  = torch.sigmoid(logits)

            total_loss += loss.item()
            preds = (probs > 0.5).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.long().cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy, np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────

def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses,   label="Val Loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(train_accs, label="Train Accuracy")
    ax2.plot(val_accs,   label="Val Accuracy")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "training_curves.png")
    plt.savefig(path)
    plt.close()
    print(f"  Training curves saved to: {path}")


def plot_confusion_matrix(labels, preds):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Fake", "Real"])
    ax.set_yticklabels(["Fake", "Real"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.colorbar(im)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(path)
    plt.close()
    print(f"  Confusion matrix saved to: {path}")


# ─────────────────────────────────────────────
# MASTER FUNCTION: Full Training Pipeline
# ─────────────────────────────────────────────

def train(vectors_path: str = VECTORS_PATH):
    set_all_seeds(SEED)
    # ── Device ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # ── Load Phase 4 vectors ──
    print("Loading Phase 4 vectors...")
    data = np.load(vectors_path, allow_pickle=True)
    X, y = data["X"], data["y"]
    print(f"  Total samples : {len(y)}")
    print(f"  Real          : {np.sum(y==1)}")
    print(f"  Fake          : {np.sum(y==0)}")
    print(f"  Feature dim   : {X.shape[1]}\n")

    # ── Train / Val split ──
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SPLIT,
        random_state=SEED, stratify=y
    )
    print(f"Train samples: {len(y_train)}  |  Val samples: {len(y_val)}\n")

    # ── Datasets and loaders ──
    train_dataset = AudioChunkDataset(X_train, y_train)
    val_dataset   = AudioChunkDataset(X_val,   y_val)

    sampler      = get_sampler(y_train)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        sampler=sampler
    )
    val_loader   = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    # ── Model, optimizer, loss ──
    model     = build_model(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=4, factor=0.5
    )

    # BCEWithLogitsLoss combines sigmoid + binary cross entropy
    # more numerically stable than applying sigmoid manually then BCE
    pos_weight = torch.tensor(
        [np.sum(y_train==0) / max(np.sum(y_train==1), 1)]
    ).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Training loop ──
    best_val_loss   = float("inf")
    patience_count  = 0
    EARLY_STOP      = 20

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    print("Starting training...\n")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7}")
    print("-" * 55)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, _, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"{epoch:>5} | {train_loss:>10.4f} | {train_acc:>9.4f} | {val_loss:>8.4f} | {val_acc:>7.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"        ✅ Best model saved (val_loss={val_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= EARLY_STOP:
                print(f"\n⚠️  Early stopping at epoch {epoch}")
                break

    # ── Final evaluation ──
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    _, _, val_labels, val_preds, val_probs = evaluate(
        model, val_loader, criterion, device
    )

    # Compute all metrics
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
    
    accuracy = accuracy_score(val_labels, val_preds)
    cm = confusion_matrix(val_labels, val_preds)
    tn, fp, fn, tp = cm.ravel()

    precision_fake = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_fake    = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision_real = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_real    = tp / (tp + fn) if (tp + fn) > 0 else 0

    try:
        auc = roc_auc_score(val_labels, val_probs)
    except Exception:
        auc = 0.0

    try:
        eer = compute_eer(val_labels, val_probs)
    except Exception:
        eer = 0.0

    print("\n" + "="*60)
    print("         FINAL EVALUATION RESULTS")
    print("="*60)
    print(f"\n  {'Metric':<30} {'Value':>10}")
    print(f"  {'-'*42}")
    print(f"  {'Overall Accuracy':<30} {accuracy*100:>9.2f}%")
    print(f"  {'AUC-ROC':<30} {auc:>10.4f}")
    print(f"  {'EER':<30} {eer*100:>9.2f}%")
    print(f"\n  {'--- Fake Detection ---':<30}")
    print(f"  {'Precision (Fake)':<30} {precision_fake*100:>9.2f}%")
    print(f"  {'Recall (Fake)':<30} {recall_fake*100:>9.2f}%")
    print(f"\n  {'--- Real Detection ---':<30}")
    print(f"  {'Precision (Real)':<30} {precision_real*100:>9.2f}%")
    print(f"  {'Recall (Real)':<30} {recall_real*100:>9.2f}%")
    print(f"\n  {'--- Confusion Matrix ---':<30}")
    print(f"  {'True Fake (correct)':<30} {tn:>10}")
    print(f"  {'False Real (fake→real)':<30} {fp:>10}")
    print(f"  {'False Fake (real→fake)':<30} {fn:>10}")
    print(f"  {'True Real (correct)':<30} {tp:>10}")
    print("="*60)

    # ── Plots ──
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    plot_confusion_matrix(val_labels, val_preds)

    print(f"\n✅ Training complete. Best model saved to: {MODEL_PATH}")
    return model


if __name__ == "__main__":
    train()