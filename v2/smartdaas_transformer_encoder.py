"""
smartdaas_transformer_encoder.py — SmartDaaS v2 Transformer Sequence Encoder
BEHRT-style Healthcare Transformer for HIV Patient Trajectories
Author: Lakshmi Kalyani Chinthala, SmartDaaS LLC
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

CONFIG = {
    "max_seq_len": 36,
    "d_model": 64,
    "n_heads": 4,
    "n_layers": 3,
    "d_ff": 128,
    "dropout": 0.1,
    "batch_size": 64,
    "n_epochs": 20,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "patience": 5,
}

VISIT_FEATURES = [
    "missed_visit", "refill_gap_days", "cd4_count", "viral_suppressed",
    "weight_kg", "who_stage", "side_effects", "tb_event", "oi_event",
    "regimen_changed", "cumulative_missed", "adherence_poor",
    "adherence_fair", "days_since_art_norm",
]
N_FEATURES = len(VISIT_FEATURES)


def prepare_sequences(event_path, patient_path, max_seq_len=36):
    print("[Data] Loading tables...")
    events = pd.read_csv(event_path)
    patients = pd.read_csv(patient_path)
    events["adherence_poor"] = (events["adherence_level"] == "Poor").astype(float)
    events["adherence_fair"] = (events["adherence_level"] == "Fair").astype(float)
    events["days_since_art_norm"] = events["days_since_art"] / events["days_since_art"].max()
    scaler = StandardScaler()
    cont_cols = ["refill_gap_days", "cd4_count", "weight_kg", "cumulative_missed", "days_since_art_norm"]
    events[cont_cols] = scaler.fit_transform(events[cont_cols].fillna(0))
    label_map = patients.set_index("patient_id")["interrupted"].to_dict()
    sequences, labels, pids = [], [], []
    grouped = events.groupby("patient_id")
    for pid in list(grouped.groups.keys()):
        if pid not in label_map:
            continue
        visits = grouped.get_group(pid).sort_values("visit_num")
        feat = visits[VISIT_FEATURES].fillna(0).values.astype(np.float32)
        if len(feat) > max_seq_len:
            feat = feat[-max_seq_len:]
        if len(feat) < max_seq_len:
            pad = np.zeros((max_seq_len - len(feat), N_FEATURES), dtype=np.float32)
            feat = np.vstack([pad, feat])
        sequences.append(feat)
        labels.append(float(label_map[pid]))
        pids.append(pid)
    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    print(f"[Data] Sequences: {X.shape}  Interruption rate: {y.mean()*100:.1f}%")
    return X, y, pids, scaler


class HIVSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


class SmartDaaSTransformer(nn.Module):
    def __init__(self, config=CONFIG):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(N_FEATURES, config["d_model"]),
            nn.LayerNorm(config["d_model"]),
            nn.ReLU(),
        )
        self.pos_encoding = PositionalEncoding(config["d_model"], config["max_seq_len"]+1, config["dropout"])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["d_model"], nhead=config["n_heads"],
            dim_feedforward=config["d_ff"], dropout=config["dropout"],
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config["n_layers"])
        self.classifier = nn.Sequential(
            nn.Linear(config["d_model"], config["d_model"]//2),
            nn.ReLU(), nn.Dropout(config["dropout"]),
            nn.Linear(config["d_model"]//2, 1),
        )
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def forward(self, x):
        pad_mask = (x.abs().sum(dim=-1) == 0)
        x = self.pos_encoding(self.input_projection(x))
        encoded = self.transformer(x, src_key_padding_mask=pad_mask)
        mask_exp = (~pad_mask).unsqueeze(-1).float()
        embedding = (encoded * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1)
        return self.classifier(embedding).squeeze(-1), embedding


def train_transformer(X, y, config=CONFIG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Training] Device: {device}")
    idx = np.arange(len(y))
    idx_train, idx_val = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=(y>0.5).astype(int))
    train_loader = DataLoader(HIVSequenceDataset(X[idx_train], y[idx_train]), batch_size=config["batch_size"], shuffle=True)
    val_loader   = DataLoader(HIVSequenceDataset(X[idx_val],   y[idx_val]),   batch_size=config["batch_size"])
    model = SmartDaaSTransformer(config).to(device)
    print(f"[Model] Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    pos_weight = torch.tensor([(y==0).sum()/(y==1).sum()]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["n_epochs"])
    history = {"train_loss": [], "val_loss": [], "val_auc": []}
    best_auc, patience_counter, best_state = 0, 0, None
    print(f"\n{'Epoch':<8} {'Train Loss':<14} {'Val Loss':<14} {'Val AUC':<12}")
    print("-"*50)
    for epoch in range(1, config["n_epochs"]+1):
        model.train()
        tl = []
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits, _ = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl.append(loss.item())
        model.eval()
        vl, preds, labs = [], [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits, _ = model(Xb)
                vl.append(criterion(logits, yb).item())
                preds.extend(torch.sigmoid(logits).cpu().numpy())
                labs.extend(yb.cpu().numpy())
        scheduler.step()
        tl_mean = np.mean(tl); vl_mean = np.mean(vl)
        val_auc = roc_auc_score(labs, preds)
        history["train_loss"].append(tl_mean)
        history["val_loss"].append(vl_mean)
        history["val_auc"].append(val_auc)
        print(f"{epoch:<8} {tl_mean:<14.4f} {vl_mean:<14.4f} {val_auc:<12.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"\n[Early stopping] Best Val AUC: {best_auc:.4f}")
                break
    model.load_state_dict(best_state)
    return model, history, idx_train, idx_val, device


def evaluate_and_embed(model, X, y, idx_val, device):
    model.eval()
    X_val = torch.tensor(X[idx_val], dtype=torch.float32)
    preds, embs = [], []
    with torch.no_grad():
        for i in range(0, len(X_val), 128):
            Xb = X_val[i:i+128].to(device)
            logits, emb = model(Xb)
            preds.extend(torch.sigmoid(logits).cpu().numpy())
            embs.append(emb.cpu().numpy())
    preds = np.array(preds)
    embs  = np.vstack(embs)
    auc   = roc_auc_score(y[idx_val], preds)
    auprc = average_precision_score(y[idx_val], preds)
    print(f"\n[Evaluation]")
    print(f"  AUC-ROC:  {auc:.4f}")
    print(f"  AUC-PR:   {auprc:.4f}")
    print(f"  Patient embeddings: {embs.shape[0]} patients x {embs.shape[1]} dimensions")
    return preds, embs, auc, auprc


def plot_training(history, output_path="/mnt/user-data/outputs/smartdaas_transformer_training.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("SmartDaaS v2 — Transformer Encoder Training", fontsize=13, fontweight="bold")
    ep = range(1, len(history["train_loss"])+1)
    axes[0].plot(ep, history["train_loss"], color="#0072B2", label="Train", linewidth=2)
    axes[0].plot(ep, history["val_loss"],   color="#CC79A7", label="Val",   linewidth=2)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)
    axes[1].plot(ep, history["val_auc"], color="#009E73", linewidth=2.5)
    axes[1].axhline(max(history["val_auc"]), color="grey", linestyle="--", alpha=0.5,
                    label=f"Best: {max(history['val_auc']):.4f}")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("AUC-ROC")
    axes[1].set_title("Validation AUC"); axes[1].set_ylim(0.5, 1.0); axes[1].legend()
    axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {output_path}")


def run_transformer(
    event_path="/mnt/user-data/outputs/smartdaas_synthetic_event_table.csv",
    patient_path="/mnt/user-data/outputs/smartdaas_synthetic_patient_table.csv",
):
    print("="*70)
    print("SmartDaaS v2 — Transformer Sequence Encoder")
    print("BEHRT-style HIV Patient Trajectory Modeling")
    print("="*70)
    X, y, pids, scaler = prepare_sequences(event_path, patient_path)
    model, history, idx_train, idx_val, device = train_transformer(X, y)
    preds, embs, auc, auprc = evaluate_and_embed(model, X, y, idx_val, device)
    plot_training(history)
    emb_df = pd.DataFrame(embs, columns=[f"emb_{i}" for i in range(embs.shape[1])])
    emb_df.insert(0, "patient_id", [pids[i] for i in idx_val])
    emb_df.insert(1, "interrupted", y[idx_val])
    emb_df.insert(2, "pred_prob", preds)
    emb_df.to_csv("/mnt/user-data/outputs/smartdaas_patient_embeddings.csv", index=False)
    torch.save(model.state_dict(), "/mnt/user-data/outputs/smartdaas_transformer.pt")
    print("\n" + "="*70)
    print(f"Transformer complete. AUC-ROC: {auc:.4f}  AUC-PR: {auprc:.4f}")
    print(f"Patient embedding dim: {embs.shape[1]}")
    print("Next: Multi-task learning → DeepSurv → Facility graph intelligence")
    print("="*70)
    return model, embs, auc

if __name__ == "__main__":
    run_transformer()
