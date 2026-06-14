"""
smartdaas_multitask.py — SmartDaaS v2 Multi-Task Learning Module
Simultaneously predicts: interruption + viral failure + poor adherence + high missed visits
Author: Lakshmi Kalyani Chinthala, SmartDaaS LLC
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
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

TARGETS = {
    "interrupted"        : "Treatment Interruption",
    "viral_failure"      : "Viral Failure (VL>=1000)",
    "target_poor_adherence": "Poor Adherence",
    "high_missed_visits" : "High Missed Visit Rate (>20%)",
}

FEATURES = [
    "age_at_art_start","sex_female","cd4_at_art_start","most_recent_cd4",
    "cd4_improvement","who_stage_start","weight_at_start","weight_change",
    "bmi_at_start","days_to_art","had_interruption_history","opp_infection_any",
    "side_effects_any","tb_any","stage_worsened","n_total_visits","n_missed_visits",
    "missed_visit_rate","mean_refill_gap_days","max_refill_gap_days",
    "n_regimen_changes","final_viral_load","viral_suppressed_last",
    "n_poor_adherence_visits","days_follow_up",
]

CONFIG = {
    "hidden_dims"  : [256, 128, 64],
    "shared_dim"   : 64,
    "dropout"      : 0.2,
    "batch_size"   : 128,
    "n_epochs"     : 30,
    "lr"           : 1e-3,
    "weight_decay" : 1e-4,
    "patience"     : 7,
}


def prepare_data(patient_path):
    print("[Data] Loading patient table...")
    df = pd.read_csv(patient_path)
    df["viral_failure"]      = (df["final_viral_load"] >= 1000).astype(int)
    df["high_missed_visits"] = (df["missed_visit_rate"] > 0.2).astype(int)
    le = LabelEncoder()
    for col in ["last_adherence_level","last_regimen","initial_regimen"]:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
    avail = [f for f in FEATURES if f in df.columns]
    X_raw = df[avail].values.astype(np.float32)
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(X_raw).astype(np.float32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    Y = df[list(TARGETS.keys())].values.astype(np.float32)
    print(f"[Data] Features: {X.shape[1]}  Patients: {X.shape[0]}")
    for i,(k,v) in enumerate(TARGETS.items()):
        print(f"  {v}: {Y[:,i].mean()*100:.1f}%")
    return X, Y, avail


class MultiTaskDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self): return len(self.Y)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]


class SmartDaaSMultiTask(nn.Module):
    """
    Multi-task neural network with shared encoder + task-specific heads.
    
    Architecture:
        Input → Shared encoder (learns joint HIV patient representation)
        → Task heads (4 separate prediction layers, one per outcome)
    
    Why multi-task works:
        Interruption and viral failure are correlated — learning them
        together creates richer shared representations than learning each
        separately. The shared encoder becomes a general HIV patient
        risk encoder that captures common risk factors across all outcomes.
    """
    def __init__(self, n_features, config=CONFIG):
        super().__init__()
        dims = config["hidden_dims"]
        
        # Shared encoder — learns joint representation
        layers = []
        in_dim = n_features
        for h in dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(config["dropout"])]
            in_dim = h
        layers += [nn.Linear(in_dim, config["shared_dim"]), nn.ReLU()]
        self.shared_encoder = nn.Sequential(*layers)
        
        # Task-specific heads
        self.heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(config["shared_dim"], 32),
                nn.ReLU(),
                nn.Dropout(config["dropout"]/2),
                nn.Linear(32, 1)
            ) for task in TARGETS.keys()
        })
        
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def forward(self, x):
        shared = self.shared_encoder(x)
        outputs = {task: self.heads[task](shared).squeeze(-1) for task in TARGETS.keys()}
        return outputs, shared


def train_multitask(X, Y, config=CONFIG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Training] Device: {device}")
    idx = np.arange(len(Y))
    idx_tr, idx_val = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=(Y[:,0]>0.5).astype(int))
    tr_loader  = DataLoader(MultiTaskDataset(X[idx_tr],  Y[idx_tr]),  batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(MultiTaskDataset(X[idx_val], Y[idx_val]), batch_size=config["batch_size"])
    model = SmartDaaSMultiTask(X.shape[1], config).to(device)
    print(f"[Model] Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Per-task class weights
    criteria = {}
    for i, task in enumerate(TARGETS.keys()):
        pos = Y[idx_tr, i].sum(); neg = len(idx_tr) - pos
        pw = torch.tensor([neg/max(pos,1)]).to(device)
        criteria[task] = nn.BCEWithLogitsLoss(pos_weight=pw)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["n_epochs"])
    
    history = {task: [] for task in TARGETS}
    history["train_loss"] = []; history["val_loss"] = []
    best_mean_auc, patience_ctr, best_state = 0, 0, None
    
    header = f"{'Epoch':<7} {'Loss':<10} " + " ".join(f"{k[:8]:<10}" for k in TARGETS)
    print(f"\n{header}"); print("-"*70)
    
    for epoch in range(1, config["n_epochs"]+1):
        model.train()
        tl = []
        for Xb, Yb in tr_loader:
            Xb, Yb = Xb.to(device), Yb.to(device)
            optimizer.zero_grad()
            outputs, _ = model(Xb)
            loss = sum(criteria[t](outputs[t], Yb[:,i]) for i,t in enumerate(TARGETS))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl.append(loss.item())
        
        model.eval()
        vl = []; all_preds = {t:[] for t in TARGETS}; all_labs = {t:[] for t in TARGETS}
        with torch.no_grad():
            for Xb, Yb in val_loader:
                Xb, Yb = Xb.to(device), Yb.to(device)
                outputs, _ = model(Xb)
                loss = sum(criteria[t](outputs[t], Yb[:,i]) for i,t in enumerate(TARGETS))
                vl.append(loss.item())
                for i,t in enumerate(TARGETS):
                    all_preds[t].extend(torch.sigmoid(outputs[t]).cpu().numpy())
                    all_labs[t].extend(Yb[:,i].cpu().numpy())
        scheduler.step()
        
        aucs = {t: roc_auc_score(all_labs[t], all_preds[t]) for t in TARGETS}
        mean_auc = np.mean(list(aucs.values()))
        tl_m = np.mean(tl)
        history["train_loss"].append(tl_m)
        history["val_loss"].append(np.mean(vl))
        for t,a in aucs.items(): history[t].append(a)
        
        row = f"{epoch:<7} {tl_m:<10.4f} " + " ".join(f"{a:<10.4f}" for a in aucs.values())
        print(row)
        
        if mean_auc > best_mean_auc:
            best_mean_auc = mean_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= config["patience"]:
                print(f"\n[Early stopping] Best mean AUC: {best_mean_auc:.4f}")
                break
    
    model.load_state_dict(best_state)
    return model, history, idx_tr, idx_val, device


def evaluate_multitask(model, X, Y, idx_val, device):
    model.eval()
    X_val = torch.tensor(X[idx_val], dtype=torch.float32)
    all_preds = {t:[] for t in TARGETS}; all_embs = []
    with torch.no_grad():
        for i in range(0, len(X_val), 256):
            Xb = X_val[i:i+256].to(device)
            outputs, emb = model(Xb)
            for t in TARGETS: all_preds[t].extend(torch.sigmoid(outputs[t]).cpu().numpy())
            all_embs.append(emb.cpu().numpy())
    embs = np.vstack(all_embs)
    print(f"\n[Evaluation] Multi-task results:")
    print(f"  {'Task':<35} {'AUC-ROC':<12} {'AUC-PR':<12}")
    print(f"  {'-'*60}")
    aucs = {}
    for i,(t,label) in enumerate(TARGETS.items()):
        auc  = roc_auc_score(Y[idx_val,i], all_preds[t])
        aupr = average_precision_score(Y[idx_val,i], all_preds[t])
        aucs[t] = auc
        print(f"  {label:<35} {auc:<12.4f} {aupr:<12.4f}")
    print(f"\n  Mean AUC: {np.mean(list(aucs.values())):.4f}")
    print(f"  Shared embedding dim: {embs.shape[1]}")
    return all_preds, embs, aucs


def plot_multitask(history, output_path="/mnt/user-data/outputs/smartdaas_multitask_training.png"):
    tasks = list(TARGETS.keys())
    colors = ["#0072B2","#CC79A7","#009E73","#E69F00"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("SmartDaaS v2 — Multi-Task Learning", fontsize=13, fontweight="bold")
    ep = range(1, len(history["train_loss"])+1)
    axes[0].plot(ep, history["train_loss"], color="#0072B2", linewidth=2, label="Train")
    axes[0].plot(ep, history["val_loss"],   color="#CC79A7", linewidth=2, label="Val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Total Loss")
    axes[0].set_title("Combined Loss (4 tasks)"); axes[0].legend()
    axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)
    for t,c,label in zip(tasks, colors, TARGETS.values()):
        if t in history:
            axes[1].plot(ep, history[t], color=c, linewidth=2, label=label[:25])
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("AUC-ROC per task")
    axes[1].set_title("Per-Task AUC-ROC"); axes[1].set_ylim(0.5, 1.0); axes[1].legend(fontsize=8)
    axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {output_path}")


def run_multitask(patient_path="/mnt/user-data/outputs/smartdaas_synthetic_patient_table.csv"):
    print("="*70)
    print("SmartDaaS v2 — Multi-Task Learning Module")
    print("Simultaneous prediction: Interruption + Viral Failure + Adherence + Missed Visits")
    print("="*70)
    X, Y, features = prepare_data(patient_path)
    model, history, idx_tr, idx_val, device = train_multitask(X, Y)
    preds, embs, aucs = evaluate_multitask(model, X, Y, idx_val, device)
    plot_multitask(history)
    torch.save(model.state_dict(), "/mnt/user-data/outputs/smartdaas_multitask.pt")
    print(f"\n[Output] Model saved: smartdaas_multitask.pt")
    print("\n" + "="*70)
    print("Multi-task learning complete.")
    print("Shared encoder captures joint HIV risk representation across 4 outcomes.")
    print("Next: Drift detection → Facility embeddings → Causal/uplift modeling")
    print("="*70)
    return model, preds, aucs

if __name__ == "__main__":
    run_multitask()
