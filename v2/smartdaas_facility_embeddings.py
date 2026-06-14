import os
"""
smartdaas_facility_embeddings.py — SmartDaaS v2 Facility Intelligence Module
Facility embeddings and hierarchical context learning
Author: Lakshmi Kalyani Chinthala, SmartDaaS LLC
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Facility structural features
FACILITY_FEATURES = [
    "n_patients_enrolled", "staff_ratio", "distance_km_median",
    "supply_chain_score", "rural_flag", "security_index", "transport_access",
]

# Patient features (snapshot level)
PATIENT_FEATURES = [
    "age_at_art_start", "sex_female", "cd4_at_art_start", "most_recent_cd4",
    "cd4_improvement", "who_stage_start", "weight_at_start", "days_to_art",
    "missed_visit_rate", "mean_refill_gap_days", "n_regimen_changes",
    "viral_suppressed_last", "n_poor_adherence_visits",
]

CONFIG = {
    "facility_emb_dim" : 16,    # facility embedding dimension
    "patient_emb_dim"  : 32,    # patient feature embedding
    "combined_dim"     : 48,    # patient + facility combined
    "hidden_dim"       : 64,
    "dropout"          : 0.2,
    "batch_size"       : 128,
    "n_epochs"         : 25,
    "lr"               : 1e-3,
    "weight_decay"     : 1e-4,
    "patience"         : 6,
}


# ════════════════════════════════════════════════════════════════════════════
# 1. DATA PREPARATION
# ════════════════════════════════════════════════════════════════════════════

def prepare_data(patient_path, facility_path):
    print("[Data] Loading tables...")
    patients  = pd.read_csv(patient_path)
    facilities = pd.read_csv(facility_path)

    # Encode facility categorical features
    le_level = LabelEncoder()
    le_type  = LabelEncoder()
    le_partner = LabelEncoder()
    le_country = LabelEncoder()
    facilities["facility_level_enc"]  = le_level.fit_transform(facilities["facility_level"])
    facilities["facility_type_enc"]   = le_type.fit_transform(facilities["facility_type"])
    facilities["partner_enc"]         = le_partner.fit_transform(facilities["partner"])
    facilities["country_enc"]         = le_country.fit_transform(facilities["country"])

    # Facility index map
    facility_ids = facilities["facility_id"].tolist()
    fac_idx_map = {fid: i for i, fid in enumerate(facility_ids)}
    n_facilities = len(facility_ids)

    # Facility structural feature matrix
    fac_struct_cols = FACILITY_FEATURES + ["facility_level_enc","facility_type_enc","partner_enc","country_enc"]
    fac_avail = [c for c in fac_struct_cols if c in facilities.columns]
    fac_imp = SimpleImputer(strategy="median")
    fac_scaler = StandardScaler()
    F_raw = fac_imp.fit_transform(facilities[fac_avail].fillna(0))
    F = fac_scaler.fit_transform(F_raw).astype(np.float32)

    # Patient features
    pat_avail = [f for f in PATIENT_FEATURES if f in patients.columns]
    pat_imp = SimpleImputer(strategy="median")
    pat_scaler = StandardScaler()
    X_raw = pat_imp.fit_transform(patients[pat_avail].fillna(0))
    X = pat_scaler.fit_transform(X_raw).astype(np.float32)

    # Facility index per patient
    fac_indices = patients["facility_id"].map(fac_idx_map).fillna(0).astype(int).values

    # Labels
    y = patients["interrupted"].values.astype(np.float32)

    print(f"[Data] Patients: {len(X):,}  Facilities: {n_facilities}")
    print(f"[Data] Patient features: {X.shape[1]}  Facility features: {F.shape[1]}")
    print(f"[Data] Interruption rate: {y.mean()*100:.1f}%")
    print(f"[Data] Patients per facility: {len(X)/n_facilities:.0f} avg")

    return X, F, fac_indices, y, facility_ids, facilities


# ════════════════════════════════════════════════════════════════════════════
# 2. HIERARCHICAL MODEL: PATIENT + FACILITY CONTEXT
# ════════════════════════════════════════════════════════════════════════════

class FacilityContextDataset(Dataset):
    def __init__(self, X, F, fac_idx, y):
        self.X       = torch.tensor(X,       dtype=torch.float32)
        self.F       = torch.tensor(F,       dtype=torch.float32)
        self.fac_idx = torch.tensor(fac_idx, dtype=torch.long)
        self.y       = torch.tensor(y,       dtype=torch.float32)

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.F[self.fac_idx[i]], self.fac_idx[i], self.y[i]


class SmartDaaSHierarchical(nn.Module):
    """
    Hierarchical model: patient features + learned facility embeddings.

    Architecture:
        Patient features → Patient encoder (32-dim)
                                              ↘
                                               Concat → Combined (48-dim) → Prediction
                                              ↗
        Facility ID → Facility embedding (16-dim)
        Facility structural features → Facility encoder (16-dim) → Initialise embedding

    The facility embedding is LEARNED during training — it captures
    everything about a facility that influences patient outcomes,
    beyond just the structural features we can measure:
        - Quality of counselling
        - Staff retention and morale
        - Local community trust
        - Operational efficiency
        - Microgeographic factors

    This is programme intelligence — not just patient ML.
    """
    def __init__(self, n_patients_feat, n_facility_feat, n_facilities, config=CONFIG):
        super().__init__()

        # Learned facility embedding table (one vector per facility)
        self.facility_embedding = nn.Embedding(n_facilities, config["facility_emb_dim"])

        # Facility structural encoder (initialise embeddings with structural info)
        self.facility_encoder = nn.Sequential(
            nn.Linear(n_facility_feat, 32),
            nn.ReLU(),
            nn.Linear(32, config["facility_emb_dim"]),
        )

        # Patient encoder
        self.patient_encoder = nn.Sequential(
            nn.Linear(n_patients_feat, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(64, config["patient_emb_dim"]),
            nn.ReLU(),
        )

        # Combined prediction head
        combined = config["patient_emb_dim"] + config["facility_emb_dim"]
        self.predictor = nn.Sequential(
            nn.Linear(combined, config["hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden_dim"], 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def forward(self, x_patient, x_facility_struct, fac_idx):
        # Patient representation
        pat_emb = self.patient_encoder(x_patient)

        # Facility representation: learned embedding + structural encoding
        learned_emb   = self.facility_embedding(fac_idx)
        structural_emb = self.facility_encoder(x_facility_struct)
        fac_emb = learned_emb + structural_emb   # combine both signals

        # Concatenate patient + facility context
        combined = torch.cat([pat_emb, fac_emb], dim=-1)

        logit = self.predictor(combined).squeeze(-1)
        return logit, pat_emb, fac_emb


# ════════════════════════════════════════════════════════════════════════════
# 3. TRAINING
# ════════════════════════════════════════════════════════════════════════════

def train_hierarchical(X, F, fac_indices, y, config=CONFIG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Training] Device: {device}")

    idx = np.arange(len(y))
    idx_tr, idx_val = train_test_split(idx, test_size=0.2, random_state=SEED,
                                        stratify=(y>0.5).astype(int))

    n_facilities = F.shape[0]
    tr_ds  = FacilityContextDataset(X[idx_tr],  F, fac_indices[idx_tr],  y[idx_tr])
    val_ds = FacilityContextDataset(X[idx_val], F, fac_indices[idx_val], y[idx_val])

    tr_loader  = DataLoader(tr_ds,  batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"])

    model = SmartDaaSHierarchical(X.shape[1], F.shape[1], n_facilities, config).to(device)
    print(f"[Model] Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    pos_weight = torch.tensor([(y==0).sum()/(y==1).sum()]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["n_epochs"])

    history = {"train_loss":[], "val_loss":[], "val_auc":[]}
    best_auc, patience_ctr, best_state = 0, 0, None

    print(f"\n{'Epoch':<8}{'Train Loss':<14}{'Val Loss':<14}{'Val AUC':<12}")
    print("-"*50)

    for epoch in range(1, config["n_epochs"]+1):
        model.train()
        tl = []
        for Xb, Fb, fidx, yb in tr_loader:
            Xb, Fb, fidx, yb = Xb.to(device), Fb.to(device), fidx.to(device), yb.to(device)
            optimizer.zero_grad()
            logit, _, _ = model(Xb, Fb, fidx)
            loss = criterion(logit, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl.append(loss.item())

        model.eval()
        vl, preds, labs = [], [], []
        with torch.no_grad():
            for Xb, Fb, fidx, yb in val_loader:
                Xb, Fb, fidx, yb = Xb.to(device), Fb.to(device), fidx.to(device), yb.to(device)
                logit, _, _ = model(Xb, Fb, fidx)
                vl.append(criterion(logit, yb).item())
                preds.extend(torch.sigmoid(logit).cpu().numpy())
                labs.extend(yb.cpu().numpy())

        scheduler.step()
        tl_m = np.mean(tl); vl_m = np.mean(vl)
        val_auc = roc_auc_score(labs, preds)
        history["train_loss"].append(tl_m)
        history["val_loss"].append(vl_m)
        history["val_auc"].append(val_auc)
        print(f"{epoch:<8}{tl_m:<14.4f}{vl_m:<14.4f}{val_auc:<12.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.clone() for k,v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= config["patience"]:
                print(f"\n[Early stopping] Best AUC: {best_auc:.4f}")
                break

    model.load_state_dict(best_state)
    return model, history, idx_tr, idx_val, device


# ════════════════════════════════════════════════════════════════════════════
# 4. EXTRACT AND ANALYSE FACILITY EMBEDDINGS
# ════════════════════════════════════════════════════════════════════════════

def extract_facility_embeddings(model, F, facility_ids, facilities_df, device):
    """
    Extract learned facility embeddings and analyse what they captured.
    Cluster facilities by embedding similarity — reveals which facilities
    share similar patient outcome profiles beyond structural features.
    """
    model.eval()
    F_tensor = torch.tensor(F, dtype=torch.float32).to(device)
    fac_idx  = torch.arange(len(facility_ids), dtype=torch.long).to(device)

    with torch.no_grad():
        learned = model.facility_embedding(fac_idx).cpu().numpy()
        struct  = model.facility_encoder(F_tensor).cpu().numpy()
        combined = learned + struct

    emb_df = pd.DataFrame(combined, columns=[f"fac_emb_{i}" for i in range(combined.shape[1])])
    emb_df.insert(0, "facility_id", facility_ids)

    # Merge facility metadata
    emb_df = emb_df.merge(
        facilities_df[["facility_id","country","facility_level","facility_type",
                        "partner","rural_flag","supply_chain_score","n_patients_enrolled"]],
        on="facility_id", how="left"
    )

    print(f"\n[Facility Embeddings] Shape: {combined.shape}")
    print(f"  Each facility = {combined.shape[1]}-dimensional learned representation")
    print(f"  Captures: structural features + learned outcome patterns")
    print(f"\n  [NOTE] Overfitting risk with small facility counts:")
    print(f"  With synthetic data ({len(facility_ids)} facilities), embeddings may memorize")
    print(f"  facility identity rather than generalizable patterns.")
    print(f"  For real APIN data:")
    print(f"    → Validate on held-out facilities (not just held-out patients)")
    print(f"    → Compare patient-only AUC vs patient+facility AUC")
    print(f"    → Use facility-level cross-validation")

    # Facility performance profile
    print(f"\n  Embedding statistics by country:")
    emb_cols = [c for c in emb_df.columns if c.startswith("fac_emb_")]
    emb_norm = np.linalg.norm(combined, axis=1)
    emb_df["embedding_norm"] = emb_norm

    for country in emb_df["country"].unique():
        mask = emb_df["country"] == country
        print(f"    {country:<12} n={mask.sum()}  mean norm: {emb_norm[mask].mean():.3f}")

    return emb_df, combined


# ════════════════════════════════════════════════════════════════════════════
# 5. VISUALISE FACILITY EMBEDDINGS (t-SNE)
# ════════════════════════════════════════════════════════════════════════════

def plot_facility_embeddings(emb_df, combined,
    output_path=f"{output_dir}/smartdaas_facility_embeddings.png"):

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("SmartDaaS v2 — Facility Embeddings: Learned Programme Intelligence",
                 fontsize=13, fontweight="bold")

    # t-SNE reduction to 2D
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=min(15, len(combined)-1))
    coords = tsne.fit_transform(combined)

    country_colors = {c: plt.cm.tab10(i) for i,c in enumerate(emb_df["country"].unique())}
    level_markers  = {"Primary":"o","Secondary":"s","Tertiary":"^"}
    rural_colors   = {0:"#0072B2", 1:"#E69F00"}

    # Panel A: by country
    ax = axes[0]
    for country in emb_df["country"].unique():
        mask = emb_df["country"] == country
        ax.scatter(coords[mask,0], coords[mask,1], c=[country_colors[country]],
                   label=country, s=80, alpha=0.8, edgecolors="white", linewidths=0.5)
    ax.set_title("A. Facility Embeddings by Country")
    ax.legend(fontsize=7, loc="best")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Panel B: by facility level
    ax = axes[1]
    for level, marker in level_markers.items():
        mask = emb_df["facility_level"] == level
        if mask.sum() > 0:
            ax.scatter(coords[mask,0], coords[mask,1], marker=marker,
                       c="#0072B2", label=level, s=80, alpha=0.8,
                       edgecolors="white", linewidths=0.5)
    ax.set_title("B. Facility Embeddings by Level")
    ax.legend(fontsize=8)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Panel C: by supply chain score (continuous)
    ax = axes[2]
    sc = ax.scatter(coords[:,0], coords[:,1],
                    c=emb_df["supply_chain_score"], cmap="RdYlGn",
                    s=80, alpha=0.85, edgecolors="white", linewidths=0.5,
                    vmin=0.4, vmax=1.0)
    plt.colorbar(sc, ax=ax, label="Supply Chain Score")
    ax.set_title("C. Facility Embeddings by Supply Chain Score")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {output_path}")


# ════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ════════════════════════════════════════════════════════════════════════════

def run_facility_embeddings(
    patient_path="/mnt/user-data/outputs/smartdaas_synthetic_patient_table.csv",
    facility_path="/mnt/user-data/outputs/smartdaas_synthetic_facility_table.csv",
):
    print("="*70)
    print("SmartDaaS v2 — Facility Embeddings Module")
    print("Hierarchical patient + facility context learning")
    print("="*70)

    X, F, fac_indices, y, facility_ids, facilities_df = prepare_data(patient_path, facility_path)
    model, history, idx_tr, idx_val, device = train_hierarchical(X, F, fac_indices, y)

    # Evaluate
    model.eval()
    X_val = torch.tensor(X[idx_val], dtype=torch.float32)
    F_tensor = torch.tensor(F, dtype=torch.float32)
    fac_idx_val = torch.tensor(fac_indices[idx_val], dtype=torch.long)

    preds = []
    with torch.no_grad():
        for i in range(0, len(X_val), 256):
            Xb = X_val[i:i+256].to(device)
            Fb = F_tensor[fac_idx_val[i:i+256]].to(device)
            fi = fac_idx_val[i:i+256].to(device)
            logit, _, _ = model(Xb, Fb, fi)
            preds.extend(torch.sigmoid(logit).cpu().numpy())

    final_auc = roc_auc_score(y[idx_val], preds)
    print(f"\n[Evaluation] Final AUC-ROC: {final_auc:.4f}")
    print(f"  (Includes facility context — patient + programme intelligence)")

    # Extract embeddings
    emb_df, combined = extract_facility_embeddings(model, F, facility_ids, facilities_df, device)

    # Plot
    plot_facility_embeddings(emb_df, combined)

    # Save
    emb_df.to_csv(f"{output_dir}/smartdaas_facility_embeddings.csv", index=False)
    torch.save(model.state_dict(), f"{output_dir}/smartdaas_hierarchical.pt")

    print("\n" + "="*70)
    print(f"Facility embeddings complete. AUC: {final_auc:.4f}")
    print(f"Facility embedding dim: {combined.shape[1]}")
    print("\nWhat the embeddings captured:")
    print("  → Structural: rural/urban, supply chain, staff ratio, distance")
    print("  → Learned: outcome patterns beyond what we can measure directly")
    print("  → Geographic: country and regional clustering visible in t-SNE")
    print("\nNext: Causal / uplift modeling (THE BIG ONE)")
    print("="*70)

    return model, emb_df, final_auc

if __name__ == "__main__":
    run_facility_embeddings()
