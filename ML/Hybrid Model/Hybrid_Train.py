import os, json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from scripts.autoencoder.AE_train import train as ae_train
import scripts.autoencoder.AE_config as cfg
# --------- paths ----------

DATA   = "Train and Test Files/Kitsune Offline Benchmarks/Mirai_replay_features_labeled115.csv"
MODEL  = "scripts/rf_model.joblib"
METRICS= "Results and Reports/rf_metrics.json"
SCORES = "Results and Reports/rf_scores.csv"
THRESHOLDS_CSV = "Results and Reports/rf_thresholds.csv"
BESTF1_JSON    = "Results and Reports/rf_metrics_best_f1.json"
TARGETREC_JSON = "Results and Reports/rf_metrics_target_recall.json"

# --------- Load dataset (float32 halves RAM vs default float64) ----------
print("Loading processed dataset...")
df = pd.read_csv(DATA)
float_cols = df.select_dtypes("float64").columns
df[float_cols] = df[float_cols].astype(np.float32)
print(f"Rows: {len(df):,}, Columns: {len(df.columns)}")

# --------- CREATE TRAIN AND TEST SPLITS ----------
label_col = df.columns[0]
ae_train_partition = 1_000_000  # First 1,000,000 rows for the AE (must be benign-only)
print(f"AE train partition size (benign only): {ae_train_partition:,} rows")

# Vectorised benign check — much faster than itertuples on 1M rows
if (df.iloc[:ae_train_partition, 0] != 0).any():
    print("ERROR: Non-benign row found in AE train set.")

# AE train slice (no label column)
AE_train_df = df.iloc[:ae_train_partition, 1:]
print(f"AE train shape: {AE_train_df.shape}")

# RF train: all rows up to the 30th-percentile attack index
attack_indices = df.index[df[label_col] == 1].to_list()
total_attacks = len(attack_indices)
cutoff_idx = attack_indices[int(0.30 * total_attacks)]
rf_train_df = df.loc[:cutoff_idx]

# Test set: everything after the RF train cutoff
test_df = df.iloc[cutoff_idx + 1:].reset_index(drop=True)
del df  # free the full dataset — no longer needed
print(f"TEST SET shape: {test_df.shape}")

# Split X/Y
X_train = rf_train_df.iloc[:, 1:]
y_train = rf_train_df.iloc[:, 0]
del rf_train_df
print(f"RF TRAIN: {X_train.shape}  attacks={int(y_train.sum())}  benign={int((y_train==0).sum())}")

X_test = test_df.iloc[:, 1:]
y_test = test_df.iloc[:, 0]
del test_df
print(f"TEST SET: {X_test.shape}")

# --------- Save AE train file and run AE training ----------
AE_train_df.to_csv(cfg.TRAIN_FILE, index=False)
del AE_train_df
print(f"Saved AE train file to: {cfg.TRAIN_FILE}")

ae_train()
print("AE training complete.")

# --------- Save test files ----------
X_test.to_csv("Train and Test Files/X_test.csv", index=False)
y_test.to_csv("Train and Test Files/y_test.csv", index=False)

# --------- RF TRAINING ----------
TASK = "binary"  # ("binary" | "multiclass")
print(f"Train size: {len(X_train):,} rows   Test size: {len(X_test):,} rows")

num_cols = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()

pre = ColumnTransformer(
    transformers=[("num", "passthrough", num_cols)],
    remainder="drop",
    verbose_feature_names_out=False,
)

rf = RandomForestClassifier(
    n_estimators=400, 
    max_depth=None,
    class_weight="balanced" if TASK == "binary" else None,
    n_jobs=-1,
    random_state=42,
)

Xtr = pre.fit_transform(X_train)
Xte = pre.transform(X_test)
print(f"Transformed shapes -> X_train: {Xtr.shape}, X_test: {Xte.shape}")

rf.fit(Xtr, y_train)

dump({"pre": pre, "rf": rf, "num_cols": num_cols}, MODEL)
print(f"Done: Saved model bundle -> {MODEL}")