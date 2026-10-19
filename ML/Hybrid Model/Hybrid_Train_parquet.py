import os, json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from scripts.autoencoder.AE_train import train as ae_train
import scripts.autoencoder.AE_config as cfg

# --------- file paths ----------
# parquet file contains captured packet features
PARQUET  = "Train and Test Files/Kitsune Online Benchmarks/streamed_ArpMitM_onlinereplay.parquet"

# csv file contains one label per packet
# label values use 0 for benign and 1 for attack
LABELS   = "NetworkIngestion/ARP MitM_labels.csv"

# output files for saved model and evaluation artifacts
MODEL          = "scripts/rf_model.joblib"
METRICS        = "Results and Reports/rf_metrics.json"
THRESHOLDS_CSV = "Results and Reports/rf_thresholds.csv"
BESTF1_JSON    = "Results and Reports/rf_metrics_best_f1.json"
TARGETREC_JSON = "Results and Reports/rf_metrics_target_recall.json"

# --------- load parquet features ----------
# read feature table from parquet and convert to pandas
print("Loading parquet features...")
table    = pq.read_table(PARQUET)
df_feat  = table.to_pandas()
print(f"Parquet: {len(df_feat):,} rows, {len(df_feat.columns)} columns")

# --------- load and align labels ----------
# load all labels from csv
print("Loading labels...")
labels_all = pd.read_csv(LABELS, index_col=0)["x"].to_numpy(dtype=np.int32)
print(f"Labels: {len(labels_all):,} total")

# packet_num in the feature file starts at 1
# use it to match each feature row to its correct label
pkt_nums       = df_feat["packet_num"].to_numpy()
labels_aligned = labels_all[pkt_nums - 1]   # packet_num is 1-based

# --------- build unified dataframe ----------
# final layout puts label in first column and features after it
feature_cols = [c for c in df_feat.columns if c != "packet_num"]
df_clean = pd.concat([
    pd.Series(labels_aligned, name="label", dtype=np.int32),
    df_feat[feature_cols].reset_index(drop=True),
], axis=1)

# reduce feature memory usage before training
df_clean[feature_cols] = df_clean[feature_cols].astype(np.float32)
print(f"Rows: {len(df_clean):,}, Columns: {len(df_clean.columns)}")

# free intermediate data before splitting
del df_feat, labels_aligned, pkt_nums   # free before split

# --------- create train and test splits ----------
# first column is the label column
df        = df_clean
label_col = df.columns[0]   # "label"

# reserve first 1 million rows for AE training
# this assumes early rows are benign traffic
ae_train_partition = 1_000_000   # first 1M rows for AE
print(f"AE train partition size: {ae_train_partition:,} rows")

# verify AE training rows are benign only
ae_train_plus_labels = df.iloc[:ae_train_partition]
for row in ae_train_plus_labels.itertuples():
    if row[1] != 0:
        print("ERROR: Non-benign row found in AE train set.")
        break

# remove label column before writing AE training data
AE_train_df = df.iloc[:ae_train_partition, 1:]
print(f"AE train shape: {AE_train_df.shape}, columns: {len(AE_train_df.columns)}")
print(f"AE train total benign rows: {len(AE_train_df):,}")

# --------- create RF training set ----------
# use data up to the 30 percent attack cutoff for RF training
# keep later rows for testing
attack_indices   = df.index[df[label_col] == 1].to_list()
total_attacks    = len(attack_indices)
attack_train_size = int(0.30 * total_attacks)
cutoff_idx       = attack_indices[attack_train_size]
rf_train_df      = df.loc[:cutoff_idx]

# everything after the RF training rows becomes test data
test_df = df.drop(rf_train_df.index)
print(f"TEST SET shape: {test_df.shape}, columns: {len(test_df.columns)}")

# --------- split features and labels ----------
# training features and labels for RF
X_train = rf_train_df.iloc[:, 1:]
y_train = rf_train_df.iloc[:, 0]
print(f"RF TRAIN: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"RF TRAIN: columns: {len(X_train.columns)}")
print(f"RF TRAIN: Total attacks in y_train: {y_train.sum()}")
print(f"RF TRAIN: Total benign in y_train: {(y_train == 0).sum()}")

# test features and labels for evaluation
X_test = test_df.iloc[:, 1:]
y_test = test_df.iloc[:, 0]
print(f"TEST SET: X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print(f"TEST SET: columns: {len(X_test.columns)}")
del test_df, rf_train_df, df, df_clean   # free large raw dataframes before model fitting

# --------- save AE training file ----------
# write AE training features to the configured csv path
AE_train_df.to_csv(cfg.TRAIN_FILE, index=False)
print(f"Saved AE train file to: {cfg.TRAIN_FILE}")

# --------- train autoencoder ----------
# AE learns normal traffic behavior from benign-only data
ae_train()
print("AE training complete.")
del AE_train_df, ae_train_plus_labels   # free memory before RF training

# --------- RF training ----------
# binary mode predicts benign or attack
TASK = "binary"  # ("binary" | "multiclass")
print(f"Train size: {len(X_train):,} rows   Test size: {len(X_test):,} rows")
print(f"Train features: {len(X_train.columns)}   Test features: {len(X_test.columns)}")

# save test files for downstream hybrid evaluation script
X_test.to_csv("Train and Test Files/X_test.csv", index=False)
y_test.to_csv("Train and Test Files/y_test.csv", index=False)

# build preprocessing pipeline
# current setup keeps numeric and boolean columns as they are
num_cols = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
pre = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)

# random forest uses many decision trees and combines their outputs
# class_weight helps with class imbalance in binary mode
rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    class_weight="balanced" if TASK == "binary" else None,
    n_jobs=-1,
    random_state=42,
)

# fit preprocessor on training data and transform both splits
Xtr = pre.fit_transform(X_train)
Xte = pre.transform(X_test)
print(f"Transformed shapes -> X_train: {Xtr.shape}, X_test: {Xte.shape}")

# train the random forest classifier
rf.fit(Xtr, y_train)

# save both the preprocessor and trained model
dump({"pre": pre, "rf": rf, "num_cols": num_cols}, MODEL)
print(f"Done: Saved model bundle -> {MODEL}")
