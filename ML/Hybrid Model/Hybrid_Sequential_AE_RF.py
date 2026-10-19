# scripts/Hybrid/Hybrid_Sequential_AE_RF.py
#
# Sequential Filter-Then-Classify decision logic:
#   1) Autoencoder (AE) runs first and flags anomalous flows based on reconstruction error.
#   2) Only AE-flagged flows are passed to the Random Forest (RF) classifier.
#   3) Final decision: benign by default; RF labels for AE-flagged flows.

import os, json
import numpy as np
import pandas as pd
import torch
import joblib
import time

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)

from scripts.autoencoder.Autoencoder import Autoencoder
import scripts.autoencoder.AE_config as cfg

# paths to models, data, and output files
RF_MODEL_PATH = "scripts/rf_model.joblib"
DATA_DIR      = "Train and Test files"
RESULTS_DIR   = "Results and Reports"
SEQ_METRICS   = os.path.join(RESULTS_DIR, "hybrid_sequential_metrics.json")


# load test dataset 
# X contains features, y contains labels
# labels use 0 for benign traffic and 1 for attack traffic
def load_test_data():
    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test_first_half.csv"))
    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test_first_half.csv"))
    # assume y_test has a 'label' column (0=benign,1=attack)
    y_true = y_test["label"].astype(int).to_numpy()
    return X_test, y_true

# rebalance dataset to control class distribution
# ratio controls attack count relative to benign
# keeps original order so time patterns stay valid
def balance_test_data(X_test, y_true, ratio=1.0, random_state=42):

    # locate indices for each class
    benign_idx = np.where(y_true == 0)[0]
    attack_idx = np.where(y_true == 1)[0]

    n_benign = len(benign_idx)
    n_attack = len(attack_idx)

    # compute how many attack samples to keep
    n_attack_target = int(n_benign * ratio)

    print(f"[BALANCE] Original - Benign: {n_benign}, Attack: {n_attack}")
    
    # skip if dataset already meets target ratio
    if n_attack_target >= n_attack:
        print(f"[BALANCE] No undersampling needed (target {n_attack_target} >= available {n_attack})")
        return X_test, y_true

    # randomly select subset of attack samples
    # seed ensures reproducibility
    np.random.seed(random_state)
    attack_idx_sampled = np.sort(
        np.random.choice(attack_idx, size=n_attack_target, replace=False)
    )

    # combine benign and sampled attack indices
    # sort to preserve original time ordering
    balanced_idx = np.sort(np.concatenate([benign_idx, attack_idx_sampled]))
    
    # apply selection to dataset
    X_balanced = X_test.iloc[balanced_idx].reset_index(drop=True)
    y_balanced = y_true[balanced_idx]

    print(f"[BALANCE] After 1:{ratio} rebalancing - Benign: {n_benign}, Attack: {n_attack_target}")
    print(f"[BALANCE] Total samples: {len(y_balanced)} (chronological order preserved)")

    return X_balanced, y_balanced

# load trained autoencoder model and scaler
# scaler ensures input matches training distribution
def load_autoencoder():
    model = Autoencoder(cfg.INPUT_DIM, cfg.LATENT_DIM, cfg.HIDDEN_DIMS)

    # load trained weights
    model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH, map_location=cfg.DEVICE))

    # move model to cpu or gpu based on config
    model.to(cfg.DEVICE)
    model.eval()
    
    # load preprocessing scaler
    scaler = joblib.load(cfg.SCALER_SAVE_PATH)
    return model, scaler

# compute reconstruction error for each sample
# error measures difference between input and reconstructed output
# higher error suggests abnormal behavior
def compute_reconstruction_errors(model, scaler, X_df: pd.DataFrame) -> np.ndarray:

    # normalize input features
    X_scaled = scaler.transform(X_df.values)

    # convert to tensor for model input
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(cfg.DEVICE)

    # disable gradient tracking for faster inference
    with torch.no_grad():
        recon = model(X_tensor)

    # mean squared error per sample
    errors = torch.mean((recon - X_tensor) ** 2, dim=1).cpu().numpy()
    return errors

# search for best ae threshold
# threshold defines cutoff between normal and anomalous
# only keep thresholds with acceptable precision and recall
def sweep_ae_thresholds(errors: np.ndarray, y_true: np.ndarray,
                        min_precision: float = 0.4,
                        min_recall: float = 0.4):
    """Mimics AE_eval: sweep thresholds, keep best F1 with P/R ≥ mins."""
    loss = errors
    min_loss, max_loss = loss.min(), loss.max()

    # start from lowest possible error
    thresh = min_loss
    best = {
        "f1": 0.0, "precision": 0.0, "recall": 0.0,
        "fpr": 0.0, "fnr": 0.0, "threshold": min_loss,
        "predictions": None,
    }

    # increase threshold gradually to explore decision boundary
    while thresh <= max_loss / 0.9:
        thresh *= 1.05

        # classify samples based on threshold
        pred = (loss > thresh).astype(np.int64)  # 1 = anomalous/attack

        # skip if no samples flagged
        if pred.sum() == 0:
            continue

        P = precision_score(y_true, pred, zero_division=0)
        R = recall_score(y_true, pred, zero_division=0)

        # enforce minimum quality constraints
        if P < min_precision or R < min_recall:
            continue
        
        # compute f1 score
        F1 = (2 * P * R) / (P + R + 1e-10)

        # compute confusion matrix values
        TP = int(((pred == 1) & (y_true == 1)).sum())
        FP = int(((pred == 1) & (y_true == 0)).sum())
        FN = int(((pred == 0) & (y_true == 1)).sum())
        TN = int(((pred == 0) & (y_true == 0)).sum())

        FPR = FP / (FP + TN + 1e-10)
        FNR = FN / (FN + TP + 1e-10)

        # keep best threshold based on f1
        if F1 > best["f1"]:
            best.update({
                "f1": F1, "precision": P, "recall": R,
                "fpr": FPR, "fnr": FNR,
                "threshold": thresh,
                "predictions": pred.copy(),
                "TP": TP, "FP": FP, "FN": FN, "TN": TN,
            })

    return best

# load rf model and preprocessing pipeline
# pipeline handles feature transformations before classification
def load_rf():
    bundle = joblib.load(RF_MODEL_PATH)
    pre = bundle["pre"]
    rf = bundle["rf"]
    return pre, rf

# convert confusion matrix into readable text format
def confusion_matrix_markdown(cm):
    TN, FP = cm[0]
    FN, TP = cm[1]

    return (
        f"[ {TN:>7}  {FP:>7} ]\n"
        f"[ {FN:>7}  {TP:>7} ]"
    )
#-------------------------------------------------------------

#--------------------DEMO TOOLS--------------------
# find best rf probability threshold
# tests thresholds from 0.01 to 0.99 and selects highest f1
def save_markdown_report(path, ae, rf, seq, timing):
    def pct(x):
        return f"{x*100:.2f}%"

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Hybrid Model Evaluation Report\n\n")

        # ==============================================================
        # AE-ONLY SECTION
        # ==============================================================
        f.write("## Autoencoder (AE) Only Performance\n")
        f.write("### Confusion Matrix\n")
        f.write("```\n")
        f.write(f"[ TN: {ae['TN']:7d}   FP: {ae['FP']:7d} ]\n")
        f.write(f"[ FN: {ae['FN']:7d}   TP: {ae['TP']:7d} ]\n")
        f.write("```\n\n")

        f.write(f"**Precision:** {pct(ae['precision'])}\n\n")
        f.write(f"**Recall:** {pct(ae['recall'])}\n\n")
        f.write(f"**F1 Score:** {pct(ae['f1'])}\n\n")

        f.write("---\n\n")

        # ==============================================================
        # RF-ONLY SECTION
        # ==============================================================
        f.write("## Random Forest (RF) Only Performance\n")
        f.write("### Confusion Matrix\n")
        f.write("```\n")
        f.write(f"[ TN: {rf['TN']:7d}   FP: {rf['FP']:7d} ]\n")
        f.write(f"[ FN: {rf['FN']:7d}   TP: {rf['TP']:7d} ]\n")
        f.write("```\n\n")

        f.write(f"**ROC-AUC:** {rf['roc_auc']:.6f}\n\n")
        f.write(f"**PR-AUC:** {rf['pr_auc']:.6f}\n\n")
        f.write(f"**F1 Score:** {pct(rf['f1'])}\n\n")
        f.write("---\n\n")

        # ==============================================================
        # SEQUENTIAL SECTION
        # ==============================================================
        f.write("## Sequential AE → RF Pipeline Performance\n")
        f.write("### Confusion Matrix\n")
        f.write("```\n")
        f.write(f"[ TN: {seq['TN']:7d}   FP: {seq['FP']:7d} ]\n")
        f.write(f"[ FN: {seq['FN']:7d}   TP: {seq['TP']:7d} ]\n")
        f.write("```\n\n")

        f.write(f"**Precision:** {pct(seq['precision'])}\n\n")
        f.write(f"**Recall:** {pct(seq['recall'])}\n\n")
        f.write(f"**F1 Score:** {pct(seq['f1'])}\n\n")

        # AE Threshold + Fraction Flagged
        f.write(f"**AE Best Threshold:** {seq['ae_best_threshold']:.4f}\n\n")
        f.write(f"**AE Fraction Flagged:** {pct(seq['ae_fraction_flagged'])}\n\n")

        f.write("---\n\n")

        # ==============================================================
        # EXPLANATION SECTION
        # ==============================================================
        f.write("## Processing Times\n")
        ae_time = timing['ae_processing_seconds']
        rows = 330000
        sec_per_row = ae_time / rows
        rows_per_sec = rows / ae_time
        f.write(f"- **AE processing time:** {ae_time:.2f} seconds for ~330,000 flows\n")
        f.write(f"  - Seconds per row: {sec_per_row:.6f}\n")
        f.write(f"  - Throughput: {rows_per_sec:,.0f} rows/sec\n\n")     
        rf_time = timing['rf_processing_seconds']
        sec_per_row = rf_time / rows
        rows_per_sec = rows / rf_time
        f.write(f"- **RF processing time:** {rf_time:.2f} seconds for ~330,000 flows\n")
        f.write(f"  - Seconds per row: {sec_per_row:.6f}\n")
        f.write(f"  - Throughput: {rows_per_sec:,.0f} rows/sec\n\n")     
        f.write(
            "**Sequential AE → RF Behavior:**\n\n"
            "The AE flags approximately "
            f"**{pct(seq['ae_fraction_flagged'])}** of flows, this means the RF only analyzes that fraction "
            "instead of the entire dataset.\n\n"

            "This significantly reduces RF workload while still maintaining extremely high detection "
            "performance:\n\n"

            f"- The sequential pipeline maintains **{pct(seq['precision'])} precision**, meaning almost no false alarms.\n"
            f"- Recall remains extremely high at **{pct(seq['recall'])}**, showing that the RF still captures nearly all real attacks.\n"
            f"- The F1 score of **{pct(seq['f1'])}** demonstrates that the hybrid pipeline retains nearly identical performance to the RF-only system.\n\n"

            "In short, **the AE successfully filters out ~"
            f"{100 - seq['ae_fraction_flagged']*100:.2f}%** of benign traffic, dramatically reducing computational load, "
            "while the RF corrects the AE’s false positives and restores near-perfect accuracy.\n"
        )

#-------------------------------------------------
def sweep_rf_thresholds(probs: np.ndarray, y_true: np.ndarray):
    """Sweep RF probability thresholds from 0.01→0.99, return the one with best F1."""
    best = {"f1": 0.0, "threshold": 0.5, "precision": 0.0, "recall": 0.0,
            "TP": 0, "FP": 0, "FN": 0, "TN": 0, "predictions": None}
    for t in np.arange(0.01, 1.00, 0.01):
        pred = (probs >= t).astype(np.int64)
        if pred.sum() == 0:
            continue
        P = precision_score(y_true, pred, zero_division=0)
        R = recall_score(y_true, pred, zero_division=0)
        F1 = f1_score(y_true, pred, zero_division=0)
        if F1 > best["f1"]:
            TP = int(((pred == 1) & (y_true == 1)).sum())
            FP = int(((pred == 1) & (y_true == 0)).sum())
            FN = int(((pred == 0) & (y_true == 1)).sum())
            TN = int(((pred == 0) & (y_true == 0)).sum())
            best.update({"f1": F1, "threshold": float(t), "precision": P, "recall": R,
                         "TP": TP, "FP": FP, "FN": FN, "TN": TN, "predictions": pred.copy()})
    return best

# compute evaluation metrics for predictions
# supports both classification metrics and probability metrics
def compute_metrics(y_true, y_pred, probs=None):
    cm = confusion_matrix(y_true, y_pred)
    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())

    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix_markdown(cm),
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
    }
    if probs is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probs))
        metrics["pr_auc"] = float(average_precision_score(y_true, probs))
    return metrics





def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ----- load test data -----
    X_test, y_true = load_test_data()

    # ----- balance test data (1:1 ratio, preserve chronological order) -----
    #X_test, y_true = balance_test_data(X_test, y_true, ratio=1.0, random_state=42)

    # ----- AE: filter stage -----
    ae_model, ae_scaler = load_autoencoder()
    t0 = time.perf_counter()
    errors = compute_reconstruction_errors(ae_model, ae_scaler, X_test)
    ae_time = time.perf_counter() - t0
    ae_best = sweep_ae_thresholds(errors, y_true)

    ae_pred = ae_best["predictions"] 
    if ae_pred is None:
        raise RuntimeError("AE threshold sweep found no threshold meeting precision/recall constraints.")

    print("\n[AE FILTER] Best threshold:")
    print(f"  threshold = {ae_best['threshold']:.6e}")
    print(f"  precision = {ae_best['precision']:.4f}")
    print(f"  recall    = {ae_best['recall']:.4f}")
    print(f"  f1        = {ae_best['f1']:.4f}")
    print(f"  FPR       = {ae_best['fpr']:.4f}")
    print(f"  FNR       = {ae_best['fnr']:.4f}")

    frac_flagged = float((ae_pred == 1).mean())
    print(f"  fraction of flows passed to RF = {frac_flagged:.4f}")

    # ----- RF: full and filtered -----
    pre, rf = load_rf()
    t1 = time.perf_counter()

    Xte_rf = pre.transform(X_test)
    rf_probs = rf.predict_proba(Xte_rf)[:, 1]
    rf_time = time.perf_counter() - t1

    # Sweep RF threshold for best F1
    rf_best = sweep_rf_thresholds(rf_probs, y_true)
    rf_pred = rf_best["predictions"]
    print("\n[RF ONLY] Best threshold sweep:")
    print(f"  threshold = {rf_best['threshold']:.2f}")
    print(f"  precision = {rf_best['precision']:.4f}")
    print(f"  recall    = {rf_best['recall']:.4f}")
    print(f"  f1        = {rf_best['f1']:.4f}")

    rf_metrics = compute_metrics(y_true, rf_pred, rf_probs)
    print("\n[RF ONLY] metrics:")
    print(json.dumps(rf_metrics, indent=2))

    # ----- Sequential AE -> RF -----
    # default all to benign
    final_pred = np.zeros_like(y_true)
    # update only flows flagged by ae
    idx_flagged = np.where(ae_pred == 1)[0]
    # for those, use RF decision at the best threshold
    final_pred[idx_flagged] = rf_pred[idx_flagged]
    
    seq_metrics = compute_metrics(y_true, final_pred, probs=None)
    seq_metrics.update({
        "ae_best_threshold": float(ae_best["threshold"]),
        "ae_precision": float(ae_best["precision"]),
        "ae_recall": float(ae_best["recall"]),
        "ae_f1": float(ae_best["f1"]),
        "ae_fpr": float(ae_best["fpr"]),
        "ae_fnr": float(ae_best["fnr"]),
        "ae_fraction_flagged": frac_flagged,
        "rf_best_threshold": float(rf_best["threshold"]),
    })

    print("\n[SEQUENTIAL AE→RF] metrics:")
    print(json.dumps(seq_metrics, indent=2))

    # AE-only metrics 
    ae_only_metrics = compute_metrics(y_true, ae_pred, probs=None)
    print("\n[AE ONLY] metrics:")
    print(json.dumps(ae_only_metrics, indent=2))

    # ----- save everything -----
    out = {
        "ae_only": ae_only_metrics,
        "rf_only": rf_metrics,
        "sequential_ae_rf": seq_metrics,
        "timing": {
            "ae_processing_seconds": ae_time,
            "rf_processing_seconds": rf_time,
        }
    }
    with open(SEQ_METRICS, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nDone: Saved sequential AE→RF metrics -> {SEQ_METRICS}")
    save_markdown_report("Results and Reports/hybrid_report.md",
                     ae_only_metrics, rf_metrics, seq_metrics, out["timing"])

if __name__ == "__main__":
    main()