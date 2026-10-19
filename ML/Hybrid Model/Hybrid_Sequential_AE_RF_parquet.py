#Loads the test data from the X_test.csv
#The logic is the same as Hybrid_Sequential_AE_RF.py
#1-Only AE flagged flows are passed to the random forest
#2-AE flags anomolous flows based on the reconstruction error
#3-Final decision, begign by default, RF label for the AE flagged flows 
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

RF_MODEL_PATH = "scripts/rf_model.joblib"
DATA_DIR      = "Train and Test Files"
RESULTS_DIR   = "Results and Reports"
SEQ_METRICS   = os.path.join(RESULTS_DIR, "hybrid_sequential_metrics.json")


#threshholds
#set to a float to skip the sweep and set to none for autosweeping for best F1
AE_THRESHOLD = None  # e.g. 57.34
RF_THRESHOLD = None  # e.g. 0.08

def load_test_data():
    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test_first400k.csv"))
    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test_first400k.csv"))
    y_true = y_test.iloc[:, 0].astype(int).to_numpy()
    return X_test, y_true



#undersample the attack class to get the begign to attack ratio 
def balance_test_data(X_test, y_true, ratio=1.0, random_state=42):
    benign_idx = np.where(y_true == 0)[0]
    attack_idx = np.where(y_true == 1)[0]

    n_benign = len(benign_idx)
    n_attack = len(attack_idx)
    n_attack_target = int(n_benign * ratio)

    print(f"[BALANCE] Original - Benign: {n_benign}, Attack: {n_attack}")

    if n_attack_target >= n_attack:
        print(f"[BALANCE] No undersampling needed (target {n_attack_target} >= available {n_attack})")
        return X_test, y_true

    np.random.seed(random_state)
    attack_idx_sampled = np.sort(
        np.random.choice(attack_idx, size=n_attack_target, replace=False)
    )
    balanced_idx = np.sort(np.concatenate([benign_idx, attack_idx_sampled]))

    X_balanced = X_test.iloc[balanced_idx].reset_index(drop=True)
    y_balanced = y_true[balanced_idx]

    print(f"[BALANCE] After 1:{ratio} rebalancing - Benign: {n_benign}, Attack: {n_attack_target}")
    print(f"[BALANCE] Total samples: {len(y_balanced)}")
    return X_balanced, y_balanced

#loads the autoencoder model and scalar from the files, already trained
def load_autoencoder():
    model = Autoencoder(cfg.INPUT_DIM, cfg.LATENT_DIM, cfg.HIDDEN_DIMS)
    model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH, map_location=cfg.DEVICE))
    model.to(cfg.DEVICE)
    model.eval()
    scaler = joblib.load(cfg.SCALER_SAVE_PATH)
    return model, scaler

#computes the reconstruction errors of the input data X_df using the ae, gives errors in numpy array
def compute_reconstruction_errors(model, scaler, X_df: pd.DataFrame) -> np.ndarray:
    X_scaled = scaler.transform(X_df.values)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(cfg.DEVICE)
    with torch.no_grad():
        recon = model(X_tensor)
    errors = torch.mean((recon - X_tensor) ** 2, dim=1).cpu().numpy()
    return errors

#goes through different ae valyes to max f1, minimum precision and recall
def sweep_ae_thresholds(errors: np.ndarray, y_true: np.ndarray,
                        min_precision: float = 0.4,
                        min_recall: float = 0.4):
    loss = errors
    min_loss, max_loss = loss.min(), loss.max()

    thresh = min_loss
    best = {
        "f1": 0.0, "precision": 0.0, "recall": 0.0,
        "fpr": 0.0, "fnr": 0.0, "threshold": min_loss,
        "predictions": None,
    }

    while thresh <= max_loss / 0.9:
        thresh *= 1.05
        pred = (loss > thresh).astype(np.int64)

        if pred.sum() == 0:
            continue

        P = precision_score(y_true, pred, zero_division=0)
        R = recall_score(y_true, pred, zero_division=0)
        if P < min_precision or R < min_recall:
            continue

        F1 = (2 * P * R) / (P + R + 1e-10)

        TP = int(((pred == 1) & (y_true == 1)).sum())
        FP = int(((pred == 1) & (y_true == 0)).sum())
        FN = int(((pred == 0) & (y_true == 1)).sum())
        TN = int(((pred == 0) & (y_true == 0)).sum())

        FPR = FP / (FP + TN + 1e-10)
        FNR = FN / (FN + TP + 1e-10)

        if F1 > best["f1"]:
            best.update({
                "f1": F1, "precision": P, "recall": R,
                "fpr": FPR, "fnr": FNR,
                "threshold": thresh,
                "predictions": pred.copy(),
                "TP": TP, "FP": FP, "FN": FN, "TN": TN,
            })

    return best

#loads random forest 
def load_rf():
    bundle = joblib.load(RF_MODEL_PATH)
    return bundle["pre"], bundle["rf"]

#converts confusion matrix into readable format
def confusion_matrix_markdown(cm):
    TN, FP = cm[0]
    FN, TP = cm[1]
    return (
        f"[ {TN:>7}  {FP:>7} ]\n"
        f"[ {FN:>7}  {TP:>7} ]"
    )

#summary of rf performance
def save_markdown_report(path, ae, rf, seq, timing):
    def pct(x):
        return f"{x*100:.2f}%"

    n_rows = timing.get("n_test_rows", 330000)

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Hybrid Model Evaluation Report\n\n")

        f.write("## Autoencoder Only Performance\n")
        f.write("### Confusion Matrix\n```\n")
        f.write(f"[ TN: {ae['TN']:7d}   FP: {ae['FP']:7d} ]\n")
        f.write(f"[ FN: {ae['FN']:7d}   TP: {ae['TP']:7d} ]\n")
        f.write("```\n\n")
        f.write(f"**Precision:** {pct(ae['precision'])}\n\n")
        f.write(f"**Recall:** {pct(ae['recall'])}\n\n")
        f.write(f"**F1 Score:** {pct(ae['f1'])}\n\n")
        f.write("---\n\n")

        f.write("## Random Forest Only Performance\n")
        f.write("### Confusion Matrix\n```\n")
        f.write(f"[ TN: {rf['TN']:7d}   FP: {rf['FP']:7d} ]\n")
        f.write(f"[ FN: {rf['FN']:7d}   TP: {rf['TP']:7d} ]\n")
        f.write("```\n\n")
        f.write(f"**ROC-AUC:** {rf['roc_auc']:.6f}\n\n")
        f.write(f"**PR-AUC:** {rf['pr_auc']:.6f}\n\n")
        f.write(f"**F1 Score:** {pct(rf['f1'])}\n\n")
        f.write("---\n\n")

        f.write("## Sequential AE to RF Pipeline Performance\n")
        f.write("### Confusion Matrix\n```\n")
        f.write(f"[ TN: {seq['TN']:7d}   FP: {seq['FP']:7d} ]\n")
        f.write(f"[ FN: {seq['FN']:7d}   TP: {seq['TP']:7d} ]\n")
        f.write("```\n\n")
        f.write(f"**Precision:** {pct(seq['precision'])}\n\n")
        f.write(f"**Recall:** {pct(seq['recall'])}\n\n")
        f.write(f"**F1 Score:** {pct(seq['f1'])}\n\n")
        f.write(f"**AE Best Threshold:** {seq['ae_best_threshold']:.4f}\n\n")
        f.write(f"**AE Fraction Flagged:** {pct(seq['ae_fraction_flagged'])}\n\n")
        f.write("---\n\n")

        ae_time = timing["ae_processing_seconds"]
        rf_time = timing["rf_processing_seconds"]
        f.write("## Processing Times\n")
        f.write(f"- **AE:** {ae_time:.2f}s for {n_rows:,} rows "
                f"({n_rows/ae_time:,.0f} rows/sec)\n\n")
        f.write(f"- **RF:** {rf_time:.2f}s for {n_rows:,} rows "
                f"({n_rows/rf_time:,.0f} rows/sec)\n\n")

#goes through rf prob thresholds from 0.01 to 0.99, returns best f1
def sweep_rf_thresholds(probs: np.ndarray, y_true: np.ndarray):
    best = {"f1": 0.0, "threshold": 0.5, "precision": 0.0, "recall": 0.0,
            "TP": 0, "FP": 0, "FN": 0, "TN": 0, "predictions": None}
    for t in np.arange(0.01, 1.00, 0.01):
        pred = (probs >= t).astype(np.int64)
        if pred.sum() == 0:
            continue
        P  = precision_score(y_true, pred, zero_division=0)
        R  = recall_score(y_true, pred, zero_division=0)
        F1 = f1_score(y_true, pred, zero_division=0)
        if F1 > best["f1"]:
            TP = int(((pred == 1) & (y_true == 1)).sum())
            FP = int(((pred == 1) & (y_true == 0)).sum())
            FN = int(((pred == 0) & (y_true == 1)).sum())
            TN = int(((pred == 0) & (y_true == 0)).sum())
            best.update({"f1": F1, "threshold": float(t), "precision": P, "recall": R,
                         "TP": TP, "FP": FP, "FN": FN, "TN": TN, "predictions": pred.copy()})
    return best

#computes eval metrics for a given set of true and preducted labels 
def compute_metrics(y_true, y_pred, probs=None):
    cm = confusion_matrix(y_true, y_pred)
    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())

    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix_markdown(cm),
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
    }
    if probs is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probs))
        metrics["pr_auc"]  = float(average_precision_score(y_true, probs))
    return metrics


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    X_test, y_true = load_test_data()
    print(f"Loaded test set: {X_test.shape[0]:,} rows, {X_test.shape[1]} features")
    print(f"  Benign:  {(y_true == 0).sum():,}  Attack: {(y_true == 1).sum():,}")
    #ae filter 
    ae_model, ae_scaler = load_autoencoder()
    t0     = time.perf_counter()
    errors = compute_reconstruction_errors(ae_model, ae_scaler, X_test)
    ae_time = time.perf_counter() - t0
    if AE_THRESHOLD is not None:
        ae_pred = (errors > AE_THRESHOLD).astype(np.int64)
        _P  = precision_score(y_true, ae_pred, zero_division=0)
        _R  = recall_score(y_true, ae_pred, zero_division=0)
        _F1 = f1_score(y_true, ae_pred, zero_division=0)
        _TP = int(((ae_pred == 1) & (y_true == 1)).sum())
        _FP = int(((ae_pred == 1) & (y_true == 0)).sum())
        _FN = int(((ae_pred == 0) & (y_true == 1)).sum())
        _TN = int(((ae_pred == 0) & (y_true == 0)).sum())
        _FPR = _FP / (_FP + _TN + 1e-10)
        _FNR = _FN / (_FN + _TP + 1e-10)
        ae_best = {
            "threshold": AE_THRESHOLD, "predictions": ae_pred,
            "precision": _P, "recall": _R, "f1": _F1,
            "fpr": _FPR, "fnr": _FNR,
            "TP": _TP, "FP": _FP, "FN": _FN, "TN": _TN,
        }
        print(f"\n[AE FILTER] Using hard-coded threshold: {AE_THRESHOLD}")
    else:
        ae_best = sweep_ae_thresholds(errors, y_true)
        if ae_best["predictions"] is None:
            raise RuntimeError("AE threshold sweep found no threshold meeting precision/recall constraints.")
        ae_pred = ae_best["predictions"]

    print("\n[AE FILTER] Threshold:")
    print(f"  threshold = {ae_best['threshold']:.6e}")
    print(f"  precision = {ae_best['precision']:.4f}")
    print(f"  recall    = {ae_best['recall']:.4f}")
    print(f"  f1        = {ae_best['f1']:.4f}")
    print(f"  FPR       = {ae_best['fpr']:.4f}")
    print(f"  FNR       = {ae_best['fnr']:.4f}")

    frac_flagged = float((ae_pred == 1).mean())
    print(f"  fraction of flows passed to RF = {frac_flagged:.4f}")

    #rf filtered 
    pre, rf = load_rf()
    t1     = time.perf_counter()
    Xte_rf = pre.transform(X_test)
    rf_probs = rf.predict_proba(Xte_rf)[:, 1]
    rf_time = time.perf_counter() - t1

    if RF_THRESHOLD is not None:
        rf_pred = (rf_probs >= RF_THRESHOLD).astype(np.int64)
        _P  = precision_score(y_true, rf_pred, zero_division=0)
        _R  = recall_score(y_true, rf_pred, zero_division=0)
        _F1 = f1_score(y_true, rf_pred, zero_division=0)
        _TP = int(((rf_pred == 1) & (y_true == 1)).sum())
        _FP = int(((rf_pred == 1) & (y_true == 0)).sum())
        _FN = int(((rf_pred == 0) & (y_true == 1)).sum())
        _TN = int(((rf_pred == 0) & (y_true == 0)).sum())
        rf_best = {
            "threshold": RF_THRESHOLD, "predictions": rf_pred,
            "precision": _P, "recall": _R, "f1": _F1,
            "TP": _TP, "FP": _FP, "FN": _FN, "TN": _TN,
        }
        print(f"\n[RF ONLY] Using hard-coded threshold: {RF_THRESHOLD}")
    else:
        rf_best = sweep_rf_thresholds(rf_probs, y_true)
        rf_pred = rf_best["predictions"]
    print("\n[RF ONLY] Threshold:")
    print(f"  threshold = {rf_best['threshold']:.2f}")
    print(f"  precision = {rf_best['precision']:.4f}")
    print(f"  recall    = {rf_best['recall']:.4f}")
    print(f"  f1        = {rf_best['f1']:.4f}")

    rf_metrics = compute_metrics(y_true, rf_pred, rf_probs)
    print("\n[RF ONLY] metrics:")
    print(json.dumps(rf_metrics, indent=2))

    #sequential ae to rf 
    final_pred = np.zeros_like(y_true)
    idx_flagged = np.where(ae_pred == 1)[0]
    final_pred[idx_flagged] = rf_pred[idx_flagged]

    seq_metrics = compute_metrics(y_true, final_pred, probs=None)
    seq_metrics.update({
        "ae_best_threshold":  float(ae_best["threshold"]),
        "ae_precision":       float(ae_best["precision"]),
        "ae_recall":          float(ae_best["recall"]),
        "ae_f1":              float(ae_best["f1"]),
        "ae_fpr":             float(ae_best["fpr"]),
        "ae_fnr":             float(ae_best["fnr"]),
        "ae_fraction_flagged": frac_flagged,
        "rf_best_threshold":  float(rf_best["threshold"]),
    })

    print("\n[SEQUENTIAL AE->RF] metrics:")
    print(json.dumps(seq_metrics, indent=2))

    ae_only_metrics = compute_metrics(y_true, ae_pred, probs=None)
    print("\n[AE ONLY] metrics:")
    print(json.dumps(ae_only_metrics, indent=2))

    timing = {
        "ae_processing_seconds": ae_time,
        "rf_processing_seconds": rf_time,
        "n_test_rows": len(y_true),
    }
    out = {
        "ae_only":           ae_only_metrics,
        "rf_only":           rf_metrics,
        "sequential_ae_rf":  seq_metrics,
        "timing":            timing,
    }
    with open(SEQ_METRICS, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nDone: Saved sequential AE->RF metrics -> {SEQ_METRICS}")

    save_markdown_report(
        "Results and Reports/hybrid_report.md",
        ae_only_metrics, rf_metrics, seq_metrics, timing,
    )


if __name__ == "__main__":
    main()
