# online_sequential_pipeline.py — Online sequential AE→RF hybrid detector.
#
# For each incoming packet feature vector:
#   1. Scale with the saved MinMaxScaler
#   2. Compute AE reconstruction error
#   3. If error exceeds threshold → pass raw features to RF for classification
#   4. Return the final benign/attack decision
#
# The AE runs on SCALED input (scaler.pkl from training).
# The RF runs on RAW features through its own preprocessor stored in rf_model.joblib.
#
# Usage (from project root):
#   python -m scripts.Hybrid.smoke_test_online      # single-sample smoke test
#   python -m scripts.Hybrid.batch_smoke_test_115    # batch test from CSV
#
# Note: the feature dimension of the ingestion pipeline must match INPUT_DIM
# in AE_config (currently 115). A mismatch will raise a ValueError.

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import torch

from scripts.autoencoder.Autoencoder import Autoencoder
import scripts.autoencoder.AE_config as cfg

RF_BUNDLE_PATH = os.path.join("scripts", "rf_model.joblib")


class OnlineSequentialAERF:
    """Sequential AE→RF decision engine, called once per packet feature vector.

    Attributes:
        ae_threshold: Reconstruction-error cutoff — packets above this are
                      forwarded to the Random Forest for a second opinion.
        rf_threshold: Probability cutoff for the RF's attack class (default 0.5).
    """

    def __init__(
        self,
        ae_threshold: float,
        rf_threshold: float = 0.5,
        device: Optional[str] = None,
        rf_bundle_path: str = RF_BUNDLE_PATH,
    ):
        self.device = device or cfg.DEVICE

        # ── Load the scaler used during AE training ──────────────────
        if not os.path.exists(cfg.SCALER_SAVE_PATH):
            raise FileNotFoundError(f"Missing scaler at: {cfg.SCALER_SAVE_PATH}")
        self.scaler = joblib.load(cfg.SCALER_SAVE_PATH)

        # Infer input dimension from the fitted scaler when possible
        self.input_dim = int(getattr(self.scaler, "n_features_in_", cfg.INPUT_DIM))

        # ── Load the trained autoencoder ─────────────────────────────
        if not os.path.exists(cfg.MODEL_SAVE_PATH):
            raise FileNotFoundError(f"Missing AE model at: {cfg.MODEL_SAVE_PATH}")
        self.ae = Autoencoder(self.input_dim, cfg.LATENT_DIM, cfg.HIDDEN_DIMS)
        self.ae.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH, map_location=self.device))
        self.ae.to(self.device)
        self.ae.eval()

        # ── Load the Random Forest bundle (preprocessor + model) ─────
        if not os.path.exists(rf_bundle_path):
            raise FileNotFoundError(f"Missing RF bundle at: {rf_bundle_path}")
        bundle = joblib.load(rf_bundle_path)
        if "pre" not in bundle or "rf" not in bundle:
            raise KeyError("rf_model.joblib must contain keys: 'pre' and 'rf'")
        self.pre = bundle["pre"]       # Sklearn preprocessor / ColumnTransformer
        self.rf = bundle["rf"]         # Trained RandomForestClassifier
        self.rf_num_cols = bundle.get("num_cols")  # Column names if RF used a DataFrame

        self.ae_threshold = float(ae_threshold)
        self.rf_threshold = float(rf_threshold)

        # ── Running counters for summary stats ───────────────────────
        self.total = 0
        self.flagged_by_ae = 0
        self.rf_called = 0
        self.attacks_final = 0

    # ── Internal helpers ─────────────────────────────────────────────

    def _ae_error(self, x_scaled_np: np.ndarray) -> float:
        """Return the mean squared reconstruction error for one scaled sample."""
        x_t = torch.tensor(x_scaled_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            recon = self.ae(x_t)
        err = torch.mean((recon - x_t) ** 2, dim=1).item()
        return float(err)

    def _rf_predict(self, x_raw_1d: np.ndarray) -> tuple[float, int]:
        """Run the RF on a single raw feature vector and return (prob_attack, pred)."""
        # If the RF was trained with named columns, wrap the vector in a DataFrame
        if self.rf_num_cols is not None:
            if len(self.rf_num_cols) != x_raw_1d.shape[0]:
                raise ValueError(
                    f"RF expects {len(self.rf_num_cols)} features, got {x_raw_1d.shape[0]}"
                )
            x_in = pd.DataFrame([x_raw_1d.tolist()], columns=self.rf_num_cols)
            x_rf = self.pre.transform(x_in)
        else:
            x_rf = self.pre.transform(x_raw_1d.reshape(1, -1))

        prob = float(self.rf.predict_proba(x_rf)[0, 1])
        pred = int(prob >= self.rf_threshold)
        return prob, pred

    # ── Public API ───────────────────────────────────────────────────

    def process_one(self, x_raw: np.ndarray) -> Dict[str, Any]:
        """Score a single raw feature vector through the AE→RF pipeline.

        Returns a dict with: final_pred, ae_error, ae_flagged, rf_prob, rf_pred.
        """
        self.total += 1
        x_raw = np.asarray(x_raw, dtype=np.float32)

        if x_raw.ndim != 1 or x_raw.shape[0] != self.input_dim:
            raise ValueError(f"Expected vector shape ({self.input_dim},), got {x_raw.shape}")

        # Step 1 — scale for the autoencoder
        x_scaled = self.scaler.transform(x_raw.reshape(1, -1))

        # Step 2 — compute reconstruction error
        ae_err = self._ae_error(x_scaled)
        ae_flagged = ae_err > self.ae_threshold

        rf_prob: Optional[float] = None
        rf_pred: Optional[int] = None

        # Step 3 — only invoke RF if AE flags the packet as anomalous
        if ae_flagged:
            self.flagged_by_ae += 1
            self.rf_called += 1
            rf_prob, rf_pred = self._rf_predict(x_raw)
            final_pred = int(rf_pred)
        else:
            final_pred = 0

        if final_pred == 1:
            self.attacks_final += 1

        return {
            "final_pred": int(final_pred),
            "ae_error": float(ae_err),
            "ae_flagged": bool(ae_flagged),
            "rf_prob": rf_prob,
            "rf_pred": rf_pred,
        }

    def summary(self) -> Dict[str, Any]:
        """Return aggregate counts for the session so far."""
        if self.total == 0:
            return {
                "total": 0,
                "ae_flagged": 0,
                "ae_flagged_frac": 0.0,
                "rf_called": 0,
                "final_attacks": 0,
            }
        return {
            "total": int(self.total),
            "ae_flagged": int(self.flagged_by_ae),
            "ae_flagged_frac": float(self.flagged_by_ae / self.total),
            "rf_called": int(self.rf_called),
            "final_attacks": int(self.attacks_final),
        }
