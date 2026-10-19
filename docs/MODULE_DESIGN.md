# PacketVigil Module Design

Module-level specifications for the as-built system. For the big picture and rationale see [ARCHITECTURE.md](ARCHITECTURE.md).

## Contents

- [Tech stack](#tech-stack)
- [Ingestion module](#ingestion-module)
- [Feature extraction module](#feature-extraction-module)
- [Autoencoder](#autoencoder)
- [Random forest](#random-forest)
- [Sequential decision logic](#sequential-decision-logic)
- [Alerting and dashboard](#alerting-and-dashboard)
- [Module-to-requirement traceability](#module-to-requirement-traceability)
- [Interfaces](#interfaces)
- [Error handling and recovery](#error-handling-and-recovery)
- [Notes on repo state](#notes-on-repo-state)

## Tech stack

| Area | Choice |
|---|---|
| Language | Python 3.10+; AfterImage has a Cython variant (`AfterImage_cy.pyx`) |
| Capture | `dumpcap` (Linux live, preferred), `tcpdump` (Linux fallback), `tshark` (Windows and PCAP replay) |
| Packet parsing | `dpkt` on the raw pcap byte stream from capture stdout; no intermediate files |
| Feature extraction | AfterImage incremental statistics (derived from Kitsune-py) |
| ML | PyTorch (autoencoder); scikit-learn (random forest, MinMaxScaler, ColumnTransformer); joblib (model bundle); numpy, pandas |
| Web | Flask (REST) + Dash + Dash Bootstrap Components, one Flask process; flask-cors for development |
| Monitoring | psutil (CPU and memory) |
| Target platform | Ubuntu (validated on a 3-core VM, no GPU) for live capture; Windows for offline replay and development |

Install list: see [`Requirements.txt`](../Requirements.txt) and the root README.

## Ingestion module

| Field | Detail |
|---|---|
| Files | [`NetworkIngestion/live_detector_linux.py`](../NetworkIngestion/live_detector_linux.py) (live), [`NetworkIngestion/replay_detector.py`](../NetworkIngestion/replay_detector.py) (PCAP replay) |
| Input | Live interface (dumpcap) or a PCAP file (tshark, replay) |
| Output | Per-packet metadata (timestamp, src/dst IP, src/dst port, protocol, frame length) passed straight into feature extraction |
| Protocols | Ethernet, IPv4, IPv6, ARP, TCP, UDP, ICMP |
| Live path | `dumpcap -i veth_detect -B 256 -P -w -` (optionally under `ip netns exec detect_ns`), parsed by a pcap stream reader plus `dpkt` in a single reader+FE thread |
| Replay path | tshark emits 19 tab-separated fields per packet; a reader thread pushes lines into a `queue.Queue` (2,000,000) and a separate FE thread parses them |
| Exceptions | Malformed packets are counted (`fe_stats['malformed']`) and skipped. If capture cannot start (privileges, bad interface, missing PCAP) the detector exits with a diagnostic and writes no partial alerts. |

## Feature extraction module

| Field | Detail |
|---|---|
| Files | [`AfterImage.py`](../NetworkIngestion/AfterImage.py), [`AfterImage_cy.pyx`](../NetworkIngestion/AfterImage_cy.pyx) (Cython; build with [`HelperScripts/setup_AfterImage.py`](../NetworkIngestion/HelperScripts/setup_AfterImage.py)), [`netStat.py`](../NetworkIngestion/netStat.py), [`FeatureExtractor.py`](../NetworkIngestion/FeatureExtractor.py) |
| Output | 115-dim float32 vector per packet: 5 traffic dimensions across 5 decay windows with per-window statistics |
| Decay factors | lambda in {5, 3, 1, 0.1, 0.01} |
| State | Hash tables keyed by source host, host pair, channel, socket and jitter; capped and LRU-pruned (details in [ARCHITECTURE.md](ARCHITECTURE.md#feature-engine-deep-dive-afterimage)) |
| Batching | Live detector: 64 vectors per `ml_q.put()`; replay detector: batches of 64 |
| Exceptions | NaN and inf values are replaced before vectors leave the module |

## Autoencoder

Files: [`ML/autoencoder/Autoencoder.py`](../ML/autoencoder/Autoencoder.py) (model), [`AE_config.py`](../ML/autoencoder/AE_config.py) (single source of truth for hyperparameters and paths), [`AE_train.py`](../ML/autoencoder/AE_train.py) (training), plus the trained artifacts `autoencoder.pt` and `scaler.pkl`.

A symmetric feed-forward **denoising, sparse** autoencoder trained on benign traffic only. It acts as a high-recall, high-throughput filter.

**Architecture:** encoder 115 -> 64 -> 36 -> 8, decoder 8 -> 36 -> 64 -> 115. ReLU and Dropout(0.1) after each encoder hidden layer. A sigmoid of the 8-unit latent is stored for the sparsity penalty during training.

| Hyperparameter | Value |
|---|---|
| Input / latent dim | 115 / 8 |
| Hidden dims | [64, 36] |
| Loss | MSE reconstruction + KL sparsity penalty |
| Sparsity target / weight | 0.05 / 1e-3 |
| Input noise (denoising) | Gaussian, sigma = 0.1 |
| Optimizer | Adam, learning rate 1e-4 |
| Batch size / epochs | 1024 / 8 |
| Scaling | scikit-learn `MinMaxScaler` fitted on the benign training set, saved as `scaler.pkl` |
| Output at inference | Mean-squared reconstruction error (scalar per packet) |

**AE threshold.** Tuned per attack profile; adjustable. Examples from the project binder: Fuzzing 0.000179, SSDP Flood 57.347, OS Scan 2.8 (live). The repo's detectors ship with `AE_THRESHOLD = 2.8` (live) and `3.1` (replay).

### Training (pseudocode)

```text
Input : benign traffic X, sparsity target rho, sparsity weight lambda,
        noise std sigma, batch size B, learning rate eta, epochs E
Output: trained autoencoder parameters theta, fitted scaler

1.  scaler <- MinMaxScaler.fit(X);  X <- scaler.transform(X)
2.  initialize autoencoder (theta) and Adam(eta)
3.  for epoch = 1 .. E:
4.      for each minibatch x of size B:
5.          x_noisy  <- x + Normal(0, sigma^2)
6.          h        <- Encoder(x_noisy)
7.          x_hat    <- Decoder(h)
8.          L_recon  <- MSE(x_hat, x)                      # reconstruct the CLEAN input
9.          p_hat    <- mean(sigmoid(h), over batch)       # average latent activation
10.         L_sparse <- sum( rho*log(rho/p_hat) + (1-rho)*log((1-rho)/(1-p_hat)) )
11.         L_total  <- L_recon + lambda * L_sparse
12.         update theta with the gradient of L_total
13. save autoencoder weights (autoencoder.pt) and scaler (scaler.pkl)
```

Inference mirrors training without noise or dropout: scale, reconstruct, compute `e(x) = ||x - x_hat||^2 / d`, flag if `e(x) > threshold`.

## Random forest

Trained and evaluated offline in [`ML/Hybrid Model/`](../ML/Hybrid%20Model) (`Hybrid_Train.py`, `Hybrid_Sequential_AE_RF.py`, and `_parquet` variants); the fitted model ships as [`ML/rf_model.joblib`](../ML/rf_model.joblib).

| Field | Detail |
|---|---|
| Model | scikit-learn `RandomForestClassifier`: 400 trees, `class_weight='balanced'`, `max_depth=None` |
| Training data | Labeled benign + attack samples from CIC-IDS-2017 and the Kitsune datasets, split 80/20 chronologically |
| Preprocessing | One-hot encoding for categorical columns, passthrough for numeric; stored with the model in one joblib bundle |
| Leakage controls | Known-leaky features (raw IPs, ports, timestamps) dropped before training so the RF must use behavioral features |
| Input at inference | The raw (unscaled) 115-dim vector, only for AE-flagged packets |
| Output | Attack probability in [0, 1] |
| Threshold | Per attack profile (typically 0.01-0.5); repo defaults are 0.5 (live) and 0.12 (replay) |
| Offline metrics | ROC-AUC, PR-AUC, F1, precision, recall, confusion matrix; per-row scores saved to CSV |

## Sequential decision logic

Implemented by the `OnlineSequentialAERF` class in [`ML/Hybrid Model/online_sequential_pipeline.py`](../ML/Hybrid%20Model/online_sequential_pipeline.py). The original design left three options open (cascade, latent-as-RF-input, weighted voting); the build committed to the cascade:

1. Compute AE reconstruction error `e(x) = ||x - x_hat||^2 / d` on the scaled vector.
2. If `e(x) <= tau_AE`: decide **benign** and stop.
3. Otherwise compute the RF attack probability `p(x)` on the raw vector.
4. If `p(x) >= tau_RF`: decide **attack** and emit an alert; else **benign**.
5. The default verdict is benign. The AE keeps the RF's workload small enough to stay real-time; it does not make the final call.

The class exposes a per-packet method (`process_one`) returning AE error, RF probability (when computed), final verdict and metadata, and keeps running counters (packets seen, AE flags, RF calls, confirmed attacks). The live and replay detectors apply the same logic on whole batches for speed.

## Alerting and dashboard

**Alert writer:** [`NetworkIngestion/ui_bridge.py`](../NetworkIngestion/ui_bridge.py) formats confirmed attacks, appends them to `UI/data/alerts.json`, and atomically rewrites `UI/data/metrics.json`.

**Alert schema** (one JSON object per line):

| Field | Meaning |
|---|---|
| `timestamp` | ISO 8601 UTC |
| `flow_id` | Identifier derived from the packet number (`f_<n>`) |
| `src_ip`, `dst_ip`, `src_port`, `dst_port` | Packet addressing |
| `ae_score`, `ae_threshold`, `ae_verdict` | AE reconstruction error, threshold used, verdict |
| `rf_prob`, `rf_threshold`, `rf_verdict` | RF probability, threshold used, verdict |
| `final_verdict` | `attack` or `benign` |
| `attack_type` | When known (currently `null` from live detection) |
| `true_label` | Ground-truth label when replaying labeled data, else `null` |

**Metrics file** (`metrics.json`): status, packets processed, alerts, AE-flagged count, alert rate per second, CPU and memory (psutil), and, when ground truth is available, hybrid precision/recall/F1.

**Dashboard** (Flask + Dash, [`UI/api/`](../UI/api), started by [`UI/run_api.py`](../UI/run_api.py)): three tabs.

- *Dashboard*: system status, P/R/F1, alert count and rate, resource usage, active thresholds, live attack feed.
- *Anomaly Inspector*: paginated, sortable table with verdict and date filtering.
- *Settings*: AE and RF threshold sliders.

Refresh interval is 5 s (`DASHBOARD_REFRESH_INTERVAL` in `UI/api/config.py`); default port 5000.

**REST endpoints** (from `UI/api/api_routes.py`):

| Route | Methods | Purpose |
|---|---|---|
| `/api` | GET | API root |
| `/api/metrics` | GET | Current metrics |
| `/api/alerts` | GET | Alerts with `verdict`, `start_date`, `end_date`, `limit`, `offset`, `sort_by`, `sort_order` query params |
| `/api/threshold` | GET, POST | Read or update AE/RF thresholds (stored in `config.json`) |
| `/api/health` | GET | File presence, CPU, memory, recent crash count |
| `/api/logs` | GET | Export run logs as JSON or CSV (`format`, `hours`) |
| `/api/crash` | POST | Record a crash entry |
| `/api/detection/start`, `/api/detection/stop`, `/api/detection/status` | POST, POST, GET | Control and inspect a detection engine (see [repo state](#notes-on-repo-state)) |

## Module-to-requirement traceability

Requirement IDs come from the project binder (see [RESULTS.md](RESULTS.md#requirements-verification)). Paths are the repo's actual paths.

| Module / file | Responsibility | Supports |
|---|---|---|
| `NetworkIngestion/live_detector_linux.py` | Live detector: spawns dumpcap, runs reader+FE thread, owns the ML subprocess, writes alerts | FUN-1, FUN-3, FUN-4, PER-1, PER-7, PER-8 |
| `NetworkIngestion/replay_detector.py` | Offline replay through the same pipeline via tshark; deterministic benchmarking | FUN-7, TEST-1, TEST-2 |
| `NetworkIngestion/FeatureExtractor.py` | Wraps AfterImage; Kitsune-compatible 115-dim vector | FUN-2 |
| `NetworkIngestion/AfterImage.py`, `AfterImage_cy.pyx`, `netStat.py` | Incremental statistics (Python and Cython) | FUN-2, PER-7 |
| `NetworkIngestion/ui_bridge.py` | Alert schema formatting, `alerts.json`, `metrics.json` | FUN-4, FUN-6, I/O-2 |
| `ML/autoencoder/Autoencoder.py` | Symmetric AE (115-64-36-8) | FUN-3 |
| `ML/autoencoder/AE_config.py` | AE hyperparameters and paths | FUN-3 |
| `ML/autoencoder/AE_train.py` | Scaler fit, denoising MSE + KL sparsity training, weight save | TEST-1 |
| `ML/Hybrid Model/online_sequential_pipeline.py` | `OnlineSequentialAERF` cascade | FUN-3, PER-4, PER-7 |
| `ML/Hybrid Model/Hybrid_Sequential_AE_RF.py` | Offline evaluator: precision, recall, F1, confusion matrices | TEST-1, PER-2, PER-5 |
| `UI/api/app.py` | Flask + Dash bootstrap | FUN-5, I/O-3 |
| `UI/api/api_routes.py` | REST endpoints | I/O-3 |
| `UI/api/dash_layout.py`, `dash_callbacks.py` | Dashboard layout and callbacks | FUN-5 |
| `UI/api/data_manager.py` | Thread-safe reads of alerts, metrics, config | FUN-4, ENV-1 |

## Interfaces

### Internal

| Interface | Producer to consumer | Payload |
|---|---|---|
| Packet tuple | Dissection to feature extraction | timestamp, src/dst IP, ports, protocol, frame length |
| Feature batch | Feature extraction to ML process (`mp.Queue`, cap 50,000) | numpy array of 115-dim float32 vectors (64 per put) |
| AE | ML process | In: scaled 115-dim float32. Out: MSE scalar |
| RF | ML process | In: raw 115-dim float32, AE-flagged only. Out: probability in [0, 1] |
| Decision result | Decision logic to alerting | AE error, RF probability, verdict, packet metadata |

### External

| Interface | Detail |
|---|---|
| Capture | Subprocess pipe to dumpcap or tshark, raw pcap bytes on stdout |
| Alert log | `UI/data/alerts.json`, JSON lines |
| Metrics | `UI/data/metrics.json` |
| REST | `http://<host>:5000/api/...` |
| Dashboard | `http://<host>:5000/dashboard/` |

## Error handling and recovery

| Fault | Response |
|---|---|
| Capture fails to start | `RuntimeError` with a diagnostic; clean exit; no partial alerts |
| Malformed packet | Counted and skipped; pipeline continues |
| Missing `scaler.pkl`, `autoencoder.pt`, or `rf_model.joblib` | `FileNotFoundError` at startup, before any packet is processed |
| Feature dimension mismatch (for example a scaler from a different model) | `ValueError` before inference |
| NaN / inf features | Replaced before reaching the AE |
| ML process dies | Parent logs the event and stops; no silent restart, so a real fault is not hidden |
| Fault detection | `metrics.json` carries packets processed, AE flags, RF calls, alerts, latency, throughput, memory and CPU; the dashboard shows a Normal / Elevated / Critical alert-rate badge |

Recovery model: single runs that the operator restarts. The 86,890-packet warm-up rebuilds AfterImage state before scoring resumes. Alerts are append-only JSON lines, so a restart does not corrupt earlier alerts.

Security notes: the detector only reads traffic and never generates or modifies packets; capture needs packet-capture privileges (`CAP_NET_RAW`) while the ML and dashboard processes do not. Role-based access control (requirement CON-1) was deferred.

## Notes on repo state

Accuracy caveats found while writing these docs, so readers are not surprised:

- `UI/api/api_routes.py` imports `api.detection_engine` for the `/api/detection/*` routes, but no `detection_engine.py` is tracked in the repo. Those three routes will return an error here. The detectors run from `NetworkIngestion/` and write alert and metrics files directly.
- `/api/threshold` writes thresholds to `config.json`, and validates them to the range 0 to 1. The detector scripts in this repo define their thresholds as constants at the top of each file and do not read `config.json`, so the "live apply" behavior described in the project binder is not visible in the tracked detector code.
- `ML/autoencoder/Autoencoder.py` (`import scripts.autoencoder.AE_config`), `ML/autoencoder/AE_config.py` (`MODEL_DIR`) and `ML/Hybrid Model/online_sequential_pipeline.py` (`RF_BUNDLE_PATH`) still point at the pre-reorganization `scripts/` paths, so adjust them when running from this layout.
