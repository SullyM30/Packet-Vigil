#!/usr/bin/env python3
"""
Hybrid Network Intrusion Detector — Deterministic PCAP Replay

Offline detection pipeline:
    tshark (reads PCAP) -> AfterImage (115 features) -> Hybrid ML (AE + RF) -> alert log

Pipeline architecture (2 threads + 1 process):
    Main process:
        Reader thread:  tshark stdout -> packet_q
        FE thread:      packet_q -> parse TSV -> AfterImage -> ml_q
    ML process:         ml_q -> AE -> RF -> alert + JSON log

The ML detector runs in a separate process so it gets its own CPU core and GIL,
running true parallel to the feature extractor.

Usage (from project root):
    python -m NetworkIngestion.replay_detector
"""

import csv
import json
import subprocess
import sys
import time
import queue
import threading
import multiprocessing as mp
import os
import numpy as np
import pandas as pd

# Imports
try:
    import NetworkIngestion.netStat as ns
except ImportError:
    import netStat as ns

from scripts.Hybrid.online_sequential_pipeline import OnlineSequentialAERF

# --- CONFIGURATION ---
TSHARK = r"D:\Wireshark\tshark.exe"

# PCAP file to replay. Use the full replayed PCAP (not a trimmed test file) so
# AfterImage builds a proper baseline before the test segment begins.
REPLAY_PCAP = r"Train and Test Files\Online Testing PCAPs\OSScan_Test_replayed.pcap"

# --- QUEUE SIZES ---
PACKET_QUEUE_MAX = 2000000
ML_QUEUE_MAX = 50000    # Smaller than packet queue; mp.Queue has serialization overhead

# How often (in packets) to print a progress update
REPORT_INTERVAL = 25000

# Number of feature vectors to batch before sending to the ML process
ML_BATCH_SIZE = 64

# --- ML THRESHOLDS ---
AE_THRESHOLD = 3.1      # Autoencoder reconstruction error cutoff (tuned via F1 sweep)
RF_THRESHOLD = 0.12     # Random Forest attack probability cutoff

# --- WARM-UP ---
# Feed this many packets through AfterImage before starting ML scoring.
# Lets AfterImage reach the same trained state before the test segment begins.
# replayed_fuzzing.pcap: 2,244,139 total - 661,763 test = 1,582,376 warmup
WARMUP_PACKETS = 86890

# --- OUTPUT ---
ALERT_LOG = "NetworkIngestion/attack_log.json"

# AfterImage feature engine (main process only)
maxHost = 100000
maxSess = 100000

nstat = ns.netStat(np.nan, maxHost, maxSess)
headers = nstat.getNetStatHeaders()
FEATURE_DIM = len(headers)

# Shared state used by the feature extractor thread
packet_q = queue.Queue(maxsize=PACKET_QUEUE_MAX)

fe_stats = {
    "packet_count": 0,
    "malformed": 0,
    "fe_total_time": 0.0,
    "fe_max_time": 0.0,
}


# =============================================================================
# THREAD 1 — READER: tshark stdout -> packet_q
# =============================================================================

def reader(tshark_proc, stop_event):
    for line in tshark_proc.stdout:
        if stop_event.is_set():
            break
        if not line.strip():
            continue
        # Block on full queue — we never drop packets in replay mode
        packet_q.put((line, time.time()), block=True)

    stop_event.set()


# =============================================================================
# THREAD 2 — FEATURE EXTRACTOR: packet_q -> AfterImage -> ml_q
# =============================================================================

def feature_extractor(ml_q, stop_event):
    """Parse each TSV row, extract 115 AfterImage features, and send batches to ml_q.

    Batches are (matrix, meta_list) tuples where matrix has shape (B, 115) float32.
    Sending one batch instead of one packet at a time reduces inter-process
    pickle overhead significantly.
    """

    batch_vecs = []
    batch_meta = []

    def _flush_batch():
        if not batch_vecs:
            return
        matrix = np.stack(batch_vecs)   # (B, 115) float32 — single allocation
        ml_q.put((matrix, list(batch_meta)), block=True)
        batch_vecs.clear()
        batch_meta.clear()

    while not stop_event.is_set() or not packet_q.empty():
        try:
            line, t_read = packet_q.get(timeout=0.5)
        except queue.Empty:
            continue

        # Parse TSV row; pad to 19 columns if tshark omitted trailing fields
        row = next(csv.reader([line], delimiter='\t'))
        if len(row) < 19:
            row += [""] * (19 - len(row))

        IPtype = np.nan
        timestamp = row[0]
        framelen = row[1]
        srcIP = ''
        dstIP = ''
        srcMAC = row[2]
        dstMAC = row[3]

        # Determine IP version from tshark fields
        if row[4] != '':
            srcIP = row[4]
            dstIP = row[5]
            IPtype = 0
        elif row[17] != '':
            srcIP = row[17]
            dstIP = row[18]
            IPtype = 1

        srcproto = row[6] + row[8]
        dstproto = row[7] + row[9]

        # Fill in protocol and IPs for ARP/ICMP/non-IP packets
        if srcproto == '':
            if row[12] != '':
                srcproto = 'arp'
                dstproto = 'arp'
                srcIP = row[14]
                dstIP = row[16]
                IPtype = 0
            elif row[10] != '':
                srcproto = 'icmp'
                dstproto = 'icmp'
                IPtype = 0
            elif srcIP + srcproto + dstIP + dstproto == '':
                srcIP = row[2]
                dstIP = row[3]

        try:
            fe_start = time.perf_counter()
            vec = nstat.updateGetStats(
                IPtype, srcMAC, dstMAC,
                srcIP, srcproto, dstIP,
                dstproto, int(framelen), float(timestamp),
            )
            fe_time = time.perf_counter() - fe_start
            fe_stats["fe_total_time"] += fe_time
            fe_stats["fe_max_time"] = max(fe_stats["fe_max_time"], fe_time)
        except Exception:
            fe_stats["malformed"] += 1
            continue

        if len(vec) != FEATURE_DIM:
            fe_stats["malformed"] += 1
            continue

        fe_stats["packet_count"] += 1

        # During warm-up, feed AfterImage but skip the ML queue
        if WARMUP_PACKETS > 0 and fe_stats["packet_count"] <= WARMUP_PACKETS:
            if fe_stats["packet_count"] % REPORT_INTERVAL == 0:
                print(
                    f"  [WARMUP] {fe_stats['packet_count']:,}/{WARMUP_PACKETS:,} pkts | "
                    f"pkt_q={packet_q.qsize():,}"
                )
            if fe_stats["packet_count"] == WARMUP_PACKETS:
                print(f"  [WARMUP] Complete — switching to detection mode")
            continue

        # Periodic progress report for the FE thread
        if fe_stats["packet_count"] % REPORT_INTERVAL == 0:
            avg_fe = (fe_stats["fe_total_time"] / fe_stats["packet_count"]) * 1000
            max_fe = fe_stats["fe_max_time"] * 1000
            try:
                mlq_size = ml_q.qsize()
            except NotImplementedError:
                mlq_size = -1
            print(
                f"  [FE] {fe_stats['packet_count']:,} pkts | "
                f"avg={avg_fe:.2f}ms max={max_fe:.1f}ms | "
                f"pkt_q={packet_q.qsize():,} ml_q={mlq_size:,}"
            )

        batch_vecs.append(np.array(vec, dtype=np.float32))
        batch_meta.append({
            "pkt_num": fe_stats["packet_count"],
            "timestamp": timestamp,
            "srcIP": srcIP,
            "dstIP": dstIP,
            "srcproto": srcproto,
            "dstproto": dstproto,
            "t_read": t_read,
        })

        if len(batch_vecs) >= ML_BATCH_SIZE:
            _flush_batch()

    # Flush any remaining partial batch, then signal the ML process that we're done
    _flush_batch()
    ml_q.put(None)


# =============================================================================
# PROCESS — ML DETECTOR: ml_q -> AE -> RF -> alert
# Runs in a separate process so it gets its own CPU core and GIL
# =============================================================================

def ml_detector_process(ml_q, result_q):
    """Entry point for the ML subprocess. Wraps _ml_detector_inner with error handling."""
    import traceback

    try:
        _ml_detector_inner(ml_q, result_q)
    except KeyboardInterrupt:
        pass  # _ml_detector_inner's finally block already sent results
    except Exception:
        traceback.print_exc()
        result_q.put({
            "stats": {"ml_processed": 0},
            "attack_log": [],
            "detector_counters": {"ae_flagged": 0, "rf_called": 0, "total": 0, "attacks_final": 0},
        })


def _ml_detector_inner(ml_q, result_q):
    import torch

    # Load models inside the child process to avoid pickling CUDA tensors across processes
    print("[ML process] Loading models on CPU...")
    detector = OnlineSequentialAERF(
        ae_threshold=AE_THRESHOLD,
        rf_threshold=RF_THRESHOLD,
        device="cpu",
    )
    print(f"[ML process] Models loaded (device=cpu)")

    # Open the UI bridge for live Dash dashboard updates.
    # The file handle stays open for the full run so each write is a cheap
    # buffered append rather than an open/close per alert.
    uib = None
    ui_alerts_f = None
    try:
        try:
            from NetworkIngestion import ui_bridge as uib
        except ImportError:
            import ui_bridge as uib
        uib.clear_alerts()
        ui_alerts_f = uib.open_alerts_writer()
        uib.update_metrics(status="starting", total_processed=0,
                           total_alerts=0, ae_flagged=0, run_start=time.time())
        print("[ML process] UI bridge initialised -> UI/data/")
    except Exception as _e:
        print(f"[ML process] UI bridge unavailable ({_e}) — dashboard feed disabled")
        uib = None
        ui_alerts_f = None
    _ui_last_update = 0.0   # wall-clock time of last metrics.json write

    # Per-run counters
    stats = {
        "ml_processed": 0,
        "attacks_detected": 0,
        "ae_flagged": 0,
        "ae_total_time": 0.0,
        "ae_max_time": 0.0,
        "rf_total_time": 0.0,
        "rf_max_time": 0.0,
        "rf_calls": 0,
        "ml_total_time": 0.0,
        "ml_max_time": 0.0,
    }

    # Write attack entries to disk incrementally to avoid large in-memory lists
    attack_log_path = ALERT_LOG + ".attacks.jsonl"
    attack_log_f = open(attack_log_path, "w", buffering=64 * 1024)
    start = None

    # Per-batch timing samples for percentile calculations
    batch_ml_times = []
    batch_ae_times = []

    # End-to-end latency per packet (time from packet read to ML decision)
    e2e_latencies = []

    det_ae_flagged = 0
    det_rf_called = 0
    det_total = 0
    det_attacks_final = 0

    ml_cpu_start = time.process_time()

    # Cache MinMaxScaler parameters for fast numpy-based scaling
    # (keeps float64 to match the offline training pipeline before converting to float32 for AE)
    scaler = detector.scaler
    scale_ = scaler.scale_ if hasattr(scaler, 'scale_') else None
    min_ = scaler.min_ if hasattr(scaler, 'min_') else None

    def _collect_batch():
        """Pull one batch from ml_q. Returns (batch_item, done) where done=True on sentinel."""
        try:
            item = ml_q.get(timeout=0.5)
        except Exception:
            return None, False

        if item is None:    # None is the sentinel sent by the FE thread when finished
            return None, True

        return item, False

    def _compute_latency_percentiles(batch_times):
        """Convert (time, batch_size) pairs to per-packet ms and return percentile stats."""
        if not batch_times:
            return {}
        per_pkt = np.array([t / b * 1000 for t, b in batch_times])
        return {
            "avg_ms": round(float(per_pkt.mean()), 4),
            "p50_ms": round(float(np.percentile(per_pkt, 50)), 4),
            "p95_ms": round(float(np.percentile(per_pkt, 95)), 4),
            "p99_ms": round(float(np.percentile(per_pkt, 99)), 4),
            "max_ms": round(float(per_pkt.max()), 4),
        }

    def _send_results():
        ml_cpu_total = time.process_time() - ml_cpu_start
        ml_elapsed = time.time() - start if start else 0

        e2e_stats = {}
        if e2e_latencies:
            e2e_ms = np.array(e2e_latencies) * 1000
            e2e_stats = {
                "avg_ms": round(float(e2e_ms.mean()), 4),
                "p50_ms": round(float(np.percentile(e2e_ms, 50)), 4),
                "p95_ms": round(float(np.percentile(e2e_ms, 95)), 4),
                "p99_ms": round(float(np.percentile(e2e_ms, 99)), 4),
                "max_ms": round(float(e2e_ms.max()), 4),
            }

        latency = {
            "ml_per_packet": _compute_latency_percentiles(batch_ml_times),
            "ae_per_packet": _compute_latency_percentiles(batch_ae_times),
            "e2e_per_packet": e2e_stats,
            "ml_process_cpu_s": round(ml_cpu_total, 2),
            "ml_wall_clock_s": round(ml_elapsed, 2),
            "ml_throughput_pps": round(stats["ml_processed"] / ml_elapsed, 1) if ml_elapsed > 0 else 0,
        }

        rf_n = stats["rf_calls"]
        if rf_n > 0:
            latency["rf_per_call_avg_ms"] = round(stats["rf_total_time"] / rf_n * 1000, 4)
            latency["rf_per_call_max_ms"] = round(stats["rf_max_time"] * 1000, 4)
            latency["rf_total_calls"] = rf_n

        result_q.put({
            "stats": stats,
            "attack_log_path": attack_log_path,
            "detector_counters": {
                "ae_flagged": det_ae_flagged,
                "rf_called": det_rf_called,
                "total": det_total,
                "attacks_final": det_attacks_final,
            },
            "latency": latency,
        })

    try:
        finished = False
        while not finished:
            batch_item, done = _collect_batch()

            if batch_item is not None:
                raw_matrix, batch = batch_item  # (B, 115) float32 matrix + metadata list
                batch_size = len(batch)
                ml_start = time.perf_counter()

                # --- AE: scale features and compute reconstruction error ---
                ae_start = time.perf_counter()

                # Apply MinMaxScaler in float64 (matches offline training), then cast to float32 for AE
                if scale_ is not None and min_ is not None:
                    raw_f64 = raw_matrix.astype(np.float64)
                    scaled_matrix = raw_f64 * scale_ + min_
                else:
                    scaled_matrix = scaler.transform(raw_matrix)

                x_t = torch.tensor(scaled_matrix, dtype=torch.float32, device=detector.device)
                with torch.no_grad():
                    recon = detector.ae(x_t)
                ae_errors = torch.mean((recon - x_t) ** 2, dim=1).cpu().numpy()  # shape (B,)

                ae_time = time.perf_counter() - ae_start
                stats["ae_total_time"] += ae_time
                stats["ae_max_time"] = max(stats["ae_max_time"], ae_time)
                batch_ae_times.append((ae_time, batch_size))

                # --- RF: run only on packets that exceeded the AE threshold ---
                ae_flagged_mask = ae_errors > detector.ae_threshold
                flagged_indices = np.where(ae_flagged_mask)[0]
                num_flagged = len(flagged_indices)

                rf_probs = np.full(batch_size, np.nan)
                rf_preds = np.zeros(batch_size, dtype=int)

                if num_flagged > 0:
                    rf_start = time.perf_counter()

                    # Transform flagged packets through RF's preprocessing step before inference
                    flagged_raw = raw_matrix[flagged_indices]
                    if detector.rf_num_cols is not None:
                        flagged_df = pd.DataFrame(flagged_raw, columns=detector.rf_num_cols)
                        flagged_transformed = detector.pre.transform(flagged_df)
                    else:
                        flagged_transformed = detector.pre.transform(flagged_raw)
                    probs_batch = detector.rf.predict_proba(flagged_transformed)[:, 1]
                    preds_batch = (probs_batch >= detector.rf_threshold).astype(int)

                    rf_time = time.perf_counter() - rf_start
                    stats["rf_total_time"] += rf_time
                    stats["rf_max_time"] = max(stats["rf_max_time"], rf_time)
                    stats["rf_calls"] += num_flagged

                    # Write RF results back into the full-batch arrays
                    rf_probs[flagged_indices] = probs_batch
                    rf_preds[flagged_indices] = preds_batch

                # --- Log results and update counters per packet ---
                t_decision = time.time()
                for i in range(batch_size):
                    item = batch[i]
                    ae_err = float(ae_errors[i])
                    flagged = bool(ae_flagged_mask[i])
                    final_pred = int(rf_preds[i])

                    e2e_latencies.append(t_decision - item["t_read"])

                    if flagged:
                        det_ae_flagged += 1
                        det_rf_called += 1
                        stats["ae_flagged"] += 1

                    det_total += 1
                    if final_pred == 1:
                        det_attacks_final += 1

                    stats["ml_processed"] += 1
                    if stats["ml_processed"] == 1:
                        start = time.time()
                        print(f"[*] First packet through ML pipeline (batch_size={batch_size})")

                    # Log every AE-flagged packet so offline threshold sweeps can replay decisions
                    if flagged:
                        prob = float(rf_probs[i])
                        if final_pred == 1:
                            stats["attacks_detected"] += 1
                        alert = {
                            "packet_num": item["pkt_num"],
                            "timestamp": item["timestamp"],
                            "src_ip": item["srcIP"],
                            "dst_ip": item["dstIP"],
                            "src_port": item["srcproto"],
                            "dst_port": item["dstproto"],
                            "ae_error": round(ae_err, 4),
                            "rf_prob": round(prob, 4),
                        }
                        attack_log_f.write(json.dumps(alert) + "\n")
                        # Forward RF-confirmed attacks to the live dashboard
                        if final_pred == 1 and ui_alerts_f:
                            uib.write_alert(ui_alerts_f, alert, AE_THRESHOLD, RF_THRESHOLD)

                ml_time = time.perf_counter() - ml_start
                stats["ml_total_time"] += ml_time
                stats["ml_max_time"] = max(stats["ml_max_time"], ml_time)
                batch_ml_times.append((ml_time, batch_size))

                # Periodic progress report
                if stats["ml_processed"] >= REPORT_INTERVAL and stats["ml_processed"] % REPORT_INTERVAL < batch_size and start:
                    elapsed = time.time() - start
                    pps = stats["ml_processed"] / elapsed
                    n = stats["ml_processed"]

                    avg_ml = (stats["ml_total_time"] / n) * 1000
                    avg_ae = (stats["ae_total_time"] / n) * 1000

                    rf_n = stats["rf_calls"]
                    if rf_n > 0:
                        avg_rf = (stats["rf_total_time"] / rf_n) * 1000
                        rf_str = f"avg={avg_rf:.2f}ms ({rf_n:,} calls)"
                    else:
                        rf_str = "0 calls"

                    print(
                        f"  [ML] {n:,} pkts | {pps:.0f} pps | "
                        f"AE: avg={avg_ae:.3f}ms/pkt | "
                        f"RF: {rf_str} | "
                        f"total: avg={avg_ml:.3f}ms/pkt | "
                        f"ae_flagged={stats['ae_flagged']} "
                        f"attacks={stats['attacks_detected']}"
                    )

                    # Rate-limited dashboard update (at most once every 5 seconds)
                    _now = time.time()
                    if uib and (_now - _ui_last_update >= 5.0):
                        uib.update_metrics(
                            status="running",
                            total_processed=det_total,
                            total_alerts=det_attacks_final,
                            ae_flagged=det_ae_flagged,
                            run_start=start,
                        )
                        _ui_last_update = _now

            if done:
                finished = True
    except (KeyboardInterrupt, BrokenPipeError):
        print("[ML process] Interrupted — saving partial results...")
    finally:
        try:
            attack_log_f.close()
        except Exception:
            pass
        try:
            if ui_alerts_f:
                ui_alerts_f.close()
        except Exception:
            pass
        # Push final status so the dashboard shows "stopped" with accurate totals
        if uib:
            try:
                uib.update_metrics(
                    status="stopped",
                    total_processed=det_total,
                    total_alerts=det_attacks_final,
                    ae_flagged=det_ae_flagged,
                    run_start=start or time.time(),
                )
            except Exception:
                pass
        _send_results()


# =============================================================================
# MAIN
# =============================================================================

def _get_peak_memory_mb():
    """Return peak memory usage of this process in MB, or None if unavailable."""
    try:
        import resource    # Linux/macOS only
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:
        pass
    try:
        import psutil
        return psutil.Process().memory_info().peak_wset / (1024 * 1024)
    except Exception:
        return None


def main():
    wall_start = time.time()
    cpu_start = time.process_time()

    if not os.path.exists(REPLAY_PCAP):
        print(f"[ERROR] PCAP file not found: {REPLAY_PCAP}")
        sys.exit(1)

    pcap_size = os.path.getsize(REPLAY_PCAP) / 1e6

    print("=" * 70)
    print("  HYBRID NETWORK INTRUSION DETECTOR — PCAP REPLAY")
    print("=" * 70)
    print(f"  PCAP file:      {REPLAY_PCAP} ({pcap_size:.1f} MB)")
    print(f"  Feature dim:    {FEATURE_DIM}")
    print(f"  AE threshold:   {AE_THRESHOLD}")
    print(f"  RF threshold:   {RF_THRESHOLD}")
    print(f"  ML batch size:  {ML_BATCH_SIZE}")
    print(f"  Warm-up pkts:   {WARMUP_PACKETS:,}" + (" (disabled)" if WARMUP_PACKETS == 0 else ""))
    print(f"  Architecture:   2 threads + 1 process (GIL-free ML)")
    print(f"  Alert log:      {ALERT_LOG}")
    print()

    # Load models in the main process first to verify file paths before spawning subprocesses
    print("[*] Loading hybrid ML models (AE + RF)...")
    try:
        detector = OnlineSequentialAERF(
            ae_threshold=AE_THRESHOLD,
            rf_threshold=RF_THRESHOLD,
        )
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(f"  AE input dim: {detector.input_dim}")
    print(f"  Models loaded successfully")
    print()

    # Set up inter-process communication queues and a shared stop event
    ml_q = mp.Queue(maxsize=ML_QUEUE_MAX)
    stop_event = mp.Event()
    result_q = mp.Queue()

    # Launch tshark to read the PCAP and emit each packet as a tab-separated row
    tshark_cmd = [
        TSHARK,
        "-r", REPLAY_PCAP,
        "-T", "fields",
        "-e", "frame.time_epoch",
        "-e", "frame.len",
        "-e", "eth.src",
        "-e", "eth.dst",
        "-e", "ip.src",
        "-e", "ip.dst",
        "-e", "tcp.srcport",
        "-e", "tcp.dstport",
        "-e", "udp.srcport",
        "-e", "udp.dstport",
        "-e", "icmp.type",
        "-e", "icmp.code",
        "-e", "arp.opcode",
        "-e", "arp.src.hw_mac",
        "-e", "arp.src.proto_ipv4",
        "-e", "arp.dst.hw_mac",
        "-e", "arp.dst.proto_ipv4",
        "-e", "ipv6.src",
        "-e", "ipv6.dst",
        "-E", "occurrence=f",
        "-E", "header=n",
        "-E", "separator=\t",
    ]

    print("[*] Starting tshark replay...")
    tshark_proc = subprocess.Popen(
        tshark_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )

    # Start reader and FE as daemon threads; ML runs in a separate process
    reader_t = threading.Thread(target=reader, args=(tshark_proc, stop_event), daemon=True, name="reader")
    fe_t = threading.Thread(target=feature_extractor, args=(ml_q, stop_event), daemon=True, name="feature_extractor")
    ml_proc = mp.Process(target=ml_detector_process, args=(ml_q, result_q), name="ml_detector")

    reader_t.start()
    fe_t.start()
    ml_proc.start()

    print("[*] Replay running (FE=thread, ML=process)...\n")

    # Wait for the full pipeline to drain in order
    try:
        tshark_proc.wait()
        print("[*] tshark finished reading PCAP")

        reader_t.join(timeout=30)
        print(f"[*] Reader done. Waiting for FE to drain packet_q ({packet_q.qsize():,} remaining)...")

        last_report = time.time()
        while not packet_q.empty() and fe_t.is_alive():
            time.sleep(1)
            if time.time() - last_report >= 5:
                try:
                    mlq_size = ml_q.qsize()
                except NotImplementedError:
                    mlq_size = -1
                print(f"  Draining: pkt_q={packet_q.qsize():,} ml_q={mlq_size:,}")
                last_report = time.time()

        fe_t.join(timeout=30)
        print(f"[*] FE done. Waiting for ML process to finish...")

        # Read from result_q BEFORE joining to prevent deadlock on large result payloads.
        # Timeout covers worst-case ML processing time (661k packets @ ~1200 pps = ~550s).
        ml_results = None
        try:
            ml_results = result_q.get(timeout=3600)
        except Exception:
            pass
        ml_proc.join(timeout=600)
        if ml_proc.is_alive():
            ml_proc.terminate()
            ml_proc.join(timeout=5)

    except KeyboardInterrupt:
        print("\n[*] Interrupted.")
        stop_event.set()
        # Send sentinel to unblock the ML process if it's waiting on an empty queue
        try:
            ml_q.put_nowait(None)
        except Exception:
            pass
        tshark_proc.terminate()
        fe_t.join(timeout=10)
        ml_results = None
        try:
            ml_results = result_q.get(timeout=10)
        except Exception:
            pass
        ml_proc.join(timeout=10)
        if ml_proc.is_alive():
            ml_proc.terminate()
            ml_proc.join(timeout=5)

    if ml_results is None:
        print("[WARNING] Could not retrieve ML results")

    wall_elapsed = time.time() - wall_start
    cpu_elapsed = time.process_time() - cpu_start

    ml_stats = ml_results["stats"] if ml_results else {}
    det_counters = ml_results["detector_counters"] if ml_results else {}
    ml_latency = ml_results.get("latency", {}) if ml_results else {}

    # Read attack entries from the temporary JSONL file written by the ML process
    attack_log = []
    attack_log_path = ml_results.get("attack_log_path", ALERT_LOG + ".attacks.jsonl") if ml_results else ""
    if attack_log_path and os.path.exists(attack_log_path):
        with open(attack_log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        attack_log.append(json.loads(line))
                    except Exception:
                        pass
        os.remove(attack_log_path)

    # --- Print detection summary ---
    print()
    print("=" * 70)
    print("  DETECTION SUMMARY")
    print("=" * 70)

    fe_count = fe_stats["packet_count"]
    ml_count = ml_stats.get("ml_processed", 0)
    ae_flagged = det_counters.get("ae_flagged", 0)
    rf_called = det_counters.get("rf_called", 0)
    total_det = det_counters.get("total", 0)
    attacks_final = det_counters.get("attacks_final", 0)

    ae_frac = ae_flagged / total_det if total_det > 0 else 0
    overall_pps = fe_count / wall_elapsed if wall_elapsed > 0 else 0

    print(f"  PCAP:                  {REPLAY_PCAP}")
    print(f"  Total packets (FE):    {fe_count:,}")
    print(f"  Warm-up (FE only):     {WARMUP_PACKETS:,}")
    print(f"  Total packets (ML):    {ml_count:,}")
    print(f"  AE flagged:            {ae_flagged:,} ({ae_frac:.2%})")
    print(f"  RF called:             {rf_called:,}")
    print(f"  Attacks detected:      {attacks_final:,}")
    print(f"  Malformed:             {fe_stats['malformed']:,}")
    print()

    print(f"  --- Throughput ---")
    print(f"  Wall clock:            {wall_elapsed:.1f}s")
    print(f"  Overall throughput:    {overall_pps:.0f} pps (end-to-end)")
    ml_pps = ml_latency.get("ml_throughput_pps", 0)
    if ml_pps:
        print(f"  ML throughput:         {ml_pps:.0f} pps")
    print()

    print(f"  --- Detection Latency (per packet) ---")
    if fe_count > 0:
        fe_avg = fe_stats["fe_total_time"] / fe_count * 1000
        fe_max = fe_stats["fe_max_time"] * 1000
        print(f"  FE (AfterImage):       avg={fe_avg:.2f}ms  max={fe_max:.1f}ms")

    ml_pkt = ml_latency.get("ml_per_packet", {})
    if ml_pkt:
        print(f"  ML total (AE+RF):      avg={ml_pkt.get('avg_ms',0):.3f}ms  p50={ml_pkt.get('p50_ms',0):.3f}ms  p95={ml_pkt.get('p95_ms',0):.3f}ms  p99={ml_pkt.get('p99_ms',0):.3f}ms  max={ml_pkt.get('max_ms',0):.3f}ms")

    ae_pkt = ml_latency.get("ae_per_packet", {})
    if ae_pkt:
        print(f"  AE only (scale+fwd):   avg={ae_pkt.get('avg_ms',0):.3f}ms  p50={ae_pkt.get('p50_ms',0):.3f}ms  p95={ae_pkt.get('p95_ms',0):.3f}ms  max={ae_pkt.get('max_ms',0):.3f}ms")

    if ml_latency.get("rf_total_calls", 0) > 0:
        print(f"  RF (when called):      avg={ml_latency.get('rf_per_call_avg_ms',0):.2f}ms  max={ml_latency.get('rf_per_call_max_ms',0):.2f}ms  ({ml_latency.get('rf_total_calls',0):,} calls)")

    e2e = ml_latency.get("e2e_per_packet", {})
    if e2e:
        print(f"  E2E (read->decision):  avg={e2e.get('avg_ms',0):.2f}ms  p50={e2e.get('p50_ms',0):.2f}ms  p95={e2e.get('p95_ms',0):.2f}ms  p99={e2e.get('p99_ms',0):.2f}ms  max={e2e.get('max_ms',0):.2f}ms")
    print()

    print(f"  --- Resource Usage ---")
    print(f"  Main process CPU:      {cpu_elapsed:.1f}s")
    ml_cpu = ml_latency.get("ml_process_cpu_s", 0)
    if ml_cpu:
        print(f"  ML process CPU:        {ml_cpu:.1f}s")
    peak_mem = _get_peak_memory_mb()
    if peak_mem is not None:
        print(f"  Peak memory (main):    {peak_mem:.1f} MB")

    # Write full results to the alert log JSON file
    log_dir = os.path.dirname(ALERT_LOG)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    fe_latency = {}
    if fe_count > 0:
        fe_latency = {
            "avg_ms": round(fe_stats["fe_total_time"] / fe_count * 1000, 4),
            "max_ms": round(fe_stats["fe_max_time"] * 1000, 4),
        }

    log_output = {
        "summary": {
            "pcap_file": REPLAY_PCAP,
            "total_packets_fe": fe_count,
            "total_packets_ml": ml_count,
            "warmup_packets": WARMUP_PACKETS,
            "ae_flagged": ae_flagged,
            "ae_flagged_fraction": round(ae_frac, 4),
            "rf_called": rf_called,
            "attacks_detected": attacks_final,
            "ae_threshold": AE_THRESHOLD,
            "rf_threshold": RF_THRESHOLD,
        },
        "performance": {
            "wall_clock_s": round(wall_elapsed, 2),
            "overall_throughput_pps": round(overall_pps, 1),
            "ml_throughput_pps": ml_latency.get("ml_throughput_pps", 0),
            "fe_latency": fe_latency,
            "ml_latency_per_packet": ml_latency.get("ml_per_packet", {}),
            "ae_latency_per_packet": ml_latency.get("ae_per_packet", {}),
            "e2e_latency_per_packet": ml_latency.get("e2e_per_packet", {}),
            "rf_latency_per_call": {
                "avg_ms": ml_latency.get("rf_per_call_avg_ms", 0),
                "max_ms": ml_latency.get("rf_per_call_max_ms", 0),
                "total_calls": ml_latency.get("rf_total_calls", 0),
            } if ml_latency.get("rf_total_calls", 0) > 0 else {},
            "resource_usage": {
                "main_process_cpu_s": round(cpu_elapsed, 2),
                "ml_process_cpu_s": ml_latency.get("ml_process_cpu_s", 0),
                "ml_process_wall_s": ml_latency.get("ml_wall_clock_s", 0),
                "peak_memory_mb": round(peak_mem, 1) if peak_mem is not None else None,
            },
        },
        "attacks": attack_log,
    }

    with open(ALERT_LOG, "w") as f:
        json.dump(log_output, f, indent=2)
    print(f"\n  Attack log written to: {ALERT_LOG}")
    print("=" * 70)


if __name__ == "__main__":
    main()
