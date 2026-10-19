#!/usr/bin/env python3
"""
Live Network Intrusion Detector — Linux

Pipeline is as follows:
  dumpcap (capture) → dpkt → AfterImage (115 features) → Hybrid ML (AE→RF) → alert

Architecture (1 thread + 1 process):
    Main process: Reader + FE thread:  dpkt → [parse packets → AfterImage] → batched ml_q
    ML process:  ml_q → [batched AE → batched RF] → alert + JSON log

"""

import json
import socket
import struct
import subprocess
import sys
import time
import threading
import multiprocessing as mp
import dpkt
import gc
import ctypes
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# ─────────────────────────────Imports─────────────────────────────
try:
    import NetworkIngestion.netStat as ns
except ImportError:
    import netStat as ns


# ─────────────CONFIGURATION ───────────────────────────────────────
DUMPCAP = "/usr/bin/dumpcap"
TCPDUMP = "/usr/bin/tcpdump"  # fallback if dumpcap unavailable

# Network interface
INTERFACE = "veth_detect"

# Network namespace to run capture in 
NAMESPACE = "detect_ns"

# Capture buffer size in MB
BUFFER_MB = 256

# ─── QUEUE / BATCH SIZES ─────────────────────────────────────────
ML_QUEUE_MAX = 50000         # Feature vectors waiting for ML pipieline 
FE_BATCH_SIZE = 64           # Packets batched per ml_q.put

# Reporting interval
REPORT_INTERVAL = 10000

# GC interval: force malloc_trim every N packets
GC_INTERVAL = 100_000

# ML batch size — process N vectors through AE in one forward pass
ML_BATCH_SIZE = 256

# ─── ML THRESHOLDS ────────────────────────────────────────────────
AE_THRESHOLD = 2.8   
RF_THRESHOLD = 0.5   

# ─── WARM-UP ─────────────────────────────────────────────────────
# number of packets to run through AfterImage before starting ML inference and alerting.
# Set to 0 to disable and score all packets from the start
WARMUP_PACKETS = 86890

# ─── OUTPUT ───────────────────────────────────────────────────────
ALERT_LOG = "NetworkIngestion/attack_log.json"

# ─── GROUND-TRUTH LABELS ─────────────────────────────────────────
TEST_SIZE  = 330204   # last N packets in the dataset used for evaluation
LABELS_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "UI", "OS Scan_labels.csv"
)
    
# ─── AfterImage Feature Engine Initialization ───────────────
maxHost = 100000
maxSess = 100000
nstat = ns.netStat(np.nan, maxHost, maxSess)
headers = nstat.getNetStatHeaders()
FEATURE_DIM = len(headers)

# ─── Main-thread state (combined reader+FE) ──────────────────────
fe_stats = {
    "packet_count": 0,        
    "read_count": 0,          
    "malformed": 0, 
    "fe_total_time": 0.0,
    "fe_max_time": 0.0,
    "parse_total_time": 0.0,  
    "qput_total_time": 0.0,   # np.array + batched ml_q.put (amortised)
}


#  ────────────────────────────────────────────
#  CAPTURE PIPELINE (dumpcap → dpkt)
# ────────────────────────────────────────────

def check_command(cmd):
    try:
        result = subprocess.run(["which", cmd], capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False


def build_capture_pipeline():
    """Start capture pipeline.

    pcap_stream is a raw binary pcap pipe read directly by dpkt inside
    reader_feature_extractor (thread) 
    """
    use_dumpcap = check_command(DUMPCAP)

    # dumpcap writes raw pcap to stdout, dpkt reads it directly (no intermediate files, no tshark)
    if use_dumpcap:
        capture_cmd = [DUMPCAP, "-i", INTERFACE, "-B", str(BUFFER_MB), "-P", "-w", "-"]
        if NAMESPACE:
            capture_cmd = ["ip", "netns", "exec", NAMESPACE] + capture_cmd
        print(f"[*] Starting dumpcap...")

    #create subprocess with stdout as a pipe, stderr to DEVNULL, and unbuffered output
    capture_proc = subprocess.Popen(
        capture_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0,
    )

    # wait for the subprocess to start and bind to the interface
    time.sleep(1) 
    if capture_proc.poll() is not None:
        raise RuntimeError("Capture process failed to start "
                           "(check interface/namespace/permissions)")

    print(f"[*] Capture started on {INTERFACE}")
    return capture_proc, capture_proc.stdout   # binary pcap stream


def _fmt_mac(b):
    """Format raw MAC bytes to 'aa:bb:cc:dd:ee:ff' string (matches tshark eth.src/dst)."""
    return ':'.join('%02x' % x for x in b)


#pcap global-header magic numbers
_PCAP_MAGIC_LE    = 0xa1b2c3d4   # little-endian microseconds
_PCAP_MAGIC_NS_LE = 0xa1b23c4d   # little-endian nanoseconds
_PCAP_MAGIC_BE    = 0xd4c3b2a1   # big-endian microseconds
_PCAP_MAGIC_NS_BE = 0x4d3cb2a1   # big-endian nanoseconds


def _pcap_stream_reader(fileobj):
    """Get (timestamp_float, origlen, buf) from a raw pcap binary stream.
    """
    ghdr = fileobj.read(24) # pcap global header is 24 bytes
    if len(ghdr) < 24: # EOF or malformed stream
        return
    magic = ghdr[0] | ghdr[1] << 8 | ghdr[2] << 16 | ghdr[3] << 24 # determine endianness and timestamp resolution from magic number
    if magic in (_PCAP_MAGIC_LE, _PCAP_MAGIC_NS_LE): # little-endian microseconds or nanoseconds
        endian, nanosec = '<', (magic == _PCAP_MAGIC_NS_LE)
    elif magic in (_PCAP_MAGIC_BE, _PCAP_MAGIC_NS_BE): #big-endian microseconds or nanoseconds
        endian, nanosec = '>', (magic == _PCAP_MAGIC_NS_BE)
    else:
        print(f"[FE] WARNING: unrecognised pcap magic {magic:#010x} — cannot parse stream")
        return 
    phdr_fmt = endian + 'IIII'   # ts_sec, ts_frac, caplen, origlen
    phdr_len = 16

    while True:
        # Read packet header + payload directly from the binary stream, yield parsed fields.
        hdr = fileobj.read(phdr_len)
        if len(hdr) < phdr_len:
            break
        ts_sec, ts_frac, caplen, origlen = struct.unpack(phdr_fmt, hdr)
        buf = fileobj.read(caplen)
        if len(buf) < caplen:
            break
        ts = ts_sec + (ts_frac * 1e-9 if nanosec else ts_frac * 1e-6) # timestamp as float seconds since epoch
        yield ts, origlen, buf


#────────────────────────────────────────────
#  THREAD — COMBINED READER + FEATURE EXTRACTOR 
# ────────────────────────────────────────────

def reader_feature_extractor(pcap_stream, ml_q, stop_event, warmup_done_event):
    """Read raw pcap from dumpcap, parse with dpkt, extract AfterImage features.
    """

    batch_vecs     = []
    batch_treads   = []
    batch_pkt_nums = []
    batch_meta     = []
    read_start     = None

    for ts, origlen, buf in _pcap_stream_reader(pcap_stream): 
        if stop_event.is_set(): # Check for stop signal from main thread (e.g. on Ctrl+C) to exit cleanly
            break

        t_read = time.time() # wall-clock read time for latency measurement in ML process
        fe_stats["read_count"] += 1 # Count packets read 

        if fe_stats["read_count"] == 1:
            read_start = time.time()
            print("[*] First packet from capture")

        # ── Parse packet with dpkt
        parse_start = time.perf_counter()

        # Defaults for missing/unknown fields
        IPtype    = np.nan
        timestamp = str(ts)      # float epoch as string
        framelen  = origlen      # original wire length
        srcMAC = dstMAC = srcIP = dstIP = srcproto = dstproto = '' 

        try:
            eth = dpkt.ethernet.Ethernet(buf) # parse raw packet bytes into dpkt Ethernet object
        except Exception:
            fe_stats["malformed"] += 1
            continue

        srcMAC = _fmt_mac(eth.src) # format raw MAC bytes to string 
        dstMAC = _fmt_mac(eth.dst) # format raw MAC bytes to string

        if isinstance(eth.data, dpkt.ip.IP): # IPv4
            ip    = eth.data
            IPtype = 0
            srcIP  = socket.inet_ntoa(ip.src)
            dstIP  = socket.inet_ntoa(ip.dst)
            if isinstance(ip.data, dpkt.tcp.TCP): # TCP
                srcproto = str(ip.data.sport)
                dstproto = str(ip.data.dport)
            elif isinstance(ip.data, dpkt.udp.UDP): # UDP
                srcproto = str(ip.data.sport)
                dstproto = str(ip.data.dport)
            elif isinstance(ip.data, dpkt.icmp.ICMP): # ICMP (protocol number 1)
                srcproto = 'icmp'
                dstproto = 'icmp'

        elif isinstance(eth.data, dpkt.ip6.IP6): # IPv6
            ip6    = eth.data
            IPtype = 1
            srcIP  = socket.inet_ntop(socket.AF_INET6, ip6.src)
            dstIP  = socket.inet_ntop(socket.AF_INET6, ip6.dst)
            if isinstance(ip6.data, dpkt.tcp.TCP): # TCP
                srcproto = str(ip6.data.sport)
                dstproto = str(ip6.data.dport)
            elif isinstance(ip6.data, dpkt.udp.UDP): # UDP
                srcproto = str(ip6.data.sport)
                dstproto = str(ip6.data.dport)

        elif isinstance(eth.data, dpkt.arp.ARP): # ARP (non-IP)
            arp    = eth.data
            IPtype = 0
            try:
                srcIP = socket.inet_ntoa(arp.spa)   # sender protocol address
                dstIP = socket.inet_ntoa(arp.tpa)   # target protocol address
            except Exception:
                srcIP = dstIP = ''
            srcproto = 'arp'
            dstproto = 'arp'

        else:
            # Unknown L3 — use MACs as identifiers
            if srcIP == '' and srcproto == '' and dstIP == '' and dstproto == '':
                srcIP = srcMAC
                dstIP = dstMAC

        fe_stats["parse_total_time"] += time.perf_counter() - parse_start 

        # ── AfterImage feature extraction
        try:
            fe_start = time.perf_counter() 
            vec = nstat.updateGetStats( # update internal state and extract features as numpy array
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

        # ── Warm-up: run AfterImage but skip ML ──
        if WARMUP_PACKETS > 0 and fe_stats["packet_count"] <= WARMUP_PACKETS:
            n_wu = fe_stats["packet_count"]
            if n_wu % REPORT_INTERVAL == 0:
                pct = n_wu / WARMUP_PACKETS * 100
                print(f"  [WARMUP] {n_wu:,}/{WARMUP_PACKETS:,} ({pct:.1f}%)")
            if n_wu % GC_INTERVAL == 0:
                gc.collect()
                try:
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                except Exception:
                    pass
            if n_wu == WARMUP_PACKETS:
                print(f"  [WARMUP] Complete — switching to detection mode")
                warmup_done_event.set()
            continue

        # ── Accumulate into batch ──
        batch_vecs.append(vec)
        batch_treads.append(t_read)
        batch_pkt_nums.append(fe_stats["packet_count"])
        batch_meta.append((timestamp, srcIP, dstIP, srcproto, dstproto))

        if len(batch_vecs) >= FE_BATCH_SIZE: #Once we reach the batch size limit
            qput_start = time.perf_counter() #start the timer
            ml_q.put({ # put the stacked vectors into the queue
                "vecs":     np.array(batch_vecs,     dtype=np.float32),
                "t_read":   np.array(batch_treads),
                "pkt_nums": np.array(batch_pkt_nums, dtype=np.int32),
                "meta":     batch_meta,
            }, block=True)
            fe_stats["qput_total_time"] += time.perf_counter() - qput_start # calculate amortized put time
            batch_vecs = []; batch_treads = []; batch_pkt_nums = []; batch_meta = []

        #Simple incremental progress updates to the terminal
        #for every REPORT_INTERVAL packets processed
        if fe_stats["packet_count"] % REPORT_INTERVAL == 0: 
            n = fe_stats["packet_count"]
            avg_parse = (fe_stats["parse_total_time"] / n) * 1000
            avg_fe    = (fe_stats["fe_total_time"]    / n) * 1000
            avg_qput  = (fe_stats["qput_total_time"]  / n) * 1000
            try:
                mlq_size = ml_q.qsize()
            except NotImplementedError:
                mlq_size = -1
            pps = n / (time.time() - read_start) if read_start else 0
            print(
                f"  [FE] {n:,} pkts | {pps:.0f} pps | "
                f"parse={avg_parse:.3f} fe={avg_fe:.3f} qput={avg_qput:.3f} ms/pkt | "
                f"ml_q={mlq_size:,}"
            )

    # Flush remaining packets
    if batch_vecs:
        ml_q.put({
            "vecs":     np.array(batch_vecs,     dtype=np.float32),
            "t_read":   np.array(batch_treads),
            "pkt_nums": np.array(batch_pkt_nums, dtype=np.int32),
            "meta":     batch_meta,
        }, block=True)

    warmup_done_event.set()   # safety: unblock main if stream ended before warmup
    ml_q.put(None)
    stop_event.set()


# ══════════════════════════════════════════════════════════════════
#  PROCESS — ML DETECTOR: Runs in a SEPARATE Process
# ══════════════════════════════════════════════════════════════════

def ml_detector_process(ml_q, result_q):
    """ML inference in a separate process to bypass GIL ."""
    import traceback

    try:
        _ml_detector_inner(ml_q, result_q)
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
        result_q.put({
            "stats": {"ml_processed": 0, "attacks_detected": 0, "ae_flagged": 0},
            "attack_log": [],
            "detector_counters": {"ae_flagged": 0, "rf_called": 0, "total": 0, "attacks_final": 0},
        })


def _ml_detector_inner(ml_q, result_q):
    """
        Inner function for ML detection process.
        Calls the OnlineSequentialAERF function to load the AE and RF models, 
        then processes batches of feature vectors from ml_q.
    """
    import torch
    from scripts.Hybrid.online_sequential_pipeline import OnlineSequentialAERF

    # Load models inside child process 
    print("[ML process] Loading models on CPU...")
    detector = OnlineSequentialAERF(
        ae_threshold=AE_THRESHOLD,
        rf_threshold=RF_THRESHOLD,
        device="cpu",
    )
    print(f"[ML process] Models loaded (device=cpu)")

    # ── Ground-truth labels for P/R/F1 evaluation ──
    # This calculates a running P/R/F1 score as online inference proceeds
    _labels_arr = None
    if os.path.exists(LABELS_CSV):
        import csv as _csv
        _tmp = []
        with open(LABELS_CSV) as _f:
            _rdr = _csv.reader(_f)
            next(_rdr)
            for _row in _rdr:
                _tmp.append(int(_row[1]))
        _labels_arr = np.array(_tmp[-TEST_SIZE:], dtype=np.int8)
        print(f"[ML process] Loaded {len(_labels_arr)} test labels → P/R/F1 enabled")
    else:
        print(f"[ML process] Labels not found ({LABELS_CSV}) → P/R/F1 disabled")

    # Now we initiatize UI Bridge, allowing live feed to dashboard
    # Each alert is a cheap buffered write 
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
        print("[ML process] UI bridge initialised → UI/data/")
    except Exception as _e:
        print(f"[ML process] UI bridge unavailable ({_e}) — dashboard feed disabled")
        uib = None
        ui_alerts_f = None
    _ui_last_update = 0.0   # wall-clock time of last metrics.json write

    # Local state for stats and counters
    # main process reads these after ML process finishes
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
    # Stream attack entries to disk
    # Avoids memory leaks from .json accumulation.
    attack_log_path = ALERT_LOG + ".attacks.jsonl"
    attack_log_f = open(attack_log_path, "w", buffering=64 * 1024)
    start = None

    # Per-batch latency samples for percentile computation
    batch_ml_times = []  
    batch_ae_times = []   

    # End-to-end latency: time from packet read to ML decision, per packet
    e2e_latencies = []

    # Detector counters for final report
    det_ae_flagged = 0
    det_rf_called = 0
    det_total = 0
    det_attacks_final = 0
    det_tp = det_fp = det_fn = det_tn = 0

    def _calc_hybrid_metrics(tp, fp, fn, tn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    # CPU time tracking for ML process
    ml_cpu_start = time.process_time()

    # Pre-extract scaler params for fast numpy scaling
    scaler = detector.scaler
    scale_ = scaler.scale_ if hasattr(scaler, 'scale_') else None   
    min_ = scaler.min_ if hasattr(scaler, 'min_') else None        

    # Pre-extract RF column indices for fast numpy path 
    # No need for columntransform, vector is already 115 dim and in correct order
    rf_model = detector.rf
    _rf_bypass_pre = False  

    def _collect_batch():
        """
        Drain up to ML_BATCH_SIZE items from ml_q. 

        Items arrive as dicts with pre-stacked numpy arrays from FE.
        Collects multiple FE chunks and concatenates their arrays.
        Returns a single dict with concatenated arrays -- (batch_dict, done)., or None if empty.

        """
        chunks = []
        total = 0
        done = False
        while total < ML_BATCH_SIZE:
            try:
                timeout = 0.5 if total == 0 else 0.001
                chunk = ml_q.get(timeout=timeout)
            except Exception:  # queue.Empty
                break

            if chunk is None:  # Sentinel from FE is all done
                done = True
                break

            chunks.append(chunk)
            total += len(chunk["vecs"])

        if not chunks:
            return None, done

        # Concatenate pre-stacked arrays from multiple FE batches
        if len(chunks) == 1:
            return chunks[0], done
        return {
            "vecs": np.concatenate([c["vecs"] for c in chunks]),
            "t_read": np.concatenate([c["t_read"] for c in chunks]),
            "pkt_nums": np.concatenate([c["pkt_nums"] for c in chunks]),
            "meta": sum([c["meta"] for c in chunks], []),
        }, done

    def _compute_latency_percentiles(batch_times):
        """Convert (time, batch_size) pairs to per-packet ms and compute percentiles."""
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

        # End-to-end latency percentiles 
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

        rf_n = stats.get("rf_calls", 0)
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

    # Main ML processing loop: collect batches from ml_q, run through AE+RF, evaluate, log alerts, update stats.
    # Wrapped in Try/Except to ensure we send results back to main process even if something goes wrong here.
    try:
        finished = False
        while not finished:
            batch, done = _collect_batch()

            if batch:
                batch_size = len(batch["vecs"])
                ml_start = time.perf_counter()

                # Batched AE to scale + forward and detetermine MSE in one pass
                ae_start = time.perf_counter()
                raw_matrix = batch["vecs"]

                # Scale to float64 if needed to match offline testing parameters.
                if scale_ is not None and min_ is not None:
                    raw_f64 = raw_matrix.astype(np.float64)
                    scaled_matrix = raw_f64 * scale_ + min_
                else:
                    scaled_matrix = scaler.transform(raw_matrix)

                # Batch AE forward pass
                x_t = torch.tensor(scaled_matrix, dtype=torch.float32, device=detector.device)
                with torch.no_grad():
                    recon = detector.ae(x_t)
                ae_errors = torch.mean((recon - x_t) ** 2, dim=1).cpu().numpy()  # (B,)

                ae_time = time.perf_counter() - ae_start
                stats["ae_total_time"] += ae_time
                stats["ae_max_time"] = max(stats["ae_max_time"], ae_time)
                batch_ae_times.append((ae_time, batch_size))

                # Check and save the indices of the samples that exceed the AE threshold
                ae_flagged_mask = ae_errors > detector.ae_threshold
                flagged_indices = np.where(ae_flagged_mask)[0]
                num_flagged = len(flagged_indices)

                # Send those flagged samples through the RF for final decision
                
                # Prepare batched arrays for RF results
                rf_probs = np.full(batch_size, np.nan)
                rf_preds = np.zeros(batch_size, dtype=int)

                if num_flagged > 0:
                    rf_start = time.perf_counter()

                    flagged_raw = raw_matrix[flagged_indices]  #populate with AE flagged indices
                    if _rf_bypass_pre:
                        # Fast path: skip pandas DataFrame and ColumnTransformer 
                        probs_batch = rf_model.predict_proba(flagged_raw)[:, 1]
                    elif detector.rf_num_cols is not None:
                        flagged_df = pd.DataFrame(flagged_raw, columns=detector.rf_num_cols) #create dataframe
                        flagged_transformed = detector.pre.transform(flagged_df) #transform with columntransformer
                        probs_batch = detector.rf.predict_proba(flagged_transformed)[:, 1] #predict with RF
                    else:
                        flagged_transformed = detector.pre.transform(flagged_raw)
                        probs_batch = detector.rf.predict_proba(flagged_transformed)[:, 1]
                    preds_batch = (probs_batch >= detector.rf_threshold).astype(int)

                    #append and increment stats for RF latency and calls
                    rf_time = time.perf_counter() - rf_start
                    stats["rf_total_time"] += rf_time
                    stats["rf_max_time"] = max(stats["rf_max_time"], rf_time)
                    stats["rf_calls"] += num_flagged

                    # Scatter back into full-batch arrays
                    rf_probs[flagged_indices] = probs_batch
                    rf_preds[flagged_indices] = preds_batch

                #Ground-truth P/R/F1 evaluation
                # rf predicition is 0 for any non AE-flagged packet
                # test set is last N packets, so index is pkt_num - warmup - 1
                if _labels_arr is not None:
                    _lbl_idx = batch["pkt_nums"].astype(np.int64) - WARMUP_PACKETS - 1 
                    _in_test = (_lbl_idx >= 0) & (_lbl_idx < TEST_SIZE)
                    if _in_test.any():
                        _v = _lbl_idx[_in_test] #array of indices into labels for packets in test set
                        _t = _labels_arr[_v] #true labels for packets in test set
                        _p = rf_preds[_in_test] #predicted labels for packets in test set (0 or 1)
                        det_tp += int(((_t == 1) & (_p == 1)).sum())
                        det_fp += int(((_t == 0) & (_p == 1)).sum())
                        det_fn += int(((_t == 1) & (_p == 0)).sum())
                        det_tn += int(((_t == 0) & (_p == 0)).sum())

                # Per-sample bookkeeping and alert logging
                t_decision = time.time()
                b_treads = batch["t_read"]
                b_pkt_nums = batch["pkt_nums"]
                b_meta = batch["meta"]

                # Bulk E2E latency
                e2e_latencies.extend(t_decision - b_treads)

                # Count AE flags
                num_flagged_total = int(ae_flagged_mask.sum())
                det_ae_flagged += num_flagged_total
                det_rf_called += num_flagged_total
                stats["ae_flagged"] += num_flagged_total
                det_total += batch_size

                # Count attacks
                attack_mask = rf_preds == 1
                num_attacks = int(attack_mask.sum())
                det_attacks_final += num_attacks
                stats["attacks_detected"] += num_attacks
                stats["ml_processed"] += batch_size

                # Start message and timer at first ML-processed packet (i.e. after warmup)
                if stats["ml_processed"] >= batch_size and start is None:
                    start = time.time()
                    print("[*] First packet processed through ML pipeline")

                # Log ALL AE-flagged packets for offline threshold sweep
                # attack_log stores every AE-flagged entry 
                if num_flagged > 0:
                    for i in flagged_indices:
                        ts, srcIP, dstIP, srcproto, dstproto = b_meta[i]
                        _li = int(b_pkt_nums[i]) - WARMUP_PACKETS - 1
                        alert = {
                            "packet_num": int(b_pkt_nums[i]),
                            "timestamp": ts,
                            "src_ip": srcIP,
                            "dst_ip": dstIP,
                            "src_port": srcproto,
                            "dst_port": dstproto,
                            "ae_error": round(float(ae_errors[i]), 4),
                            "rf_prob": round(float(rf_probs[i]), 4),
                            "true_label": int(_labels_arr[_li]) if (_labels_arr is not None and 0 <= _li < TEST_SIZE) else None,
                        }
                        attack_log_f.write(json.dumps(alert) + "\n")
                        # Push RF-confirmed attacks to the live dashboard
                        if rf_preds[i] == 1 and ui_alerts_f:
                            uib.write_alert(ui_alerts_f, alert, AE_THRESHOLD, RF_THRESHOLD)

                #caclulate the ML time for this batch and update stats
                ml_time = time.perf_counter() - ml_start
                stats["ml_total_time"] += ml_time
                stats["ml_max_time"] = max(stats["ml_max_time"], ml_time)
                batch_ml_times.append((ml_time, batch_size))

                # Periodic report to the terminal
                if stats["ml_processed"] >= REPORT_INTERVAL and stats["ml_processed"] % REPORT_INTERVAL < batch_size and start:
                    elapsed = time.time() - start
                    pps = stats["ml_processed"] / elapsed

                    print(
                        f"  [{stats['ml_processed']:,} pkts | {pps:.0f} pps] "
                        f"attacks={stats['attacks_detected']} "
                        f"ae_flagged={stats['ae_flagged']} "
                    )

                    # Update UI dashboard metrics, rate limited to once every 5 seconds
                    _now = time.time()
                    if uib and (_now - _ui_last_update >= 5.0):
                        uib.update_metrics(
                            status="running",
                            total_processed=det_total,
                            total_alerts=det_attacks_final,
                            ae_flagged=det_ae_flagged,
                            run_start=start,
                            hybrid_metrics=_calc_hybrid_metrics(det_tp, det_fp, det_fn, det_tn),
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
        # Final metrics push so dashboard shows "stopped" and accurate totals
        if uib:
            try:
                uib.update_metrics(
                    status="stopped",
                    total_processed=det_total,
                    total_alerts=det_attacks_final,
                    ae_flagged=det_ae_flagged,
                    run_start=start or time.time(),
                    hybrid_metrics=_calc_hybrid_metrics(det_tp, det_fp, det_fn, det_tn),
                )
            except Exception:
                pass
        _send_results()


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def _get_peak_memory_mb():
    """Get peak memory usage of the current process in MB."""
    try:
        import resource  # for linux
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

    print("=" * 70)
    print("  HYBRID NETWORK INTRUSION DETECTOR — LIVE (Linux)")
    print("=" * 70)
    print(f"  Interface:      {INTERFACE}")
    if NAMESPACE:
        print(f"  Namespace:      {NAMESPACE}")
    print(f"  Feature dim:    {FEATURE_DIM}")
    print(f"  AE threshold:   {AE_THRESHOLD}")
    print(f"  RF threshold:   {RF_THRESHOLD}")
    print(f"  ML batch size:  {ML_BATCH_SIZE}")
    print(f"  FE batch size:  {FE_BATCH_SIZE}")
    print(f"  Queue size:     ml_q={ML_QUEUE_MAX}")
    print(f"  Alert log:      {ALERT_LOG}")
    print()

    # ── Cross-process communication ──
    ml_q = mp.Queue(maxsize=ML_QUEUE_MAX)
    stop_event = mp.Event()  # Used by reader + FE threads only
    result_q = mp.Queue()

    # ── Start capture pipeline ──
    print("[*] Starting capture pipeline...")
    dumpcap_proc, pcap_stream = build_capture_pipeline()
    print("[*] Pipeline running. Waiting for packets...\n")

    # ── Start 1 thread + 1 process ──
    # ML process loads its own models inside the child
    # ML process is NOT started until warmup completes so the large
    # RF model does not occupy RAM during the warmup phase.
    warmup_done_event = threading.Event()
    fe_t = threading.Thread(
        target=reader_feature_extractor,
        args=(pcap_stream, ml_q, stop_event, warmup_done_event),
        daemon=True, name="reader_fe",
    )
    ml_proc = mp.Process(target=ml_detector_process, args=(ml_q, result_q), name="ml_detector")

    fe_t.start()

    # Wait for warmup to finish before loading ML models into RAM
    if WARMUP_PACKETS > 0:
        print("[*] Waiting for warmup to complete before loading ML models...")
        warmup_done_event.wait()
        if not fe_t.is_alive():
            print("[!] Stream ended before warmup completed — no detection will run.")
        else:
            print("[*] Warmup complete — starting ML process (loading models now)...")
    ml_proc.start()

    # Monitor threads and processes, and handle clean shutdown on Ctrl+C or if dumpcap exits unexpectedly
    try:
        while fe_t.is_alive() and ml_proc.is_alive():
            time.sleep(1)
            if dumpcap_proc and dumpcap_proc.poll() is not None:
                print("[!] dumpcap exited unexpectedly")
                break

    except KeyboardInterrupt:
        print("\n\n[*] Shutting down detector...")

    finally:
        stop_event.set()
        if dumpcap_proc:
            dumpcap_proc.terminate()
        # pcap_stream closes automatically when dumpcap_proc is terminated

        # Let FE thread finish and push sentinel
        fe_t.join(timeout=5)
        # Put sentinel in case FE exited without sending one
        try:
            ml_q.put_nowait(None)
        except Exception:
            pass

        # Read result_q BEFORE join
        ml_results = None
        try:
            ml_results = result_q.get(timeout=30)
        except Exception:
            pass

        ml_proc.join(timeout=10)

    if ml_results is None:
        print("[WARNING] Could not retrieve ML results")

    wall_elapsed = time.time() - wall_start
    cpu_elapsed = time.process_time() - cpu_start

    ml_stats = ml_results["stats"] if ml_results else {}
    det_counters = ml_results["detector_counters"] if ml_results else {}
    ml_latency = ml_results.get("latency", {}) if ml_results else {}

    # Read attack entries from the JSONL file written by the ML process
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

    # Summary
    print()
    print("=" * 70)
    print("  DETECTION SUMMARY")
    print("=" * 70)

    ae_flagged = det_counters.get("ae_flagged", 0)
    rf_called = det_counters.get("rf_called", 0)
    total_det = det_counters.get("total", 0)
    attacks_final = det_counters.get("attacks_final", 0)
    ae_frac = ae_flagged / total_det if total_det > 0 else 0

    fe_count = fe_stats["packet_count"]
    ml_count = ml_stats.get("ml_processed", 0)
    read_count = fe_stats["read_count"]
    overall_pps = fe_count / wall_elapsed if wall_elapsed > 0 else 0

    print(f"  Total packets (read):  {read_count:,}")
    print(f"  Total packets (FE):    {fe_count:,}")
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
        parse_avg = fe_stats["parse_total_time"] / fe_count * 1000
        qput_avg = fe_stats["qput_total_time"] / fe_count * 1000
        loop_total = parse_avg + fe_avg + qput_avg
        print(f"  FE loop breakdown (avg ms/pkt):")
        print(f"    parse+extract:       {parse_avg:.3f}ms")
        print(f"    AfterImage:          {fe_avg:.3f}ms  (max={fe_max:.1f}ms)")
        print(f"    batch q.put:         {qput_avg:.3f}ms  (amortised over {FE_BATCH_SIZE}-pkt batches)")
        print(f"    loop total:          {loop_total:.3f}ms  ({1000/loop_total:.0f} pps capacity)" if loop_total > 0 else "")

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
        print(f"  E2E (read→decision):   avg={e2e.get('avg_ms',0):.2f}ms  p50={e2e.get('p50_ms',0):.2f}ms  p95={e2e.get('p95_ms',0):.2f}ms  p99={e2e.get('p99_ms',0):.2f}ms  max={e2e.get('max_ms',0):.2f}ms")
    print()

    print(f"  --- Resource Usage ---")
    print(f"  Main process CPU:      {cpu_elapsed:.1f}s")
    ml_cpu = ml_latency.get("ml_process_cpu_s", 0)
    if ml_cpu:
        print(f"  ML process CPU:        {ml_cpu:.1f}s")
    peak_mem = _get_peak_memory_mb()
    if peak_mem is not None:
        print(f"  Peak memory (main):    {peak_mem:.1f} MB")

    # Write JSON log
    log_dir = os.path.dirname(ALERT_LOG)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    fe_latency = {}
    if fe_count > 0:
        fe_latency = {
            "afterimage_avg_ms": round(fe_stats["fe_total_time"] / fe_count * 1000, 4),
            "afterimage_max_ms": round(fe_stats["fe_max_time"] * 1000, 4),
            "parse_avg_ms": round(fe_stats["parse_total_time"] / fe_count * 1000, 4),
            "qput_avg_ms": round(fe_stats["qput_total_time"] / fe_count * 1000, 4),
        }

    log_output = {
        "summary": {
            "total_packets_read": read_count,
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

