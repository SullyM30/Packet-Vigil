#!/usr/bin/env python3
# evaluate_detection.py — Compare hybrid detector predictions (attack_log.json)
# against ground-truth labels to compute classification metrics.
#
# Two alignment modes depending on how the replay was performed:
#
#   Warm-up mode (warmup_packets > 0):
#     The full PCAP was replayed.  The first `warmup_packets` went through
#     AfterImage only; ML scored the rest.  packet_num in the log is the
#     global FE index (1-based), so label index = packet_num - 1.
#
#   Legacy mode (warmup_packets == 0 or absent):
#     A pre-trimmed test PCAP was replayed.  The last TEST_SEGMENT_SIZE
#     packets from the labels CSV are used for alignment.
#
# Usage (from project root):
#   python -m NetworkIngestion.evaluate_detection

import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# ─── Configuration ───────────────────────────────────────────────────
ATTACK_LOG = "NetworkIngestion/attack_log.json"
LABELS_CSV = "Train and Test Files/Labels/OS Scan_labels.csv"

# Override the RF threshold at eval time (None → use the value stored in attack_log).
# The replay detector logs ALL AE-flagged packets, so this can be swept freely.
EVAL_RF_THRESHOLD = None

# Cap evaluation to the first N test-segment packets (None → use full test set).
# Helpful for measuring performance before AfterImage state drift kicks in.
EVAL_TEST_SIZE = None

# Direct label-row selection — overrides auto-detection when set.
# Use when the replay PCAP starts from packet 1 (a head-snipped PCAP that
# includes warmup packets) rather than being a tail of the full dataset.
#
# LABELS_SKIP: 0-based row where the TEST segment begins in the CSV.
# LABELS_TAKE: how many rows to read (None → all remaining rows).
LABELS_SKIP = 1367647
LABELS_TAKE = None


def main():
    # ── Load attack log ──────────────────────────────────────────────
    with open(ATTACK_LOG) as f:
        data = json.load(f)

    summary = data["summary"]
    # total_packets_fe = every packet the feature extractor saw (used for label alignment)
    # total_packets_ml may be fewer because ML drops partial last batches
    total_fe = summary.get("total_packets_fe", summary["total_packets_ml"])
    total_fe_actual = summary["total_packets_ml"]
    warmup = summary.get("warmup_packets", 0)
    attacks = data["attacks"]

    ae_threshold = summary.get("ae_threshold", "N/A")
    rf_threshold = summary.get("rf_threshold", "N/A")
    eval_rf_thresh = EVAL_RF_THRESHOLD if EVAL_RF_THRESHOLD is not None else rf_threshold
    ae_flagged = summary.get("ae_flagged", 0)
    ae_flagged_frac = summary.get("ae_flagged_fraction", 0.0)
    rf_called = summary.get("rf_called", 0)
    attacks_detected_summary = summary.get("attacks_detected", len(attacks))

    # ── Print header ─────────────────────────────────────────────────
    print("=" * 60)
    print("  HYBRID DETECTOR EVALUATION")
    print("=" * 60)
    print(f"  Attack log:        {ATTACK_LOG}")
    print(f"  Labels CSV:        {LABELS_CSV}")
    print(f"  AE threshold:      {ae_threshold}")
    print(f"  Eval RF threshold: {eval_rf_thresh}  (applied)")
    print(f"  Eval test size:    {EVAL_TEST_SIZE:,}" if EVAL_TEST_SIZE else "  Eval test size:    (full)")
    print(f"  Total packets FE:  {total_fe:,}  (used for label alignment / cutoff)")
    print(f"  Total packets ML:  {total_fe_actual:,}  (gap={total_fe - total_fe_actual} pkts not scored by ML)")
    print(f"  AE flagged:        {ae_flagged:,}  ({ae_flagged_frac:.1%} of FE packets)")
    print(f"  RF called:         {rf_called:,}")

    # ── Load ground-truth labels ─────────────────────────────────────
    labels_df = pd.read_csv(LABELS_CSV)
    labels_all = labels_df["x"].values

    # ── Align predictions to labels based on mode ────────────────────
    if warmup > 0:
        # Warm-up mode: ML only scored packets after the warmup window
        test_size = total_fe - warmup
        if LABELS_TAKE is not None:
            test_size = min(test_size, LABELS_TAKE)
        if EVAL_TEST_SIZE is not None:
            test_size = min(test_size, EVAL_TEST_SIZE)

        if LABELS_SKIP is not None:
            # Direct override — PCAP starts from packet 1 (head-snipped)
            labels_test = labels_all[LABELS_SKIP : LABELS_SKIP + test_size]
        else:
            # Auto-detect: PCAP corresponds to the tail of the labels CSV
            labels_offset = max(0, len(labels_all) - total_fe)
            labels_test = labels_all[labels_offset + warmup : labels_offset + warmup + test_size]
            print(f"  Labels offset:     {labels_offset:,}  (auto-detected)")

        # packet_num is global FE index (1-based, includes warmup count)
        # Convert to 0-based test index: idx = packet_num - warmup - 1
        attack_indices = set()
        attack_scores = {}
        for a in attacks:
            if a["rf_prob"] < eval_rf_thresh:
                continue
            idx = a["packet_num"] - warmup - 1
            if 0 <= idx < test_size:
                attack_indices.add(idx)
                attack_scores[idx] = a["rf_prob"]
    else:
        # Legacy mode: use the last TEST_SEGMENT_SIZE packets
        test_size = TEST_SEGMENT_SIZE
        if LABELS_TAKE is not None:
            test_size = min(test_size, LABELS_TAKE)
        if EVAL_TEST_SIZE is not None:
            test_size = min(test_size, EVAL_TEST_SIZE)
        if LABELS_SKIP is not None:
            labels_test = labels_all[LABELS_SKIP : LABELS_SKIP + test_size]
            print(f"  Labels rows:       {LABELS_SKIP:,}-{LABELS_SKIP + test_size:,} (LABELS_SKIP/TAKE override)")
        else:
            labels_offset = max(0, len(labels_all) - total_fe)
            labels_test = labels_all[labels_offset : labels_offset + total_fe][-test_size:]
            print(f"  Labels offset:     {labels_offset:,}  (auto-detected)")
        print(f"  Test segment:      last {test_size:,} packets (legacy mode)")

        cutoff = total_fe - test_size + 1
        attack_indices = set()
        attack_scores = {}
        for a in attacks:
            if a["rf_prob"] < eval_rf_thresh:
                continue
            if a["packet_num"] >= cutoff:
                idx = a["packet_num"] - cutoff
                attack_indices.add(idx)
                attack_scores[idx] = a["rf_prob"]

    print()
    print(f"  Attacks detected:  {len(attack_indices):,}")
    print(f"  Labels CSV rows:   {len(labels_all):,}")
    print(f"  Ground truth 1s:   {int(labels_test.sum()):,}")
    print(f"  Ground truth 0s:   {int((labels_test == 0).sum()):,}")
    print()

    # ── Build prediction arrays ──────────────────────────────────────
    y_true = labels_test.astype(int)
    y_pred = np.zeros(test_size, dtype=int)
    y_scores = np.zeros(test_size, dtype=float)

    for idx in attack_indices:
        if 0 <= idx < test_size:
            y_pred[idx] = 1
            y_scores[idx] = attack_scores[idx]

    # ── Classification metrics ───────────────────────────────────────
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_scores)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("  --- Classification Metrics ---")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Precision:   {prec:.4f}")
    print(f"  Recall:      {rec:.4f}")
    print(f"  F1 Score:    {f1:.4f}")
    print(f"  ROC-AUC:     {auc:.4f}")
    print()
    print("  --- Confusion Matrix ---")
    print(f"                 Predicted 0   Predicted 1")
    print(f"  Actual 0 (TN)  {tn:>10,}     {fp:>10,}  (FP)")
    print(f"  Actual 1 (FN)  {fn:>10,}     {tp:>10,}  (TP)")
    print()
    print(f"  True Positives:   {tp:,}")
    print(f"  False Positives:  {fp:,}")
    print(f"  True Negatives:   {tn:,}")
    print(f"  False Negatives:  {fn:,}")
    print()
    print("  --- Full Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=["Benign", "Attack"]))

    # ── Per-attack breakdown (source IPs, ports, score distributions) ──
    if attacks and "src_ip" in attacks[0]:
        from collections import Counter

        # Filter to attacks within the evaluated test segment
        if warmup > 0:
            test_attacks = [a for a in attacks
                            if warmup < a["packet_num"] <= warmup + test_size
                            and a["rf_prob"] >= eval_rf_thresh]
        else:
            cutoff = total_fe - test_size + 1
            test_attacks = [a for a in attacks
                            if a["packet_num"] >= cutoff
                            and a["rf_prob"] >= eval_rf_thresh]

        if test_attacks:
            print("  --- Top Attacking Source IPs ---")
            src_counts = Counter(a["src_ip"] for a in test_attacks)
            for ip, count in src_counts.most_common(10):
                print(f"    {ip:<20} {count:>6,} detections")
            print()

            print("  --- Top Targeted Destination IPs ---")
            dst_counts = Counter(a["dst_ip"] for a in test_attacks)
            for ip, count in dst_counts.most_common(5):
                print(f"    {ip:<20} {count:>6,} detections")
            print()

            print("  --- Top Targeted Ports (dst) ---")
            port_counts = Counter(a["dst_port"] for a in test_attacks)
            for port, count in port_counts.most_common(10):
                print(f"    port {port:<8} {count:>6,} detections")
            print()

            # AE error and RF probability distribution stats
            ae_errors = [a["ae_error"] for a in test_attacks if "ae_error" in a]
            rf_probs = [a["rf_prob"] for a in test_attacks if "rf_prob" in a]
            if ae_errors:
                ae_arr = np.array(ae_errors)
                rf_arr = np.array(rf_probs)
                print("  --- Detection Score Stats (test segment) ---")
                print(f"  AE reconstruction error:  avg={ae_arr.mean():.4f}  "
                      f"p50={np.percentile(ae_arr, 50):.4f}  "
                      f"p95={np.percentile(ae_arr, 95):.4f}  "
                      f"max={ae_arr.max():.4f}")
                print(f"  RF probability:           avg={rf_arr.mean():.4f}  "
                      f"p50={np.percentile(rf_arr, 50):.4f}  "
                      f"p95={np.percentile(rf_arr, 95):.4f}  "
                      f"max={rf_arr.max():.4f}")
                print()

    # ── Performance metrics (if present in attack_log) ───────────────
    perf = data.get("performance", {})
    if perf:
        print("  --- Performance ---")
        wall = perf.get("wall_clock_s", 0)
        if wall:
            print(f"  Wall clock:            {wall:.1f}s")
        overall_pps = perf.get("overall_throughput_pps", 0)
        if overall_pps:
            print(f"  Overall throughput:    {overall_pps:.0f} pps")
        ml_pps = perf.get("ml_throughput_pps", 0)
        if ml_pps:
            print(f"  ML throughput:         {ml_pps:.0f} pps")
        print()

        # Feature-extractor loop latency breakdown
        fe_lat = perf.get("fe_latency", {})
        if fe_lat:
            parse_ms = fe_lat.get('parse_avg_ms', 0)
            ai_ms = fe_lat.get('afterimage_avg_ms', fe_lat.get('avg_ms', 0))
            ai_max = fe_lat.get('afterimage_max_ms', fe_lat.get('max_ms', 0))
            qput_ms = fe_lat.get('qput_avg_ms', 0)
            loop_total = parse_ms + ai_ms + qput_ms
            if parse_ms or qput_ms:
                print(f"  FE loop breakdown (avg ms/pkt):")
                print(f"    parse+extract:       {parse_ms:.3f}ms")
                print(f"    AfterImage:          {ai_ms:.3f}ms  (max={ai_max:.1f}ms)")
                print(f"    batch q.put:         {qput_ms:.3f}ms")
                if loop_total > 0:
                    print(f"    loop total:          {loop_total:.3f}ms  ({1000 / loop_total:.0f} pps capacity)")
            else:
                print(f"  FE latency:            avg={ai_ms:.2f}ms  max={ai_max:.1f}ms")

        # ML pipeline latency (per packet)
        ml_lat = perf.get("ml_latency_per_packet", {})
        if ml_lat:
            print(
                f"  ML latency (per pkt):  avg={ml_lat.get('avg_ms', 0):.3f}ms  "
                f"p50={ml_lat.get('p50_ms', 0):.3f}ms  "
                f"p95={ml_lat.get('p95_ms', 0):.3f}ms  "
                f"p99={ml_lat.get('p99_ms', 0):.3f}ms  "
                f"max={ml_lat.get('max_ms', 0):.3f}ms"
            )

        # Autoencoder latency (per packet)
        ae_lat = perf.get("ae_latency_per_packet", {})
        if ae_lat:
            print(
                f"  AE latency (per pkt):  avg={ae_lat.get('avg_ms', 0):.3f}ms  "
                f"p50={ae_lat.get('p50_ms', 0):.3f}ms  "
                f"p95={ae_lat.get('p95_ms', 0):.3f}ms  "
                f"max={ae_lat.get('max_ms', 0):.3f}ms"
            )

        # Random Forest latency (per call)
        rf_lat = perf.get("rf_latency_per_call", {})
        if rf_lat and rf_lat.get("total_calls", 0) > 0:
            print(
                f"  RF latency (per call): avg={rf_lat.get('avg_ms', 0):.2f}ms  "
                f"max={rf_lat.get('max_ms', 0):.2f}ms  "
                f"({rf_lat.get('total_calls', 0):,} calls)"
            )

        # End-to-end latency (packet read to decision)
        e2e_lat = perf.get("e2e_latency_per_packet", {})
        if e2e_lat:
            print(
                f"  E2E (read->decision):   avg={e2e_lat.get('avg_ms', 0):.2f}ms  "
                f"p50={e2e_lat.get('p50_ms', 0):.2f}ms  "
                f"p95={e2e_lat.get('p95_ms', 0):.2f}ms  "
                f"p99={e2e_lat.get('p99_ms', 0):.2f}ms  "
                f"max={e2e_lat.get('max_ms', 0):.2f}ms"
            )
        print()

        # Resource usage (CPU, memory)
        res = perf.get("resource_usage", {})
        if res:
            print(f"  --- Resource Usage ---")
            main_cpu = res.get("main_process_cpu_s", 0)
            if main_cpu:
                print(f"  Main process CPU:      {main_cpu:.1f}s")
            ml_cpu = res.get("ml_process_cpu_s", 0)
            if ml_cpu:
                print(f"  ML process CPU:        {ml_cpu:.1f}s")
            ml_wall = res.get("ml_process_wall_s", 0)
            if ml_wall:
                print(f"  ML process wall:       {ml_wall:.1f}s")
            peak_mem = res.get("peak_memory_mb")
            if peak_mem is not None:
                print(f"  Peak memory (main):    {peak_mem:.1f} MB")
            print()

    print("=" * 60)


if __name__ == "__main__":
    main()
