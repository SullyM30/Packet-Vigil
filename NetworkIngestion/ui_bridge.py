"""ui_bridge.py — Writes detector output to UI/data/ so the Dash dashboard refreshes live.

Used by replay_detector.py and live_detector_linux.py to push alerts and runtime
metrics without any changes to the Flask/Dash server.

Alert schema (one JSON object per line in UI/data/alerts.json):
    timestamp, flow_id, src_ip, dst_ip, src_port, dst_port,
    ae_score, ae_threshold, ae_verdict, rf_prob, rf_threshold,
    rf_verdict, final_verdict, attack_type, true_label
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

# Resolve output paths relative to this file's location in the project tree
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
UI_DATA_DIR   = _PROJECT_ROOT / "UI" / "data"
ALERTS_FILE   = UI_DATA_DIR / "alerts.json"
METRICS_FILE  = UI_DATA_DIR / "metrics.json"


def _ensure_dir():
    UI_DATA_DIR.mkdir(parents=True, exist_ok=True)


def clear_alerts():
    """Truncate alerts.json at the start of a new detection run."""
    _ensure_dir()
    ALERTS_FILE.write_text("", encoding="utf-8")


def _to_iso(ts) -> str:
    """Convert a float epoch timestamp to ISO 8601 UTC string.

    Example: 1740830096.123 -> "2026-03-01T12:34:56.123Z"
    """
    try:
        epoch = float(ts)
        return (
            datetime.fromtimestamp(epoch, tz=timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
            + "Z"
        )
    except Exception:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def format_alert(raw: dict, ae_threshold: float, rf_threshold: float) -> dict:
    """Map a raw detector result dict to the UI alert schema.

    Expected keys in raw: packet_num, timestamp, src_ip, dst_ip,
    src_port, dst_port, ae_error, rf_prob
    """
    rf_prob = raw.get("rf_prob", 0.0)
    verdict = "attack" if rf_prob >= rf_threshold else "benign"
    return {
        "timestamp":     _to_iso(raw.get("timestamp", "")),
        "flow_id":       f"f_{raw.get('packet_num', 0)}",
        "src_ip":        raw.get("src_ip", ""),
        "dst_ip":        raw.get("dst_ip", ""),
        "src_port":      raw.get("src_port", ""),
        "dst_port":      raw.get("dst_port", ""),
        "ae_score":      raw.get("ae_error", 0.0),
        "ae_threshold":  ae_threshold,
        "ae_verdict":    "anomaly",
        "rf_prob":       rf_prob,
        "rf_threshold":  rf_threshold,
        "rf_verdict":    verdict,
        "final_verdict": verdict,
        "attack_type":   None,
        "true_label":    None,
    }


def append_alert(raw: dict, ae_threshold: float, rf_threshold: float):
    """Format and append one RF-confirmed attack alert to UI/data/alerts.json.

    Only call this for packets where rf_pred == 1 (confirmed attack).
    For high-throughput runs, use open_alerts_writer() + write_alert() instead
    to avoid the overhead of opening and closing the file for every alert.
    """
    _ensure_dir()
    alert = format_alert(raw, ae_threshold, rf_threshold)
    with open(ALERTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(alert) + "\n")


def open_alerts_writer():
    """Open UI/data/alerts.json for appending and return the file handle.

    Use this instead of append_alert() during high-throughput runs.
    Pass the handle to write_alert() for each attack, then close it when done.
    """
    _ensure_dir()
    return open(ALERTS_FILE, "a", encoding="utf-8", buffering=64 * 1024)


def write_alert(f, raw: dict, ae_threshold: float, rf_threshold: float):
    """Write one formatted alert to an already-open alerts file handle."""
    alert = format_alert(raw, ae_threshold, rf_threshold)
    f.write(json.dumps(alert) + "\n")


def update_metrics(
    status: str,
    total_processed: int,
    total_alerts: int,
    ae_flagged: int,
    run_start: float,
):
    """Atomically overwrite UI/data/metrics.json with current run statistics.

    Args:
        status:          "starting" | "running" | "stopped"
        total_processed: total packets processed by the ML pipeline
        total_alerts:    number of RF-confirmed attacks so far
        ae_flagged:      packets that exceeded the AE threshold
        run_start:       time.time() value when the run started
    """
    _ensure_dir()
    now = time.time()
    elapsed = now - run_start if run_start else 0.0
    alert_rate = total_alerts / elapsed if elapsed > 0 else 0.0

    # Try to collect CPU and memory stats; skip gracefully if psutil is not installed
    resource = {"cpu_percent": 0.0, "memory_mb": 0.0, "memory_percent": 0.0}
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        resource = {
            "cpu_percent":    psutil.cpu_percent(interval=None),
            "memory_mb":      round(proc.memory_info().rss / 1e6, 1),
            "memory_percent": round(proc.memory_percent(), 1),
        }
    except Exception:
        pass

    metrics = {
        "last_updated":          datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "status":                status,
        "evaluation_mode":       "real_time",
        "total_flows_processed": total_processed,
        "total_alerts":          total_alerts,
        "ae_flagged":            ae_flagged,
        "alert_rate_per_second": round(alert_rate, 3),
        "hybrid_metrics":        None,   # no ground-truth labels during live detection
        "resource_usage":        resource,
    }

    # Write to a temp file then rename for an atomic update (prevents partial reads by Dash)
    tmp = METRICS_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    tmp.replace(METRICS_FILE)
