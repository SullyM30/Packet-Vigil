#!/usr/bin/env python3
"""
Capture-only script for Linux: saves raw packets to a PCAP file.

This script ONLY captures packets
dumpcap writes packets to a .pcap file at wire speed with zero processing overhead.

After capture, process the PCAP offline with:
  python3 -m NetworkIngestion.replay_captured_pcap_linux <captured.pcap>

"""

import subprocess
import sys
import os
import time
import signal

# --- CONFIGURATION ---
DUMPCAP = "/usr/bin/dumpcap"
TSHARK = "/usr/bin/tshark"

# Network interface to capture on
INTERFACE = "veth_detect"

# Network namespace (set to None if not using namespaces)
NAMESPACE = "detect_ns"

# Output PCAP file path
OUTPUT_PCAP = "NetworkIngestion/captured_replay.pcap"

# Capture buffer size in MB (larger buffer reduces packet drops at high rates)
BUFFER_MB = 256
# ---------------------


def check_command(cmd):
    """Return True if the given command exists on the system."""
    try:
        result = subprocess.run(["which", cmd], capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False


def main():
    print("=" * 60)
    print("CAPTURE-ONLY MODE (Linux)")
    print("=" * 60)
    print(f"Interface:   {INTERFACE}")
    if NAMESPACE:
        print(f"Namespace:   {NAMESPACE}")
    print(f"Output:      {OUTPUT_PCAP}")
    print(f"Buffer:      {BUFFER_MB} MB")
    print()

    # Prefer dumpcap for capture; fall back to tshark if not available
    use_dumpcap = check_command(DUMPCAP)
    use_tshark = check_command(TSHARK)

    if not use_dumpcap and not use_tshark:
        print("[ERROR] Neither dumpcap nor tshark found!")
        print("Install with: sudo apt install wireshark tshark")
        sys.exit(1)

    if use_dumpcap:
        print(f"  Using dumpcap for capture (fastest)")
    else:
        print(f"  dumpcap not found, using tshark for capture")

    # Create output directory if needed
    out_dir = os.path.dirname(OUTPUT_PCAP)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Pre-create the output file with world-writable permissions before launching dumpcap.
    # dumpcap drops root privileges after opening the capture interface, so the file must
    # already exist and be writable by the unprivileged user UID before that happens.
    with open(OUTPUT_PCAP, "wb"):
        pass
    os.chmod(OUTPUT_PCAP, 0o666)

    # Build the capture command
    if use_dumpcap:
        capture_cmd = [
            DUMPCAP,
            "-i", INTERFACE,
            "-B", str(BUFFER_MB),
            "-P",               # pcap format (not pcapng)
            "-w", OUTPUT_PCAP,
        ]
    else:
        capture_cmd = [
            TSHARK,
            "-i", INTERFACE,
            "-B", str(BUFFER_MB),
            "-w", OUTPUT_PCAP,
        ]

    # Wrap with namespace if configured
    if NAMESPACE:
        capture_cmd = ["ip", "netns", "exec", NAMESPACE] + capture_cmd

    print(f"\n[*] Starting capture...")
    print(f"    Command: {' '.join(capture_cmd)}")
    print(f"\n[*] Capture running. Press Ctrl+C to stop.\n")

    # Use DEVNULL for stdout/stderr — dumpcap writes the PCAP directly to the output
    # file, and piping its progress output can block the process when the pipe buffer fills.
    proc = subprocess.Popen(
        capture_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Check that the process didn't exit immediately (bad interface, permissions, etc.)
    time.sleep(1)
    if proc.poll() is not None:
        print(f"\n[ERROR] Capture process exited immediately (return code {proc.returncode})!")
        print("Common causes:")
        print("  - Namespace/interface doesn't exist (run setup_ns.sh first?)")
        print("  - Not running as root (use sudo)")
        print(f"  - Interface '{INTERFACE}' not found in namespace '{NAMESPACE}'")
        print(f"\nTest manually: {' '.join(capture_cmd)}")
        sys.exit(1)

    # Poll until stopped by Ctrl+C, printing file size for progress
    start_time = time.time()
    try:
        while proc.poll() is None:
            time.sleep(2)
            elapsed = time.time() - start_time

            if os.path.exists(OUTPUT_PCAP):
                size_mb = os.path.getsize(OUTPUT_PCAP) / 1e6
                print(f"  [{elapsed:.0f}s] Capturing... {size_mb:.1f} MB written", end="\r")

    except KeyboardInterrupt:
        print(f"\n\n[*] Stopping capture...")
        # SIGINT triggers a clean shutdown in dumpcap
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(timeout=5)

    # Print capture summary
    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("CAPTURE COMPLETE")
    print("=" * 60)

    if os.path.exists(OUTPUT_PCAP):
        size_mb = os.path.getsize(OUTPUT_PCAP) / 1e6
        print(f"Output file:     {OUTPUT_PCAP}")
        print(f"File size:       {size_mb:.2f} MB")
        print(f"Capture time:    {elapsed:.1f}s")

        # Try to get packet count from capinfos if available
        try:
            count_cmd = ["capinfos", "-c", OUTPUT_PCAP]
            result = subprocess.run(count_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if "Number of packets" in line:
                        print(f"Packets:         {line.split(':')[-1].strip()}")
        except Exception:
            pass

        print(f"\nNext step — process offline:")
        print(f"  sudo python3 -m NetworkIngestion.replay_captured_pcap_linux {OUTPUT_PCAP}")
    else:
        print("[ERROR] No output file created!")
        sys.exit(1)

    print("=" * 60)


if __name__ == "__main__":
    main()
