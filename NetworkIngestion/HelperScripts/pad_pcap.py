# pad_pcap.py — Pad truncated PCAP packets to their original wire length.
#
# Many PCAP captures use a snaplen (e.g., 200 bytes) to save disk space,
# storing only the first N bytes while recording the full wire length in
# the packet header.  When tcpreplay sends these truncated packets the
# frame sizes don't match what the deterministic offline replay sees.
#
# This script pads each packet with null bytes up to its recorded wire
# length so that replayed packets have realistic sizes.
#
# Usage:
#   python pad_pcap.py input.pcap output.pcap [--fix-checksums]
#
# Example:
#   python pad_pcap.py "Train and Test Files/Kitsune PCAPs/OSScan.pcap" "padded_OSScan.pcap"

import sys
import os

try:
    from scapy.all import rdpcap, wrpcap, Raw, IP, TCP, UDP, Ether
    from scapy.utils import PcapReader, PcapWriter
except ImportError:
    print("[ERROR] Scapy is required. Install with: pip install scapy")
    sys.exit(1)


def pad_pcap(input_path: str, output_path: str, fix_checksums: bool = False,
             padding_byte: bytes = b'\x00', verbose: bool = True):
    """Pad every truncated packet in a PCAP to its original wire length.

    Chooses between in-memory mode (fast, for files < 100 MB) and
    streaming mode (memory-efficient, for larger captures).

    Returns a dict of padding statistics.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    stats = {
        "total_packets": 0,
        "padded_packets": 0,
        "already_full": 0,
        "total_bytes_added": 0,
        "min_padding": float('inf'),
        "max_padding": 0,
    }

    if verbose:
        print(f"[*] Reading PCAP: {input_path}")

    file_size = os.path.getsize(input_path)

    # Pick strategy based on file size
    if file_size > 100 * 1024 * 1024:  # > 100 MB
        if verbose:
            print(f"[*] Large file ({file_size / 1e6:.1f} MB) - using streaming mode")
        return _pad_pcap_streaming(input_path, output_path, fix_checksums,
                                   padding_byte, verbose, stats)
    else:
        return _pad_pcap_memory(input_path, output_path, fix_checksums,
                                padding_byte, verbose, stats)


# ── In-memory padding (small files) ─────────────────────────────────

def _pad_pcap_memory(input_path: str, output_path: str, fix_checksums: bool,
                     padding_byte: bytes, verbose: bool, stats: dict) -> dict:
    """Load entire PCAP into memory — faster for files under ~100 MB."""
    packets = rdpcap(input_path)
    padded_packets = []

    if verbose:
        print(f"[*] Loaded {len(packets)} packets")
        print(f"[*] Processing packets...")

    for i, pkt in enumerate(packets):
        stats["total_packets"] += 1

        captured_len = len(pkt)
        wire_len = pkt.wirelen if hasattr(pkt, 'wirelen') and pkt.wirelen else captured_len

        if wire_len > captured_len:
            # Packet was truncated by the snaplen — pad it
            padding_needed = wire_len - captured_len
            stats["padded_packets"] += 1
            stats["total_bytes_added"] += padding_needed
            stats["min_padding"] = min(stats["min_padding"], padding_needed)
            stats["max_padding"] = max(stats["max_padding"], padding_needed)

            padded_pkt = pkt / Raw(load=padding_byte * padding_needed)

            if fix_checksums:
                padded_pkt = _fix_checksums(padded_pkt)

            padded_packets.append(padded_pkt)
        else:
            stats["already_full"] += 1
            padded_packets.append(pkt)

        if verbose and (i + 1) % 100000 == 0:
            print(f"      Processed {i + 1:,} / {len(packets):,} packets...")

    if stats["min_padding"] == float('inf'):
        stats["min_padding"] = 0

    if verbose:
        print(f"[*] Writing padded PCAP: {output_path}")

    wrpcap(output_path, padded_packets)
    return stats


# ── Streaming padding (large files) ─────────────────────────────────

def _pad_pcap_streaming(input_path: str, output_path: str, fix_checksums: bool,
                        padding_byte: bytes, verbose: bool, stats: dict) -> dict:
    """Stream packets one at a time — keeps memory usage flat for large captures."""
    with PcapReader(input_path) as reader, PcapWriter(output_path, append=False) as writer:
        for i, pkt in enumerate(reader):
            stats["total_packets"] += 1

            captured_len = len(pkt)
            wire_len = pkt.wirelen if hasattr(pkt, 'wirelen') and pkt.wirelen else captured_len

            if wire_len > captured_len:
                padding_needed = wire_len - captured_len
                stats["padded_packets"] += 1
                stats["total_bytes_added"] += padding_needed
                stats["min_padding"] = min(stats["min_padding"], padding_needed)
                stats["max_padding"] = max(stats["max_padding"], padding_needed)

                padded_pkt = pkt / Raw(load=padding_byte * padding_needed)

                if fix_checksums:
                    padded_pkt = _fix_checksums(padded_pkt)

                writer.write(padded_pkt)
            else:
                stats["already_full"] += 1
                writer.write(pkt)

            if verbose and (i + 1) % 100000 == 0:
                print(f"      Processed {i + 1:,} packets...")

    if stats["min_padding"] == float('inf'):
        stats["min_padding"] = 0

    return stats


# ── Checksum repair ──────────────────────────────────────────────────

def _fix_checksums(pkt):
    """Recalculate IP/TCP/UDP checksums after padding.

    For Kitsune feature extraction checksums are irrelevant (tshark only
    looks at metadata), but some network stacks drop invalid-checksum frames.
    """
    if IP in pkt:
        del pkt[IP].chksum
    if TCP in pkt:
        del pkt[TCP].chksum
    if UDP in pkt:
        del pkt[UDP].chksum

    # Rebuild the packet to trigger scapy's checksum recalculation
    return Ether(bytes(pkt))


# ── CLI entry point ──────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print("\nUsage: python pad_pcap.py <input.pcap> <output.pcap> [--fix-checksums]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    fix_checksums = "--fix-checksums" in sys.argv or "-c" in sys.argv

    print("=" * 60)
    print("PCAP Packet Padder")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Fix checksums: {fix_checksums}")
    print()

    try:
        stats = pad_pcap(input_path, output_path, fix_checksums=fix_checksums)

        print()
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Total packets:     {stats['total_packets']:,}")
        print(f"Padded packets:    {stats['padded_packets']:,} "
              f"({100 * stats['padded_packets'] / stats['total_packets']:.1f}%)")
        print(f"Already full:      {stats['already_full']:,}")
        print(f"Total bytes added: {stats['total_bytes_added']:,} "
              f"({stats['total_bytes_added'] / 1e6:.2f} MB)")
        if stats['padded_packets'] > 0:
            print(f"Padding range:     {stats['min_padding']} - {stats['max_padding']} bytes")
            print(f"Avg padding:       "
                  f"{stats['total_bytes_added'] / stats['padded_packets']:.1f} bytes/packet")
        print()
        print(f"[SUCCESS] Padded PCAP written to: {output_path}")
        print()
        print("Next steps:")
        print(f"  1. Replay with tcpreplay:")
        print(f"     tcpreplay -i <interface> \"{output_path}\"")
        print(f"  2. Run live capture to extract features")
        print(f"  3. Compare with deterministic replay features")

    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to process PCAP: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
