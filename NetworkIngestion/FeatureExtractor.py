# FeatureExtractor.py — Kitsune-compatible packet feature extractor.
#
# Reads packets from a PCAP (via tshark) or a pre-parsed TSV file and
# produces a 100-dimensional feature vector per packet using the
# AfterImage incremental statistics module (netStat).
#
# Parsing priority: tshark (fast, produces a .tsv) > scapy (slower fallback).
# Once a .tsv exists it can be reused on subsequent runs.

import os
import sys
import subprocess
import csv
import platform

import numpy as np

# Optional Cython-accelerated AfterImage (experimental)
use_extrapolation = False
if use_extrapolation:
    print("Importing AfterImage Cython Library")
    if not os.path.isfile("AfterImage.c"):
        cmd = "python setup.py build_ext --inplace"
        subprocess.call(cmd, shell=True)

import NetworkIngestion.netStat as ns

print("Importing Scapy Library")
from scapy.all import *


class FE:
    """Feature Extractor — yields one feature vector per packet.

    Call get_next_vector() in a loop to iterate through the capture.
    Returns an empty list when all packets have been consumed.
    """

    def __init__(self, file_path, limit=np.inf):
        self.path = file_path
        self.limit = limit
        self.parse_type = None
        self.curPacketIndx = 0
        self.tsvin = None       # CSV reader for TSV mode
        self.scapyin = None     # Scapy packet list for scapy mode

        self.__prep__()

        # AfterImage incremental statistics tracker
        maxHost = 100000000000
        maxSess = 100000000000
        self.nstat = ns.netStat(np.nan, maxHost, maxSess)

    # ── tshark path resolution ───────────────────────────────────────
    def _get_tshark_path(self):
        """Return the path to the tshark binary, or '' if not found."""
        if platform.system() == 'Windows':
            return "D:\\Wireshark\\tshark.exe"
        else:
            system_path = os.environ['PATH']
            for path in system_path.split(os.pathsep):
                filename = os.path.join(path, 'tshark')
                if os.path.isfile(filename):
                    return filename
        return ''

    # ── File preparation ─────────────────────────────────────────────
    def __prep__(self):
        """Validate the input file, convert to TSV if needed, and open a reader."""
        if not os.path.isfile(self.path):
            print("File: " + self.path + " does not exist")
            raise FileNotFoundError(f"Input file not found: {self.path}")

        file_ext = self.path.split('.')[-1]
        self._tshark = self._get_tshark_path()

        # Already a TSV (previously parsed)
        if file_ext == "tsv":
            self.parse_type = "tsv"

        # PCAP / PCAPNG — convert to TSV via tshark if available
        elif file_ext in ("pcap", "pcapng"):
            if os.path.isfile(self._tshark):
                self.pcap2tsv_with_tshark()
                self.path += ".tsv"
                self.parse_type = "tsv"
            else:
                print("tshark not found. Trying scapy...")
                self.parse_type = "scapy"
        else:
            print("File: " + self.path + " is not a tsv or pcap file")
            raise ValueError(f"Unsupported file type: .{file_ext}")

        # Open the appropriate reader
        if self.parse_type == "tsv":
            # Increase CSV field-size limit to handle very wide rows
            max_int = sys.maxsize
            while True:
                try:
                    csv.field_size_limit(max_int)
                    break
                except OverflowError:
                    max_int = int(max_int / 10)

            print("counting lines in file...")
            num_lines = sum(1 for line in open(self.path))
            print("There are " + str(num_lines) + " Packets.")
            self.limit = min(self.limit, num_lines - 1)
            self.tsvinf = open(self.path, 'rt', encoding="utf8")
            self.tsvin = csv.reader(self.tsvinf, delimiter='\t')
            self.tsvin.__next__()  # Skip the header row

        else:  # scapy
            print("Reading PCAP file via Scapy...")
            self.scapyin = rdpcap(self.path)
            self.limit = len(self.scapyin)
            print("Loaded " + str(len(self.scapyin)) + " Packets.")

    # ── Per-packet feature extraction ────────────────────────────────
    def get_next_vector(self):
        """Parse the next packet and return its AfterImage feature vector.

        Returns an empty list when no more packets remain.
        """
        if self.curPacketIndx == self.limit:
            if self.parse_type == 'tsv':
                self.tsvinf.close()
            return []

        # ── TSV parsing ──────────────────────────────────────────────
        if self.parse_type == "tsv":
            row = self.tsvin.__next__()
            IPtype = np.nan
            timestamp = row[0]
            framelen = row[1]
            srcIP = ''
            dstIP = ''

            if row[4] != '':       # IPv4
                srcIP = row[4]
                dstIP = row[5]
                IPtype = 0
            elif row[17] != '':    # IPv6
                srcIP = row[17]
                dstIP = row[18]
                IPtype = 1

            # TCP or UDP port — concatenation yields whichever is non-empty
            srcproto = row[6] + row[8]
            dstproto = row[7] + row[9]
            srcMAC = row[2]
            dstMAC = row[3]

            if srcproto == '':     # Layer-2 / Layer-1 protocol
                if row[12] != '':          # ARP
                    srcproto = 'arp'
                    dstproto = 'arp'
                    srcIP = row[14]
                    dstIP = row[16]
                    IPtype = 0
                elif row[10] != '':        # ICMP
                    srcproto = 'icmp'
                    dstproto = 'icmp'
                    IPtype = 0
                elif srcIP + srcproto + dstIP + dstproto == '':
                    # Unknown protocol — fall back to MAC addresses
                    srcIP = row[2]
                    dstIP = row[3]

        # ── Scapy parsing ────────────────────────────────────────────
        elif self.parse_type == "scapy":
            packet = self.scapyin[self.curPacketIndx]
            IPtype = np.nan
            timestamp = packet.time
            framelen = len(packet)

            if packet.haslayer(IP):        # IPv4
                srcIP = packet[IP].src
                dstIP = packet[IP].dst
                IPtype = 0
            elif packet.haslayer(IPv6):    # IPv6
                srcIP = packet[IPv6].src
                dstIP = packet[IPv6].dst
                IPtype = 1
            else:
                srcIP = ''
                dstIP = ''

            if packet.haslayer(TCP):
                srcproto = str(packet[TCP].sport)
                dstproto = str(packet[TCP].dport)
            elif packet.haslayer(UDP):
                srcproto = str(packet[UDP].sport)
                dstproto = str(packet[UDP].dport)
            else:
                srcproto = ''
                dstproto = ''

            srcMAC = packet.src
            dstMAC = packet.dst

            if srcproto == '':     # Layer-2 / Layer-1 protocol
                if packet.haslayer(ARP):
                    srcproto = 'arp'
                    dstproto = 'arp'
                    srcIP = packet[ARP].psrc
                    dstIP = packet[ARP].pdst
                    IPtype = 0
                elif packet.haslayer(ICMP):
                    srcproto = 'icmp'
                    dstproto = 'icmp'
                    IPtype = 0
                elif srcIP + srcproto + dstIP + dstproto == '':
                    srcIP = packet.src
                    dstIP = packet.dst
        else:
            return []

        self.curPacketIndx += 1

        # Feed parsed fields into AfterImage and return the feature vector
        try:
            return self.nstat.updateGetStats(
                IPtype, srcMAC, dstMAC, srcIP, srcproto, dstIP, dstproto,
                int(framelen), float(timestamp)
            )
        except Exception as e:
            print(e)
            return []

    # ── tshark conversion ────────────────────────────────────────────
    def pcap2tsv_with_tshark(self):
        """Convert a PCAP to a tshark-parsed TSV with the fields needed for
        AfterImage feature extraction."""
        print('Parsing with tshark...')
        fields = (
            "-e frame.time_epoch -e frame.len -e eth.src -e eth.dst "
            "-e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport "
            "-e udp.srcport -e udp.dstport -e icmp.type -e icmp.code "
            "-e arp.opcode -e arp.src.hw_mac -e arp.src.proto_ipv4 "
            "-e arp.dst.hw_mac -e arp.dst.proto_ipv4 -e ipv6.src -e ipv6.dst"
        )
        cmd = (
            f'"{self._tshark}" -r {self.path} -T fields {fields} '
            f'-E header=y -E occurrence=f > {self.path}.tsv'
        )
        subprocess.call(cmd, shell=True)
        print("tshark parsing complete. File saved as: " + self.path + ".tsv")

    def get_num_features(self):
        """Return the number of features produced per packet."""
        return len(self.nstat.getNetStatHeaders())
