import numpy as np
import os
import subprocess

# Try Cython-optimized AfterImage first, fall back to pure Python
try:
    import NetworkIngestion.AfterImage_cy as af
    _USING_CYTHON = True
except ImportError:
    import NetworkIngestion.AfterImage as af
    _USING_CYTHON = False

#
# MIT License
#
# Copyright (c) 2018 Yisroel mirsky
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


class netStat:
    #Datastructure for efficent network stat queries
    # HostLimit: no more that this many Host identifiers will be tracked
    # HostSimplexLimit: no more that this many outgoing channels from each host will be tracked (purged periodically)
    # Lambdas: a list of 'window sizes' (decay factors) to track for each stream. nan resolved to default [5,3,1,.1,.01]
    def __init__(self, Lambdas = np.nan, HostLimit=255,HostSimplexLimit=1000):
        #Lambdas
        if np.isnan(Lambdas):
            self.Lambdas = [5,3,1,.1,.01]
        else:
            self.Lambdas = Lambdas

        #HT Limits
        self.HostLimit = HostLimit
        self.SessionLimit = HostSimplexLimit*self.HostLimit*self.HostLimit #*2 since each dual creates 2 entries in memory
        self.MAC_HostLimit = self.HostLimit*10

        # Cap limits to prevent memory explosion and maintain performance
        # Even with pruning, massive hash tables cause slowdown
        MAX_JIT_LIMIT = 200000      # Cap jitter table at 200k entries
        MAX_SESSION_LIMIT = 200000  # Cap session table at 200k entries
        MAX_MI_LIMIT = 200000       # Cap MAC-IP table at 200k entries
        MAX_HOST_LIMIT = 50000      # Cap host table at 50k entries

        #HTs
        self.HT_jit = af.incStatDB(limit=min(self.HostLimit*self.HostLimit, MAX_JIT_LIMIT))#H-H Jitter Stats
        self.HT_jit_last_ts = {}  # Track last timestamp per flow for jitter computation
        self.HT_MI = af.incStatDB(limit=min(self.MAC_HostLimit, MAX_MI_LIMIT))#MAC-IP relationships
        self.HT_H = af.incStatDB(limit=min(self.HostLimit, MAX_HOST_LIMIT)) #Source Host BW Stats
        self.HT_Hp = af.incStatDB(limit=min(self.SessionLimit, MAX_SESSION_LIMIT))#Source Host BW Stats


    def findDirection(self,IPtype,srcIP,dstIP,eth_src,eth_dst): #cpp: this is all given to you in the direction string of the instance (NO NEED FOR THIS FUNCTION)
        if IPtype==0: #is IPv4
            lstP = srcIP.rfind('.')
            src_subnet = srcIP[0:lstP:]
            lstP = dstIP.rfind('.')
            dst_subnet = dstIP[0:lstP:]
        elif IPtype==1: #is IPv6
            src_subnet = srcIP[0:round(len(srcIP)/2):]
            dst_subnet = dstIP[0:round(len(dstIP)/2):]
        else: #no Network layer, use MACs
            src_subnet = eth_src
            dst_subnet = eth_dst

        return src_subnet, dst_subnet

    def updateGetStats(self, IPtype, srcMAC,dstMAC, srcIP, srcProtocol, dstIP, dstProtocol, datagramSize, timestamp):
        #MAC.IP: Stats on src MAC-IP relationships
        MIstat =  np.zeros((3*len(self.Lambdas,)))
        for i in range(len(self.Lambdas)):
            MIstat[(i*3):((i+1)*3)] = self.HT_MI.update_get_1D_Stats(srcMAC+srcIP, timestamp, datagramSize, self.Lambdas[i])

        # Host BW: In original Kitsune dataset, Hstat duplicates MIstat (same srcMAC+srcIP key)
        # NOTE: Original dataset has identical values for MIstat and Hstat
        Hstat = MIstat.copy()

        # Host-Host BW: Stats on the dual traffic behavior between srcIP and dstIP
        HHstat =  np.zeros((7*len(self.Lambdas,)))
        for i in range(len(self.Lambdas)):
            HHstat[(i*7):((i+1)*7)] = self.HT_H.update_get_1D2D_Stats(srcIP, dstIP,timestamp,datagramSize,self.Lambdas[i])

        # Host-Host Jitter: track inter-arrival time (jitter) between packets in each flow
        jit_key = srcIP + dstIP
        last_ts = self.HT_jit_last_ts.get(jit_key, None)
        jitter = timestamp - last_ts if last_ts is not None else 0.0
        self.HT_jit_last_ts[jit_key] = timestamp

        HHstat_jit =  np.zeros((3*len(self.Lambdas,)))
        for i in range(len(self.Lambdas)):
            HHstat_jit[(i*3):((i+1)*3)] = self.HT_jit.update_get_1D_Stats(jit_key, timestamp, jitter, self.Lambdas[i])

        # Host-Host BW: Stats on the dual traffic behavior between srcIP and dstIP
        HpHpstat =  np.zeros((7*len(self.Lambdas,)))
        if srcProtocol == 'arp':
            for i in range(len(self.Lambdas)):
                HpHpstat[(i*7):((i+1)*7)] = self.HT_Hp.update_get_1D2D_Stats(srcMAC, dstMAC, timestamp, datagramSize, self.Lambdas[i])
        else:  # some other protocol (e.g. TCP/UDP)
            for i in range(len(self.Lambdas)):
                HpHpstat[(i*7):((i+1)*7)] = self.HT_Hp.update_get_1D2D_Stats(srcIP + srcProtocol, dstIP + dstProtocol, timestamp, datagramSize, self.Lambdas[i])

        return np.concatenate((MIstat, Hstat, HHstat, HHstat_jit, HpHpstat))  # concatenation of stats into one stat vector

    def getNetStatHeaders(self):
        MIstat_headers = []
        Hstat_headers = []
        HHstat_headers = []
        HHjitstat_headers = []
        HpHpstat_headers = []

        for i in range(len(self.Lambdas)):
            MIstat_headers += ["MI_dir_"+h for h in self.HT_MI.getHeaders_1D(Lambda=self.Lambdas[i],ID=None)]
            Hstat_headers += ["H_"+h for h in self.HT_H.getHeaders_1D(Lambda=self.Lambdas[i],ID=None)]
            HHstat_headers += ["HH_"+h for h in self.HT_H.getHeaders_1D2D(Lambda=self.Lambdas[i],IDs=None,ver=2)]
            HHjitstat_headers += ["HH_jit_"+h for h in self.HT_jit.getHeaders_1D(Lambda=self.Lambdas[i],ID=None)]
            HpHpstat_headers += ["HpHp_" + h for h in self.HT_Hp.getHeaders_1D2D(Lambda=self.Lambdas[i], IDs=None, ver=2)]
        return MIstat_headers + Hstat_headers + HHstat_headers + HHjitstat_headers + HpHpstat_headers