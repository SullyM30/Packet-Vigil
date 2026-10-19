# cython: boundscheck=False, wraparound=False, cdivision=True
# cython: language_level=3
"""
Cython-optimized AfterImage — cdef classes with typed attributes.

Drop-in replacement for AfterImage.py with ~10-30x speedup on feature extraction.
All public method signatures remain identical so netStat.py works unchanged.

Based on the original Kitsune AfterImage_extrapolate.pyx by Yisroel Mirsky,
adapted for our modified AfterImage with batch LRU pruning.

Compile with:  py -3.13 setup.py build_ext --inplace
"""

from libc.math cimport pow as c_pow, sqrt as c_sqrt, isnan as c_isnan, fabs
import numpy as np


# ────────────────────────────────────────────────────────────────────
#  incStat  — per-stream exponentially-weighted moving average stats
# ────────────────────────────────────────────────────────────────────
cdef class incStat:
    cdef public str ID
    cdef double CF1, CF2, w
    cdef double Lambda
    cdef public double lastTimestamp
    cdef double cur_mean, cur_var, cur_std
    cdef int isTypeDiff
    cdef public list covs
    cdef public int MAX_COVS

    def __init__(self, double Lambda, str ID, double init_time=0, int isTypeDiff=False):
        self.ID = ID
        self.CF1 = 0.0
        self.CF2 = 0.0
        self.w = 1e-20
        self.isTypeDiff = isTypeDiff
        self.Lambda = Lambda
        self.lastTimestamp = init_time
        self.cur_mean = float('nan')
        self.cur_var = float('nan')
        self.cur_std = float('nan')
        self.covs = []
        self.MAX_COVS = 50

    cdef void insert(self, double v, double t):
        cdef double dif
        cdef incStat_cov cov

        if self.isTypeDiff:
            dif = t - self.lastTimestamp
            v = dif if dif > 0.0 else 0.0

        self._processDecay(t)

        # update with v
        self.CF1 += v
        self.CF2 += v * v
        self.w += 1.0
        self.cur_mean = float('nan')
        self.cur_var = float('nan')
        self.cur_std = float('nan')

        # update covs (if any)
        for c in self.covs:
            cov = c
            cov._update_cov(self.ID, v, t)

    cdef double _processDecay(self, double timestamp):
        cdef double factor = 1.0
        cdef double timeDiff = timestamp - self.lastTimestamp
        if timeDiff > 0.0:
            factor = c_pow(2.0, -self.Lambda * timeDiff)
            self.CF1 *= factor
            self.CF2 *= factor
            self.w *= factor
            self.lastTimestamp = timestamp
        return factor

    # Python-visible wrapper for external callers
    def processDecay(self, double timestamp):
        return self._processDecay(timestamp)

    cdef double _weight(self):
        return self.w

    cdef double _mean(self):
        if c_isnan(self.cur_mean):
            self.cur_mean = self.CF1 / self.w
        return self.cur_mean

    cdef double _var(self):
        cdef double m
        if c_isnan(self.cur_var):
            m = self._mean()
            self.cur_var = fabs(self.CF2 / self.w - m * m)
        return self.cur_var

    cdef double _std(self):
        if c_isnan(self.cur_std):
            self.cur_std = c_sqrt(self._var())
        return self.cur_std

    # Python-visible wrappers
    def weight(self): return self._weight()
    def mean(self): return self._mean()
    def var(self): return self._var()
    def std(self): return self._std()

    def cov(self, str ID2):
        cdef incStat_cov cov
        for c in self.covs:
            cov = c
            if cov.incS1.ID == ID2 or cov.incS2.ID == ID2:
                return cov._cov()
        return float('nan')

    def pcc(self, str ID2):
        cdef incStat_cov cov
        for c in self.covs:
            cov = c
            if cov.incS1.ID == ID2 or cov.incS2.ID == ID2:
                return cov._pcc()
        return float('nan')

    cdef list _cov_pcc(self, str ID2):
        cdef incStat_cov cov
        for c in self.covs:
            cov = c
            if cov.incS1.ID == ID2 or cov.incS2.ID == ID2:
                return cov._get_stats1()
        return [float('nan'), float('nan')]

    def cov_pcc(self, str ID2):
        return self._cov_pcc(ID2)

    cdef double _radius(self, list other_incStats):
        cdef double A = self._var() * self._var()
        cdef incStat incS
        for o in other_incStats:
            incS = o
            A += incS._var() * incS._var()
        return c_sqrt(A)

    cdef double _magnitude(self, list other_incStats):
        cdef double A = self._mean() * self._mean()
        cdef incStat incS
        for o in other_incStats:
            incS = o
            A += incS._mean() * incS._mean()
        return c_sqrt(A)

    def radius(self, other_incStats):
        return self._radius(other_incStats)

    def magnitude(self, other_incStats):
        return self._magnitude(other_incStats)

    cdef list _allstats_1D(self):
        self.cur_mean = self.CF1 / self.w
        self.cur_var = fabs(self.CF2 / self.w - self.cur_mean * self.cur_mean)
        return [self.w, self.cur_mean, self.cur_var]

    def allstats_1D(self):
        return self._allstats_1D()

    cdef list _allstats_2D(self, str ID2):
        cdef list stats1D = self._allstats_1D()
        cdef list stats2D = [float('nan')] * 4
        cdef incStat_cov cov
        for c in self.covs:
            cov = c
            if cov.incS1.ID == ID2 or cov.incS2.ID == ID2:
                stats2D = cov._get_stats2()
                break
        return stats1D + stats2D

    def allstats_2D(self, str ID2):
        return self._allstats_2D(ID2)

    def getHeaders_1D(self, bint suffix=True):
        cdef str s0
        if self.ID is None:
            s0 = ""
        else:
            s0 = "_0"
        if suffix:
            s0 = "_" + self.ID
        return ["weight" + s0, "mean" + s0, "std" + s0]

    def getHeaders_2D(self, str ID2, bint suffix=True):
        hdrs1D = self.getHeaders_1D(suffix)
        cdef str s0, s1
        if self.ID is None:
            s0 = ""
            s1 = ""
        else:
            s0 = "_0"
            s1 = "_1"
        if suffix:
            s0 = "_" + self.ID
            s1 = "_" + ID2
        hdrs2D = ["radius_" + s0 + "_" + s1, "magnitude_" + s0 + "_" + s1,
                   "covariance_" + s0 + "_" + s1, "pcc_" + s0 + "_" + s1]
        return hdrs1D + hdrs2D


# ────────────────────────────────────────────────────────────────────
#  incStat_cov — cross-stream covariance / Pearson correlation
# ────────────────────────────────────────────────────────────────────
cdef class incStat_cov:
    cdef public incStat incS1
    cdef public incStat incS2
    cdef double lastRes0, lastRes1
    cdef double CF3, w3
    cdef public double lastTimestamp_cf3

    def __init__(self, incStat incS1, incStat incS2, double init_time=0):
        self.incS1 = incS1
        self.incS2 = incS2
        self.lastRes0 = 0.0
        self.lastRes1 = 0.0
        self.CF3 = 0.0
        self.w3 = 1e-20
        self.lastTimestamp_cf3 = init_time

    # Property for backward compatibility — code that accesses .incStats[0]/[1]
    @property
    def incStats(self):
        return [self.incS1, self.incS2]

    cdef void _update_cov(self, str ID, double v, double t):
        cdef int inc
        cdef double res, resid

        if ID == self.incS1.ID:
            inc = 0
        elif ID == self.incS2.ID:
            inc = 1
        else:
            return  # error

        # Decay other incStat
        if inc == 0:
            self.incS2._processDecay(t)
        else:
            self.incS1._processDecay(t)

        # Decay residuals
        self.__processDecay(t, inc)

        # Compute and update residual
        if inc == 0:
            res = v - self.incS1._mean()
            resid = res * self.lastRes1
            self.lastRes0 = res
        else:
            res = v - self.incS2._mean()
            resid = res * self.lastRes0
            self.lastRes1 = res

        self.CF3 += resid
        self.w3 += 1.0

    def update_cov(self, str ID, double v, double t):
        self._update_cov(ID, v, t)

    cdef double __processDecay(self, double t, int micro_inc_indx):
        cdef double factor = 1.0
        cdef double timeDiff = t - self.lastTimestamp_cf3
        cdef double lam
        if timeDiff > 0.0:
            if micro_inc_indx == 0:
                lam = self.incS1.Lambda
            else:
                lam = self.incS2.Lambda
            factor = c_pow(2.0, -lam * timeDiff)
            self.CF3 *= factor
            self.w3 *= factor
            self.lastTimestamp_cf3 = t
            if micro_inc_indx == 0:
                self.lastRes0 *= factor
            else:
                self.lastRes1 *= factor
        return factor

    def processDecay(self, double t, int micro_inc_indx):
        return self.__processDecay(t, micro_inc_indx)

    cdef double _cov(self):
        return self.CF3 / self.w3

    cdef double _pcc(self):
        cdef double ss = self.incS1._std() * self.incS2._std()
        if ss != 0.0:
            return self._cov() / ss
        return 0.0

    def cov(self): return self._cov()
    def pcc(self): return self._pcc()

    cdef list _get_stats1(self):
        return [self._cov(), self._pcc()]

    cdef list _get_stats2(self):
        return [self.incS1._mean(), self.incS1._var(), 0.0, 0.0]

    def get_stats1(self): return self._get_stats1()
    def get_stats2(self): return self._get_stats2()

    def get_stats3(self):
        return [self.incS1.w, self.incS1._mean(), self.incS1._std(),
                self.incS2.w, self.incS2._mean(), self.incS2._std(),
                self._cov(), self._pcc()]

    def get_stats4(self):
        return [self.incS1.w, self.incS1._mean(), self.incS1._std(),
                self.incS2.w, self.incS2._mean(), self.incS2._std(),
                self.incS1._radius([self.incS2]), self.incS1._magnitude([self.incS2]),
                self._cov(), self._pcc()]

    def getHeaders(self, int ver, bint suffix=True):
        cdef str s0 = "0"
        cdef str s1 = "1"
        if suffix:
            s0 = self.incS1.ID
            s1 = self.incS2.ID

        if ver == 1:
            return ["covariance_" + s0 + "_" + s1, "pcc_" + s0 + "_" + s1]
        if ver == 2:
            return ["radius_" + s0 + "_" + s1, "magnitude_" + s0 + "_" + s1,
                    "covariance_" + s0 + "_" + s1, "pcc_" + s0 + "_" + s1]
        if ver == 3:
            return ["weight_" + s0, "mean_" + s0, "std_" + s0,
                    "weight_" + s1, "mean_" + s1, "std_" + s1,
                    "covariance_" + s0 + "_" + s1, "pcc_" + s0 + "_" + s1]
        if ver == 4:
            return ["weight_" + s0, "mean_" + s0, "std_" + s0,
                    "covariance_" + s0 + "_" + s1, "pcc_" + s0 + "_" + s1]
        if ver == 5:
            return ["weight_" + s0, "mean_" + s0, "std_" + s0,
                    "weight_" + s1, "mean_" + s1, "std_" + s1,
                    "radius_" + s0 + "_" + s1, "magnitude_" + s0 + "_" + s1,
                    "covariance_" + s0 + "_" + s1, "pcc_" + s0 + "_" + s1]
        return []


# ────────────────────────────────────────────────────────────────────
#  incStatDB — hash-table database of incStat objects with LRU pruning
# ────────────────────────────────────────────────────────────────────
cdef class incStatDB:
    cdef public dict HT
    cdef double limit
    cdef double df_lambda
    cdef public bint enable_pruning
    cdef public int pruned_count

    def __init__(self, double limit=float('inf'), double default_lambda=float('nan'),
                 bint enable_pruning=True):
        self.HT = {}
        self.limit = limit
        self.df_lambda = default_lambda
        self.enable_pruning = enable_pruning
        self.pruned_count = 0

    cdef double _get_lambda(self, double Lambda):
        if not c_isnan(self.df_lambda):
            return self.df_lambda
        return Lambda

    def get_lambda(self, double Lambda):
        return self._get_lambda(Lambda)

    cdef int _prune_oldest_batch(self, int batch_size=100):
        """Batch LRU Pruning: remove multiple least recently used entries."""
        cdef int n = len(self.HT)
        if n == 0:
            return 0

        cdef int num_to_prune = min(n, max(batch_size, <int>(n * 0.1)))
        cdef list entries = []
        cdef str key
        cdef incStat incS
        cdef incStat other_incS
        cdef incStat_cov cov_ref
        cdef int pruned = 0
        cdef int i

        # Build (key, timestamp) list
        for key in self.HT:
            incS = <incStat>self.HT[key]
            entries.append((key, incS.lastTimestamp))

        # Sort by timestamp (oldest first)
        entries.sort(key=lambda x: x[1])

        # Remove oldest entries and cleanup covariance references
        for i in range(num_to_prune):
            key = entries[i][0]
            incS = <incStat>self.HT[key]

            # Clean up covariance references
            for c in incS.covs:
                cov_ref = <incStat_cov>c
                if cov_ref.incS1 is not incS:
                    other_incS = cov_ref.incS1
                else:
                    other_incS = cov_ref.incS2
                if other_incS is not None:
                    try:
                        other_incS.covs.remove(cov_ref)
                    except ValueError:
                        pass

            incS.covs.clear()
            del self.HT[key]
            pruned += 1

        self.pruned_count += pruned

        if self.pruned_count % 1000 == 0:
            total_covs = 0
            for v in self.HT.values():
                total_covs += len((<incStat>v).covs)
            avg_covs = <double>total_covs / len(self.HT) if len(self.HT) > 0 else 0.0
            print(f"[AfterImage] Pruned {self.pruned_count} old flows (batch={pruned}) | "
                  f"HT_size={len(self.HT)} | avg_covs_per_flow={avg_covs:.1f}")

        return pruned

    cdef incStat _register(self, str ID, double Lambda, double init_time,
                            int isTypeDiff):
        cdef str key = ID + "_" + str(Lambda)
        cdef incStat incS = <incStat>self.HT.get(key)

        if incS is None:
            if len(self.HT) >= <int>(self.limit * 0.95):
                if self.enable_pruning:
                    self._prune_oldest_batch(max(100, <int>(self.limit * 0.1)))
                else:
                    if len(self.HT) + 1 > <int>self.limit:
                        raise LookupError(
                            'Adding Entry:\n' + key +
                            '\nwould exceed incStatHT 1D limit of ' + str(self.limit) +
                            '.\nObservation Rejected.')
            incS = incStat(Lambda, ID, init_time, isTypeDiff)
            self.HT[key] = incS
        return incS

    def register(self, str ID, double Lambda=1, double init_time=0,
                 bint isTypeDiff=False):
        cdef double L = self._get_lambda(Lambda)
        return self._register(ID, L, init_time, isTypeDiff)

    cdef incStat_cov _register_cov(self, str ID1, str ID2, double Lambda,
                                     double init_time, int isTypeDiff):
        cdef incStat incS1 = self._register(ID1, Lambda, init_time, isTypeDiff)
        cdef incStat incS2 = self._register(ID2, Lambda, init_time, isTypeDiff)
        cdef incStat_cov cov_ref, oldest_cov
        cdef incStat other

        # Check for pre-existing link
        for c in incS1.covs:
            cov_ref = <incStat_cov>c
            if cov_ref.incS1.ID == ID2 or cov_ref.incS2.ID == ID2:
                return cov_ref

        # Limit covariances — LRU eviction
        if len(incS1.covs) >= incS1.MAX_COVS:
            oldest_cov = min(incS1.covs, key=lambda c: (<incStat_cov>c).lastTimestamp_cf3)
            incS1.covs.remove(oldest_cov)
            other = oldest_cov.incS1 if oldest_cov.incS1 is not incS1 else oldest_cov.incS2
            if other is not None:
                try:
                    other.covs.remove(oldest_cov)
                except ValueError:
                    pass

        if len(incS2.covs) >= incS2.MAX_COVS:
            oldest_cov = min(incS2.covs, key=lambda c: (<incStat_cov>c).lastTimestamp_cf3)
            incS2.covs.remove(oldest_cov)
            other = oldest_cov.incS1 if oldest_cov.incS1 is not incS2 else oldest_cov.incS2
            if other is not None:
                try:
                    other.covs.remove(oldest_cov)
                except ValueError:
                    pass

        # Create new link
        cdef incStat_cov inc_cov = incStat_cov(incS1, incS2, init_time)
        incS1.covs.append(inc_cov)
        incS2.covs.append(inc_cov)
        return inc_cov

    def register_cov(self, str ID1, str ID2, double Lambda=1,
                     double init_time=0, bint isTypeDiff=False):
        cdef double L = self._get_lambda(Lambda)
        return self._register_cov(ID1, ID2, L, init_time, isTypeDiff)

    cdef incStat _update(self, str ID, double t, double v, double Lambda,
                          int isTypeDiff):
        cdef incStat incS = self._register(ID, Lambda, t, isTypeDiff)
        incS.insert(v, t)
        return incS

    def update(self, str ID, double t, double v, double Lambda=1,
               bint isTypeDiff=False):
        cdef double L = self._get_lambda(Lambda)
        return self._update(ID, t, v, L, isTypeDiff)

    def get_1D_Stats(self, str ID, double Lambda=1):
        cdef double L = self._get_lambda(Lambda)
        cdef incStat incS = <incStat>self.HT.get(ID + "_" + str(L))
        if incS is None:
            return [float('nan')] * 3
        return incS._allstats_1D()

    def get_2D_Stats(self, str ID1, str ID2, double Lambda=1):
        cdef double L = self._get_lambda(Lambda)
        cdef incStat incS1 = <incStat>self.HT.get(ID1 + "_" + str(L))
        if incS1 is None:
            return [float('nan')] * 2
        return incS1._cov_pcc(ID2)

    def get_all_2D_Stats(self, str ID, double Lambda=1):
        cdef double L = self._get_lambda(Lambda)
        cdef incStat incS1 = <incStat>self.HT.get(ID + "_" + str(L))
        if incS1 is None:
            return ([], [])
        cdef list stats = []
        cdef list IDs = []
        cdef incStat_cov cov_ref
        for c in incS1.covs:
            cov_ref = <incStat_cov>c
            stats.append(cov_ref._get_stats1())
            IDs.append([cov_ref.incS1.ID, cov_ref.incS2.ID])
        return stats, IDs

    def get_nD_Stats(self, list IDs, double Lambda=1):
        cdef double L = self._get_lambda(Lambda)
        cdef incStat incS
        cdef str ID
        cdef double rad = 0.0
        cdef double mag = 0.0
        for ID in IDs:
            incS = <incStat>self.HT.get(ID + "_" + str(L))
            if incS is not None:
                rad += incS._var()
                mag += incS._mean() * incS._mean()
        return [c_sqrt(rad), c_sqrt(mag)]

    # ── Hot-path methods: called from Python (netStat.py) for every packet ──
    def update_get_1D_Stats(self, str ID, double t, double v,
                            double Lambda=1, bint isTypeDiff=False):
        cdef double L = self._get_lambda(Lambda)
        cdef incStat incS = self._update(ID, t, v, L, isTypeDiff)
        return incS._allstats_1D()

    def update_get_2D_Stats(self, str ID1, str ID2, double t1, double v1,
                            double Lambda=1, int level=1):
        cdef double L = self._get_lambda(Lambda)
        cdef incStat_cov inc_cov = self._register_cov(ID1, ID2, L, t1, False)
        inc_cov._update_cov(ID1, v1, t1)
        if level == 1:
            return inc_cov._get_stats1()
        else:
            return inc_cov._get_stats2()

    def update_get_1D2D_Stats(self, str ID1, str ID2, double t1, double v1,
                              double Lambda=1):
        cdef double L = self._get_lambda(Lambda)
        cdef incStat incS = self._update(ID1, t1, v1, L, False)
        cdef list stats_1D = incS._allstats_1D()
        stats_1D[2] = c_sqrt(stats_1D[2])  # var -> std

        cdef incStat_cov inc_cov = self._register_cov(ID1, ID2, L, t1, False)
        inc_cov._update_cov(ID1, v1, t1)
        return stats_1D + inc_cov._get_stats2()

    def getHeaders_1D(self, double Lambda=1, ID=None):
        cdef double L = self._get_lambda(Lambda)
        cdef str id_str = str(ID) if ID is not None else ""
        hdrs = incStat(L, id_str).getHeaders_1D(suffix=False)
        return [str(L) + "_" + s for s in hdrs]

    def getHeaders_2D(self, double Lambda=1, IDs=None, int ver=1):
        cdef double L = self._get_lambda(Lambda)
        if IDs is None:
            IDs = [0, 1]
        hdrs = incStat_cov(incStat(L, str(IDs[0])),
                           incStat(L, str(IDs[0])), L).getHeaders(ver, suffix=False)
        return [str(L) + "_" + s for s in hdrs]

    def getHeaders_1D2D(self, double Lambda=1, IDs=None, int ver=1):
        cdef double L = self._get_lambda(Lambda)
        if IDs is None:
            IDs = [0, 1]
        hdrs1D = self.getHeaders_1D(L, IDs[0])
        hdrs2D = self.getHeaders_2D(L, IDs, ver)
        return hdrs1D + hdrs2D

    def getHeaders_nD(self, double Lambda=1, list IDs=[]):
        cdef str ID = ":"
        for s in IDs:
            ID += "_" + str(s)
        cdef double L = self._get_lambda(Lambda)
        hdrs = ["radius" + ID, "magnitude" + ID]
        return [str(L) + "_" + s for s in hdrs]

    def cleanOutOldRecords(self, double cutoffWeight, double curTime):
        cdef int n = 0
        dump = sorted(self.HT.items(), key=lambda tup: tup[1][0].getMaxW(curTime))
        for entry in dump:
            entry[1][0].processDecay(curTime)
            W = entry[1][0].w
            if W <= cutoffWeight:
                key = entry[0]
                del entry[1][0]
                del self.HT[key]
                n += 1
            elif W > cutoffWeight:
                break
        return n
