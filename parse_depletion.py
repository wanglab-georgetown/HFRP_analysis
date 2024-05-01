from scipy.optimize import basinhopping
import statsmodels.api as sm
import os
import sys
import pandas as pd
import pyabf
from scipy import signal
import numpy as np
from scipy.signal import argrelextrema

# run using
# python parse_depletion.py mypath

max_keep = 30


def proc_trace(abf, channel=1):
    T = abf.sweepX[1] - abf.sweepX[0]
    fs = 1 / T

    nc = abf.sweepCount
    res = []
    for i in range(nc):
        print(i)
        abf.setSweep(i, channel)
        abf.sweepY = abf.sweepY * 0.5
        r = proc_single_trace(abf, fs)
        if r[0] is None:
            continue
        res.append(r)
    return res


def proc_single_trace(abf, fs, lowpass_f=400, th=-20, w=10):
    # w is number points to find intercept
    # th is the max peak of spikes

    if abf.sweepY[0] < -40:
        return None, None
    b, a = signal.butter(3, lowpass_f, btype="lowpass", fs=fs, output="ba")
    y_notched = signal.filtfilt(b, a, abf.sweepY)

    order = 80
    idxs = argrelextrema(y_notched, np.less, order=order)[0]
    pks = y_notched[idxs]

    up = pks[pks < th][0] * 1.01
    ids1 = idxs[(pks < th) & (pks > up)]

    # Using the Median Absolute Deviation to Find Outliers
    dd = np.diff(ids1)
    m = np.median(dd)
    v = max(np.median(np.abs(dd - m)) * 1.4826, 0.5)
    # v is typically small for a good trace. choose 15 here to make sure good peaks are not filtered
    up = m + 15 * v
    dn = m - 15 * v

    up = m * 2
    sel = [0, 1]
    p = 1
    for i in range(2, len(ids1)):
        d = ids1[i] - ids1[p]
        if d < up:
            sel.append(i)
            p = i
        else:
            break

    # ii = np.where((dd>dn)&(dd<up))[0]
    # sel = list(set(ii)|set(ii+1))
    # sometimes the gap can be large.
    # sel=np.array(range(sel[-1]+1))

    # sel =[]
    # for j in range(len(ii)-1):
    #     if ii[j+1]-ii[j]>1:
    #         break
    #     sel.append(ii[j+1])
    #     sel.append(ii[j])
    # sel = np.array(sel)
    # sel = list(set(sel)|set(sel+1))

    # sel=[0,1]
    # p=1
    # for i in range(2,len(ids1)):
    #     d = ids1[i]-ids1[p]
    #     if d>dn and d<up:
    #         sel.append(i)
    #         p=i

    ids1 = ids1[sel]

    order = 3
    idxm = argrelextrema(abf.sweepY, np.greater, order=order)[0]
    idxmf = argrelextrema(y_notched, np.greater, order=10)[0]

    res = []
    for i in range(len(ids1)):
        j = ids1[i]
        s = idxm[np.where((idxm < j) & ((j - idxm) > 10))[0][-1]]
        t = idxmf[np.where(idxmf > j)[0][0]]
        res.append([s, t, j])
    res = np.array(res)

    minimizer_kwargs = {"method": "Nelder-Mead", "tol": 1e-6}
    base = np.zeros(len(res))

    fun = lambda w: w[0] * np.exp(-w[1] * t)
    fun1 = lambda w: np.linalg.norm(xs - fun(w))

    for i in range(len(res) - 1):
        if i <= 1:
            a = 0.6
        else:
            a = 0.5
        # tt = abf.sweepY[res[i,2]:res[i,1]]
        tt = y_notched[res[i, 2] : res[i, 1]]
        xs = tt[int(len(tt) * a) :]
        s = int(len(tt) * a) + res[i, 2]
        t = np.linspace(0, 1, len(xs))
        dt = t[1] - t[0]

        w0 = [np.min(xs) * 0.8, 1]

        r = basinhopping(fun1, w0, minimizer_kwargs=minimizer_kwargs, niter=2)
        if r.x[1] < 0:
            raise NameError("negative exponent")

        t = (res[i + 1 :, 2] - s) * dt
        base[i + 1 :] = base[i + 1 :] + fun(r.x)

    aa = abf.sweepY[res[:, 2]] - base

    aa = aa[:max_keep]

    caa = np.cumsum(aa)[-w:]
    # caa = np.cumsum(abf.sweepY[res[:,2]])[-w:]
    xs = np.array(range(len(aa)))[-w:] + 1
    xs = sm.add_constant(xs, prepend=False)

    mod = sm.OLS(caa, xs)

    rl = mod.fit()

    return rl.params[1], aa


if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    path = "."

res = []
print(path)
for file in os.listdir(path):
    if file.endswith(".abf"):
        filepath = os.path.join(path, file)
        abf = pyabf.ABF(filepath)
        print(file)
        rs = proc_trace(abf, channel=1)
        for i in range(len(rs)):
            r = [file, i, rs[i][0]]
            r = r + list(rs[i][1])
            res.append(r)

cols = ["file", "sweep", "intercept"]
cols = cols + [str(i) for i in range(1, max_keep + 1)]
pd.DataFrame(res, columns=cols).to_csv(os.path.join(path, "results.csv"))
