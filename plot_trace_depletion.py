from scipy.optimize import basinhopping
import statsmodels.api as sm
import os
import sys
import pandas as pd
import pyabf
from scipy import signal
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

# run using
# python plot_trace_depletion.py mypath


def plot_fig(abf, f_name):
    res = []
    nc = abf.sweepCount
    channel = 1
    for i in range(nc):
        abf.setSweep(i, channel)
        if abf.sweepY[0] < -350:
            continue
        res.append(abf.sweepY)

    ys = np.mean(res, axis=0)

    lowpass_f = 400
    th = -40
    w = 10
    max_keep = 30

    T = abf.sweepX[1] - abf.sweepX[0]
    fs = 1 / T

    b, a = signal.butter(3, lowpass_f, btype="lowpass", fs=fs, output="ba")
    y_notched = signal.filtfilt(b, a, ys)

    order = 80
    idxs = argrelextrema(y_notched, np.less, order=order)[0]
    pks = y_notched[idxs]

    up = pks[pks < th][0] * 1.01
    ids1 = idxs[(pks < th) & (pks > up)]

    dd = np.diff(ids1)
    m = np.median(dd)
    v = max(np.median(np.abs(dd - m)) * 1.4826, 0.5)
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

    ids1 = ids1[sel]

    order = 3
    idxm = argrelextrema(ys, np.greater, order=order)[0]
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
        # tt = ys[res[i,2]:res[i,1]]
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

    aa = ys[res[:, 2]] - base

    aa = aa[:max_keep]

    caa = np.cumsum(aa)[-w:]
    # caa = np.cumsum(ys[res[:,2]])[-w:]
    xs = np.array(range(len(aa)))[-w:] + 1
    xs = sm.add_constant(xs, prepend=False)

    mod = sm.OLS(caa, xs)

    rl = mod.fit()

    name = f_name.split(".")[0]
    fig = plt.figure(figsize=(16, 9))
    plt.plot(ys[500:7000])
    plt.ylim([-300, 60])
    plt.xlim([0, 6500])
    plt.title(name)
    fig.savefig(name + "_trace.svg")

    fig = plt.figure(figsize=(16, 9))
    plt.plot(range(1, len(aa) + 1), -np.cumsum(aa))
    xt = np.array(range(len(aa))) + 1
    plt.plot(range(1, len(aa) + 1), -(xt * rl.params[0] + rl.params[1]))
    plt.xlim(1, len(aa))
    plt.ylim(0, 5500)
    plt.title(name + ", intercept=" + str(-np.round(rl.params[1])))
    fig.savefig(name + "_cum.svg")


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
        plot_fig(abf, file)
