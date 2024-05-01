import os
import sys
import pandas as pd
import pyabf
import numpy as np
from scipy import signal
from scipy.signal import argrelextrema
from scipy.optimize import basinhopping
import statsmodels.api as sm
import matplotlib.pyplot as plt


class HFRP(object):
    """class for high frequency train analysis"""

    def __init__(
        self,
        filename,
        channel=1,
        lowpass_f=400,
        scaling_factor=1,
        trough_max_amp=-20,
        init_min_amp=-40,
        trough_window=80,
        peak_window=8,
        min_peak_trough_gap=10,
        max_peaks_keep=30,
        intercept_window=10,
        plot_figure=True,
        plot_label_size=15,
        use_exp_fit=True,
        save_res=False,
        fix_baseline=True,
        fig_extension="svg",
        show_fig_box=True,
    ):
        super(HFRP, self).__init__()
        self.filename = filename
        self.base_name = os.path.splitext(os.path.basename(filename))[0]
        self.base_path = os.path.join(os.path.dirname(filename), "results")
        if not os.path.exists(self.base_path):
            os.mkdir(self.base_path)
        self.abf = pyabf.ABF(filename)
        self.channel = channel
        self.lowpass_f = lowpass_f
        self.trough_max_amp = trough_max_amp
        self.init_min_amp = init_min_amp
        self.trough_window = trough_window  # peak detection window size
        self.min_peak_trough_gap = min_peak_trough_gap
        self.max_peaks_keep = max_peaks_keep
        self.scaling_factor = scaling_factor
        self.peak_window = peak_window
        self.intercept_window = intercept_window
        self.plot_figure = plot_figure
        self.plot_label_size = plot_label_size
        self.use_exp_fit = use_exp_fit
        self.save_res = save_res
        self.fix_baseline = fix_baseline  # sometimes baseline is not zero
        self.fig_extension = fig_extension
        self.show_fig_box = show_fig_box

        self.T = self.abf.sweepX[1] - self.abf.sweepX[0]
        self.fs = 1 / self.T

    def exp_int(self, xs):
        t1 = 0
        y1 = xs[t1]
        t3 = len(xs) - 1
        y3 = xs[t3]
        b = np.nan
        if t3 % 2 == 0:
            t2 = int(t3 / 2)
            y2 = xs[t2]
        else:
            t2 = t3 * 0.5
            b = int(np.floor(t2))
            y2 = (xs[b] + xs[b + 1]) * 0.5
        # (a-a*e^(bt/2))/(a*e^(bt/2)-a*e^(bt))=e^(-bt/2)
        r = (y1 - y2) / (y2 - y3)

        if r > 1:
            b = 2 / (t1 - t3) * np.log(r)
        else:
            b = -1e-8
        a = (y1 - y3) / (np.exp(b * t1) - np.exp(b * t3))
        return [a, b]

    def plot_trace(self, sweep_idx, s, t, Y):
        fname = os.path.join(
            self.base_path,
            "{}_{}_trace.{}".format(self.base_name, sweep_idx, self.fig_extension),
        )

        ts = np.array(range(t - s)) * self.T * 1000

        fig = plt.figure(figsize=(16, 9))
        plt.plot(ts, Y[s:t])
        ymax = max(60, np.max(Y[s:t]))
        ymin = min(-300, np.min(Y[s:t]))
        plt.ylim([ymin, ymax])
        plt.xlim([0, ts[-1]])
        plt.xlabel("time (ms)", fontsize=self.plot_label_size)
        plt.ylabel(self.sweepUnitsY, fontsize=self.plot_label_size)
        plt.title(self.base_name + ", sweep index {}".format(sweep_idx))
        fig.savefig(fname)

    def proc_trace(self):
        nc = self.abf.sweepCount
        res0 = []
        for i in range(nc):
            print(i)
            find_sol = True
            try:
                r = self.proc_single_trace(i)
                if len(r) == 0:
                    find_sol = False
                r["success"] = 1
            except:
                find_sol = False

            if not find_sol:
                r = {}
                r["success"] = 0
                r["filename"] = self.base_name
                print("bad sweep")

            r["sweep"] = i
            res0.append(r)

        if self.save_res:
            pd.DataFrame(res0).to_csv(os.path.join(self.base_path, "results.csv"))
        return res0

    def proc_single_trace(self, sweep_idx=0):
        self.abf.setSweep(sweep_idx, self.channel)
        Y = self.abf.sweepY * self.scaling_factor
        self.sweepUnitsY = self.abf.sweepUnitsY

        # this indicate a bad trace
        if Y[0] < self.init_min_amp:
            self.plot_trace(sweep_idx, 0, len(Y), Y)
            return {}
        b, a = signal.butter(
            3, self.lowpass_f, btype="lowpass", fs=self.fs, output="ba"
        )
        y_notched = signal.filtfilt(b, a, Y)

        # first pass use filtered signal
        idxs = argrelextrema(y_notched, np.less, order=self.trough_window)[0]
        pks = y_notched[idxs]
        base_line = np.median(y_notched[: idxs[0]])

        # find the first peak and peaks with amplitude lower than the first one are just noise
        up = pks[(pks - base_line) < self.trough_max_amp][0] * 1.01
        idxs1 = idxs[((pks - base_line) < self.trough_max_amp) & (pks > up)]

        # # Using the Median Absolute Deviation to Find Outliers
        dd = np.diff(idxs1)
        m = np.median(dd)
        # v = max(np.median(np.abs(dd - m)) * 1.4826, 0.5)
        # # v is typically small for a good trace. choose 15 here to make sure good peaks are not filtered
        # up = m + 15 * v
        # dn = m - 15 * v

        # up is a bound used to identify when the trace stops
        up = m * 2
        sel = [0, 1]
        p = 1
        for i in range(2, len(idxs1)):
            d = idxs1[i] - idxs1[p]
            if d < up:
                sel.append(i)
                p = i
            else:
                break

        idxs1 = idxs1[sel]

        # find peak between two troughs to do an exponential fitting between trough and next peak
        idxmf = argrelextrema(y_notched, np.greater, order=self.peak_window)[0]

        pidxs = []
        tidxs = []
        for i in range(len(idxs1)):
            s = idxs1[i]
            t = idxmf[
                np.where(
                    (idxmf > s + self.min_peak_trough_gap)
                    & (y_notched[idxmf] > 0.5 * y_notched[s])
                )[0][0]
            ]
            pidxs.append(s)
            tidxs.append(t)
        pidxs = np.array(pidxs)
        tidxs = np.array(tidxs)

        rise_time = int(np.mean(pidxs[1:] - tidxs[:-1]))
        base_line = 0
        if self.fix_baseline:
            base_line = np.median(Y[: int(rise_time * 0.98)])
            Y = Y - base_line
            y_notched = y_notched - base_line

        minimizer_kwargs = {"method": "Nelder-Mead", "tol": 1e-6}

        fun = lambda w: w[0] * np.exp(-w[1] * ts)
        fun1 = lambda w: np.linalg.norm(xs - fun(w))

        Yt = Y.copy()
        amps = []
        exp_params = []
        Y1 = Y.copy() * 0
        pidxs = []
        for i in range(len(idxs1)):
            s = idxs1[i]
            t = tidxs[i]
            s1 = max(int(s - rise_time * 0.25), 0)
            t1 = min(int(s + rise_time * 0.25), len(Y))
            s2 = np.argmin(Yt[s1:t1]) + s1
            pidxs.append(s2)
            amps.append(Yt[s2])
            s = s2

            # amps.append(Yt[s])

            if i == len(idxs1) - 1:
                break

            if i <= 1:
                a = 0.6
            else:
                a = 0.5
            b = 0.95

            tt = Yt[s:t]
            idxt = np.where(tt < tt[0] * (1 - a))[0]
            if len(idxt) > 0:
                s1 = idxt[-1]
            else:
                s1 = int(len(tt) * a)

            idxt = np.where(tt > tt[0] * (1 - b))[0]
            if len(idxt) > 0:
                t1 = idxt[idxt > s1][0]
            else:
                t1 = int(len(tt) * b)

            # this is a stable region for exponential fit
            xs = tt[s1:t1]
            s2 = s1 + s

            if self.use_exp_fit:
                w0 = self.exp_int(xs)
                ts = np.array(range(len(xs)))
                r = basinhopping(fun1, w0, minimizer_kwargs=minimizer_kwargs, niter=2)

                if r.x[1] < 0:
                    # print(s, t)
                    # print(s1, t1)

                    # plt.plot(xs)
                    # plt.show()
                    # print(idxs1)
                    # print(r)
                    # import pdb
                    # pdb.set_trace()

                    self.plot_trace(sweep_idx, 0, len(Y), Y)

                    raise NameError("negative exponent")

                # remove exponential decay from all future signals
                ts = np.array(range(len(Yt) - s2))
                yt = fun(r.x)
                Yt[s2:] = Yt[s2:] - yt
                Y1[s2:] = Y1[s2:] + yt
                Y1[s2] = np.nan
                exp_params.append(r.x)

        pidxs = np.array(pidxs)

        aa = amps[: self.max_peaks_keep]

        caa = np.cumsum(aa)[-self.intercept_window :]
        xs = np.array(range(len(aa)))[-self.intercept_window :] + 1
        xs = sm.add_constant(xs, prepend=False)

        mod = sm.OLS(caa, xs)

        rl = mod.fit()

        res = {
            "filename": self.base_name,
            "intercept": rl.params[1],
            "amps": aa,
            "pidxs": np.array(pidxs[: self.max_peaks_keep]),
            "tidxs": np.array(tidxs[: self.max_peaks_keep]),
            "exp_params": exp_params[: self.max_peaks_keep],
            "base_line": base_line,
            "num_peaks": len(aa),
        }

        if self.plot_figure:
            fname = os.path.join(
                self.base_path,
                "{}_{}_trace.{}".format(self.base_name, sweep_idx, self.fig_extension),
            )

            rise_time = int(np.mean(res["pidxs"][1:] - res["tidxs"][:-1]))
            # decay_time = int(np.median(res["pidxs"][:-1] - res["tidxs"][:-1]))
            s = max(res["pidxs"][0] - 2 * rise_time, 0)
            # t = min(res["pidxs"][-1] + decay_time,len(Y))
            t = res["tidxs"][-1]
            ts = np.array(range(t - s)) * self.T * 1000

            # fig = plt.figure(figsize=(16, 9))
            fig, ax = plt.subplots(1, 1, figsize=(16, 9))
            plt.plot(ts, Y[s:t])
            ymax = max(60, np.max(Y[s:t]))
            ymin = min(-300, np.min(Y[s:t]))
            plt.ylim([ymin, ymax])
            plt.xlim([0, ts[-1]])
            plt.xlabel("time (ms)", fontsize=self.plot_label_size)
            plt.ylabel(self.sweepUnitsY, fontsize=self.plot_label_size)
            if self.show_fig_box:
                plt.title(self.base_name + ", sweep index {}".format(sweep_idx))
            else:
                ax.spines[["right", "top"]].set_visible(False)

            fig.savefig(fname)
            plt.close(fig)

            fname = os.path.join(
                self.base_path,
                "{}_{}_fit.{}".format(self.base_name, sweep_idx, self.fig_extension),
            )

            # fig = plt.figure(figsize=(16, 9))
            fig, ax = plt.subplots(1, 1, figsize=(16, 9))
            ts = np.array(range(len(Y1))) * self.T * 1000
            plt.plot(ts, Y)
            plt.plot(ts, Y1)
            for i in res["pidxs"]:
                plt.plot(ts[i], Y[i], "ro")
            plt.ylim([ymin, ymax])
            plt.xlim([ts[s], ts[t]])
            plt.xlabel("time (ms)", fontsize=self.plot_label_size)
            plt.ylabel(self.sweepUnitsY, fontsize=self.plot_label_size)
            if self.show_fig_box:
                plt.title(self.base_name + ", sweep index {}".format(sweep_idx))
            else:
                ax.spines[["right", "top"]].set_visible(False)
            fig.savefig(fname)
            plt.close(fig)

            fname = os.path.join(
                self.base_path,
                "{}_{}_cum.{}".format(self.base_name, sweep_idx, self.fig_extension),
            )

            # fig = plt.figure(figsize=(16, 9))
            fig, ax = plt.subplots(1, 1, figsize=(16, 9))
            plt.plot(range(1, len(aa) + 1), -np.cumsum(aa))
            xt = np.array(range(len(aa) + 1))
            plt.plot(xt, -(xt * rl.params[0] + rl.params[1]))
            plt.xlim(0, len(aa))
            ymax = max(5500, np.max(aa))
            plt.ylim(0, ymax)
            plt.xlabel("peak number", fontsize=self.plot_label_size)
            plt.ylabel(self.sweepUnitsY, fontsize=self.plot_label_size)
            if self.show_fig_box:
                plt.title(
                    self.base_name
                    + ", sweep index {}, intercept=".format(sweep_idx)
                    + str(-np.round(rl.params[1]))
                )
            else:
                ax.spines[["right", "top"]].set_visible(False)
            fig.savefig(fname)
            plt.close(fig)

        return res


if __name__ == "__main__":
    is_dir = True
    files = []
    if len(sys.argv) > 1:
        if os.path.isdir(sys.argv[1]):
            path = sys.argv[1]
        else:
            is_dir = False
            files = [sys.argv[1]]
            base_path = os.path.join(os.path.dirname(sys.argv[1]), "results")
    else:
        path = "."

    if is_dir:
        base_path = os.path.join(path, "results")
        for file in os.listdir(path):
            if file.endswith(".abf"):
                filepath = os.path.join(path, file)
                files.append(filepath)

    res = []
    for file in files:
        if file.endswith(".abf") and os.path.exists(file):
            print(file)
            hfrpo = HFRP(file)
            rs = hfrpo.proc_trace()
            res = res + rs

    if len(res) > 0:
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        pd.DataFrame(res).to_csv(os.path.join(base_path, "results.csv"))
