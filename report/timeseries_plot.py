import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from data_loader.loader import load_raw, clean_and_impute
from optim.regime import cluster_regimes
from report.plot_style import setup, style_ax, PALETTE, COLOR

import yaml


def _regime_spans(labels, ts_pd, k) :
    spans = {i : [] for i in range(k)}
    for i in range(k) :
        mask = labels == i
        if not mask.any() :
            continue
        idx = np.where(mask)[0]
        runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
        for run in runs :
            if len(run) == 0 :
                continue
            spans[i].append((ts_pd[run[0]], ts_pd[run[-1]]))
    return spans


def _paint_bands(ax, spans, palette, alpha = 0.07) :
    for i, segs in spans.items() :
        for t0, t1 in segs :
            ax.axvspan(t0, t1, alpha = alpha, color = palette[i % len(palette)], zorder = 0, linewidth = 0)


def _paint_labels(ax, spans, palette, y_frac = 0.93) :
    for i, segs in spans.items() :
        longest = max(segs, key = lambda s : s[1] - s[0])
        mid = longest[0] + (longest[1] - longest[0]) / 2
        yl = ax.get_ylim()
        ax.text(mid, yl[0] + (yl[1] - yl[0]) * y_frac, f"R{i}",
                ha = "center", va = "top", fontsize = 8,
                color = palette[i % len(palette)], fontweight = "bold",
                bbox = dict(boxstyle = "round,pad=0.12", fc = "white", ec = "none", alpha = 0.75))


def main() :
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "config.yaml"), "r", encoding = "utf-8") as f :
        cfg = yaml.safe_load(f)

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base, cfg["csv_path"]) if not os.path.isabs(cfg["csv_path"]) else cfg["csv_path"]
    out_dir = os.path.join(base, cfg["out_dir"])
    os.makedirs(out_dir, exist_ok = True)

    setup()

    df = load_raw(csv_path)
    df = clean_and_impute(df)

    regime_result = cluster_regimes(df, k = cfg.get("regime_k", 5), seed = cfg.get("seed", 42))
    labels = np.array(regime_result["labels"])
    k = regime_result["k"]

    ts_pd = pd.to_datetime(df["timestamp"].values)
    spans = _regime_spans(labels, ts_pd, k)

    step = 5
    ts_ds = ts_pd[::step]

    # ---- Figure 1: input conditions ----
    fig1, axes1 = plt.subplots(3, 1, figsize = (9, 6), sharex = True,
                                gridspec_kw = {"hspace" : 0.15})

    axes1[0].plot(ts_ds, df["C_in_gNm3"].values[::step], color = PALETTE[0], linewidth = 0.8)
    axes1[0].set_ylabel(r"$C_{in}$ (g/Nm$^3$)")
    style_ax(axes1[0])

    axes1[1].plot(ts_ds, df["Temp_C"].values[::step], color = PALETTE[1], linewidth = 0.8)
    axes1[1].set_ylabel(r"$T_{in}$ ($^\circ$C)")
    style_ax(axes1[1])

    axes1[2].plot(ts_ds, df["Q_Nm3h"].values[::step] / 1000, color = PALETTE[2], linewidth = 0.8)
    axes1[2].set_ylabel(r"$Q$ ($\times10^3$ Nm$^3$/h)")
    style_ax(axes1[2])

    for ax in axes1 :
        _paint_bands(ax, spans, PALETTE)
    _paint_labels(axes1[0], spans, PALETTE)

    axes1[-1].xaxis.set_major_locator(mdates.DayLocator())
    axes1[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig1.tight_layout()
    fig1.savefig(os.path.join(out_dir, "timeseries_input.png"), dpi = 150, bbox_inches = "tight")
    plt.close(fig1)
    print("[INFO] 图1已保存")

    # ---- Figure 2: operating params + output ----
    fig2, (ax_u, ax_cout) = plt.subplots(2, 1, figsize = (9, 5), sharex = True,
                                           gridspec_kw = {"hspace" : 0.18})

    for i, (col, lbl) in enumerate(zip(["U1_kV", "U2_kV", "U3_kV", "U4_kV"],
                                        [r"$U_1$", r"$U_2$", r"$U_3$", r"$U_4$"])) :
        ax_u.plot(ts_ds, df[col].values[::step], color = PALETTE[i], linewidth = 0.8, label = lbl)
    ax_u.set_ylabel(r"$U_i$ (kV)")
    ax_u.legend(loc = "upper right", ncol = 4, fontsize = 9, framealpha = 0.85)
    style_ax(ax_u)

    ax_cout.plot(ts_ds, df["C_out_mgNm3"].values[::step], color = PALETTE[3], linewidth = 0.8)
    ax_cout.axhline(y = 10, color = "red", linestyle = "--", linewidth = 0.9, alpha = 0.6, label = r"$C_{limit}=10$")
    ax_cout.axhline(y = 5, color = "darkred", linestyle = ":", linewidth = 0.9, alpha = 0.6, label = r"$C_{limit}'=5$")
    ax_cout.set_ylabel(r"$C_{out}$ (mg/Nm$^3$)")
    ax_cout.legend(loc = "center right", fontsize = 9, framealpha = 0.85)
    style_ax(ax_cout)

    for ax in [ax_u, ax_cout] :
        _paint_bands(ax, spans, PALETTE)
    _paint_labels(ax_u, spans, PALETTE)

    ax_cout.xaxis.set_major_locator(mdates.DayLocator())
    ax_cout.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "timeseries_output.png"), dpi = 150, bbox_inches = "tight")
    plt.close(fig2)
    print("[INFO] 图2已保存")


if __name__ == "__main__" :
    main()
