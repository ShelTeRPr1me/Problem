import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.dates as mdates

from data_loader.loader import load_raw, clean_and_impute
from optim.regime import cluster_regimes
from report.plot_style import setup, style_ax, PALETTE, COLOR

import yaml

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

    ts = df["timestamp"].values
    ts_pd = pd.to_datetime(ts)

    fig, axes = plt.subplots(5, 1, figsize = (16, 14), sharex = True)

    ax_cin = axes[0]
    ax_temp = axes[1]
    ax_q = axes[2]
    ax_u = axes[3]
    ax_cout = axes[4]

    ax_cin.plot(ts_pd, df["C_in_gNm3"], color = PALETTE[0], linewidth = 0.4, alpha = 0.8)
    ax_cin.set_ylabel(r"$C_{in}$ (g/Nm$^3$)")
    style_ax(ax_cin)

    ax_temp.plot(ts_pd, df["Temp_C"], color = PALETTE[1], linewidth = 0.4, alpha = 0.8)
    ax_temp.set_ylabel(r"$T_{in}$ (\u2103)")
    style_ax(ax_temp)

    ax_q.plot(ts_pd, df["Q_Nm3h"] / 1000, color = PALETTE[2], linewidth = 0.4, alpha = 0.8)
    ax_q.set_ylabel(r"$Q$ ($\times10^3$ Nm$^3$/h)")
    style_ax(ax_q)

    for i, (col, lbl) in enumerate(zip(["U1_kV", "U2_kV", "U3_kV", "U4_kV"], [r"$U_1$", r"$U_2$", r"$U_3$", r"$U_4$"])) :
        ax_u.plot(ts_pd, df[col], color = PALETTE[i], linewidth = 0.4, alpha = 0.7, label = lbl)
    ax_u.set_ylabel(r"$U_i$ (kV)")
    ax_u.legend(loc = "upper right", ncol = 4, fontsize = 10)
    style_ax(ax_u)

    ax_cout.plot(ts_pd, df["C_out_mgNm3"], color = PALETTE[3], linewidth = 0.4, alpha = 0.8)
    ax_cout.axhline(y = 10, color = "red", linestyle = "--", linewidth = 1.0, alpha = 0.7, label = r"$C_{limit}=10$")
    ax_cout.axhline(y = 5, color = "darkred", linestyle = ":", linewidth = 1.0, alpha = 0.7, label = r"$C_{limit}'=5$")
    ax_cout.set_ylabel(r"$C_{out}$ (mg/Nm$^3$)")
    ax_cout.set_xlabel("")
    ax_cout.legend(loc = "upper right", fontsize = 10)
    style_ax(ax_cout)

    for ax in axes :
        prev_lbl = labels[0]
        for j in range(1, len(labels)) :
            if labels[j] != prev_lbl :
                ax.axvline(x = ts_pd[j], color = PALETTE[int(labels[j])], linewidth = 0.3, alpha = 0.15)
                prev_lbl = labels[j]

    regime_colors = {i : PALETTE[i % len(PALETTE)] for i in range(k)}
    for i in range(k) :
        mask = labels == i
        t_start = ts_pd[mask].min()
        t_end = ts_pd[mask].max()
        y_top = ax_cin.get_ylim()[1]
        ax_cin.fill_between(
            [t_start, t_end], y_top, y_top * 1.02,
            color = regime_colors[i], alpha = 0.5
        )
        mid_t = t_start + (t_end - t_start) / 2
        ax_cin.text(mid_t, y_top * 1.01, f"R{i}", ha = "center", va = "center",
                    fontsize = 8, color = COLOR, fontweight = "bold")

    axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes[-1].xaxis.set_minor_locator(mdates.HourLocator(byhour = [6, 12, 18]))

    fig.suptitle("7\u5929\u8fd0\u884c\u65f6\u5e8f\u4e0e\u5de5\u51b5\u5207\u6362\u6807\u6ce8", fontsize = 18, color = COLOR, y = 0.98)
    fig.tight_layout(rect = [0, 0, 1, 0.96])

    out_path = os.path.join(out_dir, "timeseries_7day.png")
    fig.savefig(out_path, dpi = 150, bbox_inches = "tight")
    plt.close(fig)
    print(f"[INFO] 7\u5929\u65f6\u5e8f\u56fe\u5df2\u4fdd\u5b58: {out_path}")


if __name__ == "__main__" :
    main()