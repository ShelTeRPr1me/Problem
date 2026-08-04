import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from report.plot_style import setup, style_ax, PALETTE, COLOR

setup()


def plot_relation_curves(model, df, out_dir) :
    params = model["deutsch"]
    os.makedirs(out_dir, exist_ok = True)

    fig, axes = plt.subplots(2, 2, figsize = (14, 10))
    Tin_med = df["Temp_C"].median()
    Cin_med = df["C_in_gNm3"].median()
    Q_med = df["Q_Nm3h"].median()
    U_med = [df[f"U{i}_kV"].median() for i in range(1, 5)]
    T_med = [df[f"T{i}_s"].median() for i in range(1, 5)]

    from modeling.deutsch import predict_cout

    Ts = np.linspace(df["Temp_C"].min(), df["Temp_C"].max(), 50)
    axes[0, 0].plot(Ts, [predict_cout(params, t, Cin_med, Q_med, U_med, T_med) for t in Ts], color = COLOR)
    axes[0, 0].set_xlabel("入口温度 (℃)")
    axes[0, 0].set_ylabel(r"$C_{out}$ (mg/Nm$^3$)")
    axes[0, 0].set_title("温度-$C_{out}$ 关系")
    style_ax(axes[0, 0])

    Cs = np.linspace(df["C_in_gNm3"].min(), df["C_in_gNm3"].max(), 50)
    axes[0, 1].plot(Cs, [predict_cout(params, Tin_med, c, Q_med, U_med, T_med) for c in Cs], color = COLOR)
    axes[0, 1].set_xlabel(r"入口浓度 (g/Nm$^3$)")
    axes[0, 1].set_ylabel(r"$C_{out}$ (mg/Nm$^3$)")
    axes[0, 1].set_title("$C_{in}$-$C_{out}$ 关系")
    style_ax(axes[0, 1])

    Us = np.linspace(df["U1_kV"].min(), df["U1_kV"].max(), 50)
    axes[1, 0].plot(Us, [predict_cout(params, Tin_med, Cin_med, Q_med, [u] + U_med[1:], T_med) for u in Us], color = COLOR)
    axes[1, 0].set_xlabel("$U_1$ 电压 (kV)")
    axes[1, 0].set_ylabel(r"$C_{out}$ (mg/Nm$^3$)")
    axes[1, 0].set_title("电压-$C_{out}$ 关系")
    style_ax(axes[1, 0])

    from modeling.deutsch import predict_peak
    T1s = np.linspace(df["T1_s"].min(), df["T1_s"].max(), 50)
    axes[1, 1].plot(T1s, [predict_peak(params, [t] + T_med[1:], Cin_med) for t in T1s], color = COLOR)
    axes[1, 1].set_xlabel("$T_1$ 振打周期 (s)")
    axes[1, 1].set_ylabel(r"$C_{peak}$ (mg/Nm$^3$)")
    axes[1, 1].set_title("振打周期-$C_{peak}$ 关系")
    style_ax(axes[1, 1])

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "relation_curves.png"), dpi=150)
    plt.close()


def plot_regime_scatter(regimes, df, out_dir) :
    os.makedirs(out_dir, exist_ok = True)
    fig, ax = plt.subplots(figsize = (8, 6))
    labels = np.array(regimes["labels"])
    scatter = ax.scatter(df["C_in_gNm3"], df["Temp_C"], c=labels, cmap="viridis", alpha=0.6, s=10)
    plt.colorbar(scatter, ax=ax, label="工况编号")
    ax.set_xlabel(r"入口浓度 (g/Nm$^3$)")
    ax.set_ylabel(r"入口温度 (℃)")
    ax.set_title("K-Means工况划分散点图")
    style_ax(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "regime_scatter.png"), dpi=150)
    plt.close()


def plot_param_compare(cmp, out_dir) :
    os.makedirs(out_dir, exist_ok = True)
    rA, rB = cmp["regime_A"], cmp["regime_B"]
    x = np.arange(4)
    w = 0.35

    # 图A: 电压与振打周期对比 (2子图上下排列)
    fig, axes = plt.subplots(1, 2, figsize = (14, 6))
    axes[0].bar(x - w/2, rA["U"], w, label=f"工况{rA['id']}(高浓度)", color=PALETTE[0])
    axes[0].bar(x + w/2, rB["U"], w, label=f"工况{rB['id']}(低浓度)", color=PALETTE[1])
    axes[0].set_xticks(x); axes[0].set_xticklabels(["$U_1$", "$U_2$", "$U_3$", "$U_4$"])
    axes[0].set_ylabel("电压 (kV)"); axes[0].set_title("电压对比"); axes[0].legend()
    style_ax(axes[0])

    axes[1].bar(x - w/2, rA["T"], w, label=f"工况{rA['id']}", color=PALETTE[0])
    axes[1].bar(x + w/2, rB["T"], w, label=f"工况{rB['id']}", color=PALETTE[1])
    axes[1].set_xticks(x); axes[1].set_xticklabels(["$T_1$", "$T_2$", "$T_3$", "$T_4$"])
    axes[1].set_ylabel("振打周期 (s)"); axes[1].set_title("振打周期对比"); axes[1].legend()
    style_ax(axes[1])

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "param_compare.png"), dpi=150)
    plt.close()

    # 图B: 电耗对比 (单独一张)
    fig, ax = plt.subplots(figsize = (7, 6))
    ax.bar([f"工况{rA['id']}(高浓度)", f"工况{rB['id']}(低浓度)"], [rA["P"], rB["P"]], color=[PALETTE[0], PALETTE[1]], width=0.5)
    ax.set_ylabel("总电耗 (kW)"); ax.set_title("电耗对比")
    style_ax(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "power_compare.png"), dpi=150)
    plt.close()


def plot_sensitivity_heatmap(sens, out_dir) :
    os.makedirs(out_dir, exist_ok = True)

    if "EC_U" in sens:
        SC = np.array([sens["EC_U"], sens["EC_T"]])
        SP = np.array([sens["EP_U"], sens["EP_T"]])
        lbl_C, lbl_P, lbl_R = r"浓度弹性 $E^C$", r"电耗弹性 $E^P$", r"性价比 $|E^C/E^P|$"
    else :
        SC = np.array([sens["SC_U"], sens["SC_T"]])
        SP = np.array([sens["SP_U"], sens["SP_T"]])
        lbl_C, lbl_P, lbl_R = r"浓度灵敏度 $S^C$", r"电耗灵敏度 $S^P$", r"性价比 $|S^C/S^P|$"

    # 图A: 浓度弹性/灵敏度
    fig, ax = plt.subplots(figsize = (6, 5))
    im0 = ax.imshow(SC, cmap="coolwarm", aspect="auto")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["U", "T"])
    ax.set_xticks(range(4)); ax.set_xticklabels(["1", "2", "3", "4"])
    ax.set_xlabel("电场编号")
    ax.set_title(lbl_C); plt.colorbar(im0, ax=ax, label="弹性值"); style_ax(ax, grid=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sens_elastic_C.png"), dpi=150)
    plt.close()

    # 图B: 电耗弹性/灵敏度
    fig, ax = plt.subplots(figsize = (6, 5))
    im1 = ax.imshow(SP, cmap="coolwarm", aspect="auto")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["U", "T"])
    ax.set_xticks(range(4)); ax.set_xticklabels(["1", "2", "3", "4"])
    ax.set_xlabel("电场编号")
    ax.set_title(lbl_P); plt.colorbar(im1, ax=ax, label="弹性值"); style_ax(ax, grid=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sens_elastic_P.png"), dpi=150)
    plt.close()

    # 图C: 性价比
    ratio = np.abs(SC) / (np.abs(SP) + 1e-12)
    fig, ax = plt.subplots(figsize = (6, 5))
    im2 = ax.imshow(ratio, cmap="YlOrRd", aspect="auto")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["U", "T"])
    ax.set_xticks(range(4)); ax.set_xticklabels(["1", "2", "3", "4"])
    ax.set_xlabel("电场编号")
    ax.set_title(lbl_R); plt.colorbar(im2, ax=ax, label="比值"); style_ax(ax, grid=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sens_ratio.png"), dpi=150)
    plt.close()


def plot_delta_power(dp, out_dir) :
    os.makedirs(out_dir, exist_ok = True)
    deltas = dp["deltas"]
    valid = [d for d in deltas if d["delta_pct"] is not None]

    ids = [d["regime_id"] for d in valid]
    p10 = [d["P10"] for d in valid]
    p5 = [d["P5"] for d in valid]
    x = np.arange(len(ids))
    w = 0.35

    # 图A: 收紧前后电耗对比
    fig, ax = plt.subplots(figsize = (8, 6))
    ax.bar(x - w/2, p10, w, label="$P^*(10)$", color=PALETTE[0])
    ax.bar(x + w/2, p5, w, label="$P^*(5)$", color=PALETTE[3])
    ax.set_xticks(x); ax.set_xticklabels([f"工况{i}" for i in ids])
    ax.set_ylabel("电耗 (kW)"); ax.set_title("收紧前后电耗对比"); ax.legend()
    style_ax(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "delta_power.png"), dpi=150)
    plt.close()

    # 图B: 电耗增幅
    dpct = [d["delta_pct"] for d in valid]
    fig, ax = plt.subplots(figsize = (8, 6))
    ax.bar(x, dpct, color = PALETTE[2])
    ax.set_xticks(x); ax.set_xticklabels([f"工况{i}" for i in ids])
    ax.set_ylabel("电耗增幅 (%)"); ax.set_title("$\\Delta P$ 增幅")
    style_ax(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "delta_pct.png"), dpi=150)
    plt.close()


def plot_relation_3d(model, df, out_dir) :
    from mpl_toolkits.mplot3d import Axes3D
    from modeling.deutsch import predict_cout
    params = model["deutsch"]
    os.makedirs(out_dir, exist_ok = True)

    Tin_med = df["Temp_C"].median()
    Cin_med = df["C_in_gNm3"].median()
    Q_med = df["Q_Nm3h"].median()
    U_med = [df[f"U{i}_kV"].median() for i in range(1, 5)]
    T_med = [df[f"T{i}_s"].median() for i in range(1, 5)]

    def surf(ax, xs, ys, f, xl, yl, title) :
        Xg, Yg = np.meshgrid(xs, ys)
        Z = np.zeros_like(Xg)
        for i in range(len(xs)) :
            for j in range(len(ys)) :
                Z[j, i] = f(xs[i], ys[j])
        ax.plot_surface(Xg, Yg, Z, cmap="viridis", alpha=0.85, edgecolor="none")
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_zlabel(r"$C_{out}$ (mg/Nm$^3$)")
        ax.set_title(title)
        ax.tick_params(colors = COLOR)
        ax.xaxis.label.set_color(COLOR); ax.yaxis.label.set_color(COLOR)
        ax.zaxis.label.set_color(COLOR); ax.title.set_color(COLOR)
        ax.xaxis.set_pane_color((0.95, 0.95, 0.95, 1))
        ax.yaxis.set_pane_color((0.95, 0.95, 0.95, 1))
        ax.zaxis.set_pane_color((0.95, 0.95, 0.95, 1))

    n = 30
    fig = plt.figure(figsize = (16, 14))

    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    U1s = np.linspace(df["U1_kV"].min(), df["U1_kV"].max(), n)
    U2s = np.linspace(df["U2_kV"].min(), df["U2_kV"].max(), n)
    surf(ax1, U1s, U2s,
         lambda u1, u2 : predict_cout(params, Tin_med, Cin_med, Q_med, [u1, u2, U_med[2], U_med[3]], T_med),
         "U1 (kV)", "U2 (kV)", "C_out vs 前两电场电压")

    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    T1s = np.linspace(df["T1_s"].min(), df["T1_s"].max(), n)
    T2s = np.linspace(df["T2_s"].min(), df["T2_s"].max(), n)
    surf(ax2, T1s, T2s,
         lambda t1, t2 : predict_cout(params, Tin_med, Cin_med, Q_med, U_med, [t1, t2, T_med[2], T_med[3]]),
         "T1 (s)", "T2 (s)", "C_out vs 前两电场振打周期")

    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    Cis = np.linspace(df["C_in_gNm3"].min(), df["C_in_gNm3"].max(), n)
    Qs = np.linspace(df["Q_Nm3h"].min(), df["Q_Nm3h"].max(), n)
    surf(ax3, Cis, Qs,
         lambda ci, q : predict_cout(params, Tin_med, ci, q, U_med, T_med),
         r"C_in (g/Nm$^3$)", r"Q (Nm$^3$/h)", "C_out vs 入口浓度与流量")

    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    surf(ax4, U1s, T1s,
         lambda u1, t1 : predict_cout(params, Tin_med, Cin_med, Q_med, [u1, U_med[1], U_med[2], U_med[3]], [t1, T_med[1], T_med[2], T_med[3]]),
         "U1 (kV)", "T1 (s)", "C_out vs 第1电场电压与振打")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "relation_3d.png"), dpi=150)
    plt.close()