"""
补充分析: C_limit 扫描 + r 敏感性 + 振打同步性 + Sobol 可视化
"""
import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from report.plot_style import setup, style_ax, PALETTE, COLOR
setup()

import yaml
from data_loader.loader import load_raw, clean_and_impute
from modeling.power import fit_power_model, predict_power
from modeling.deutsch import fit_deutsch_params, predict_cout, predict_peak
from optim.regime import cluster_regimes
from optim.solve import solve_all_regimes
from scipy.optimize import minimize


def _load_all() :
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "config", "config.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    csv_path = os.path.join(base_dir, cfg["csv_path"])
    out_dir = os.path.join(base_dir, cfg.get("out_dir", "outputs"))
    seed = cfg.get("seed", 42)
    np.random.seed(seed)
    df = clean_and_impute(load_raw(csv_path))
    bounds = df.attrs["bounds"]
    power_model = fit_power_model(df)
    deutsch_model = fit_deutsch_params(df, bounds)
    model = {"power": power_model, "deutsch": deutsch_model}
    regimes = cluster_regimes(df, k=cfg.get("regime_k", 5), seed=seed)
    return df, bounds, model, regimes, out_dir, seed, cfg



def main() :
    df, bounds, model, regimes, out_dir, seed, cfg = _load_all()
    high_regime = max(regimes["regimes"], key=lambda r: r["mean"]["C_in"])
    params = model["deutsch"]
    power_model = model["power"]

    # ===== 1. C_limit 扫描 =====
    print("\n=== C_limit 扫描 ===")
    clims = np.arange(3, 16, 1.0)
    P_scan = []
    for cl in clims :
        sol = solve_all_regimes(regimes, model, bounds, C_limit = cl, multi_start = 5, seed = seed)
        avg_P = np.mean([s["sol"]["P"] for s in sol if s["sol"]["P"]])
        P_scan.append(avg_P)
        print(f"  C_limit = {cl:.0f} -> avg P = {avg_P:.1f}")

    fig, ax = plt.subplots(figsize = (8, 6))
    ax.plot(clims, P_scan, "o-", color=COLOR, markersize=6, linewidth=2)
    ax.axvline(10, color="red", linestyle="--", alpha=0.5, label="当前标准 10")
    ax.axvline(5, color="orange", linestyle="--", alpha=0.5, label="收紧标准 5")
    ax.set_xlabel(r"排放限值 $C_{limit}$ (mg/Nm$^3$)")
    ax.set_ylabel("平均最优电耗 $\\bar{P}^*$ (kW)")
    ax.set_title("电耗-排放限值权衡曲线")
    ax.legend(); ax.grid(True, alpha = 0.3); style_ax(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "climit_scan.png"), dpi=150)
    plt.close()

    # ===== 2. r 敏感性 =====
    print("\n=== r 敏感性测试 ===")
    r_values = [0.3, 0.5, 0.7, 1.0]
    r_results = {}
    for r_val in r_values :
        params_r = {**params, "r": r_val}
        model_r = {"power": power_model, "deutsch": params_r}
        sol = solve_all_regimes(regimes, model_r, bounds, C_limit = 10.0, multi_start = 5, seed = seed)
        avg_P = np.mean([s["sol"]["P"] for s in sol if s["sol"]["P"]])
        r_results[r_val] = avg_P
        print(f"  r = {r_val} -> avg P = {avg_P:.1f}")

    fig, ax = plt.subplots(figsize = (8, 6))
    ax.bar([str(r) for r in r_values], [r_results[r] for r in r_values], color = PALETTE[0])
    ax.set_xlabel("振打双向偏离比例 $r$")
    ax.set_ylabel("平均最优电耗 $\\bar{P}^*$ (kW)")
    ax.set_title("$r$ 敏感性测试")
    style_ax(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "r_sensitivity.png"), dpi=150)
    plt.close()

    # ===== 3. 振打同步性 =====
    print("\n=== 振打同步性分析 ===")
    T = np.column_stack([df[f"T{i}_s"].values for i in range(1, 5)])
    T_mean_row = T.mean(axis = 1)
    cv = T.std(axis = 1) / T_mean_row
    ratios_12 = df["T1_s"].values / df["T2_s"].values
    ratios_34 = df["T3_s"].values / df["T4_s"].values
    print(f"  T 行内 CV : mean = {cv.mean():.4f}, std = {cv.std():.4f}")
    print(f"  T1/T2 比值 : mean = {ratios_12.mean():.4f}, std = {ratios_12.std():.4f}")
    print(f"  T3/T4 比值 : mean = {ratios_34.mean():.4f}, std = {ratios_34.std():.4f}")

    Cin_med = df["C_in_gNm3"].median()
    T_med = [df[f"T{i}_s"].median() for i in range(1, 5)]
    peak_sync = predict_peak(params, T_med, Cin_med)
    T_stagger = [T_med[0], T_med[1] * 1.15, T_med[2] * 0.85, T_med[3] * 1.1]
    peak_stagger = predict_peak(params, T_stagger, Cin_med)
    print(f"  同步振打峰值 : {peak_sync:.1f}")
    print(f"  错峰振打峰值 : {peak_stagger:.1f}")

    fig, axes = plt.subplots(1, 2, figsize = (14, 6))
    axes[0].hist(ratios_12, bins=50, color=PALETTE[0], alpha=0.7, label="$T_1/T_2$")
    axes[0].hist(ratios_34, bins=50, color=PALETTE[1], alpha=0.7, label="$T_3/T_4$")
    axes[0].set_xlabel("振打周期比值"); axes[0].set_ylabel("频次")
    axes[0].set_title("振打周期比值分布（1.0=同步）"); axes[0].legend(); style_ax(axes[0])
    axes[1].bar(["同步振打", "错峰振打"], [peak_sync, peak_stagger], color=[PALETTE[3], PALETTE[2]])
    axes[1].set_ylabel(r"$C_{peak}$ (mg/Nm$^3$)")
    axes[1].set_title("同步 vs 错峰振打峰值对比"); style_ax(axes[1])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rapping_sync.png"), dpi=150)
    plt.close()

    # ===== 4. Sobol 可视化 =====
    sobol_path = os.path.join(out_dir, "sobol_indices.json")
    if os.path.exists(sobol_path) :
        with open(sobol_path, "r", encoding="utf-8") as f:
            sob = json.load(f)
        names = sob["names"]
        fig, axes = plt.subplots(1, 2, figsize = (14, 6))
        x = np.arange(len(names))
        w = 0.35
        for ax, key, title in [(axes[0], "cout", "$C_{out}$ Sobol"), (axes[1], "power", "$P$ Sobol")]:
            S1 = sob[key]["S1"]; ST = sob[key]["ST"]
            inter = [ST[i] - S1[i] for i in range(len(S1))]
            ax.bar(x - w, S1, w, label="一阶 $S_i$", color=PALETTE[0])
            ax.bar(x, inter, w, label="交互 $S_{Ti}-S_i$", color=PALETTE[1])
            ax.bar(x + w, ST, w, label="总效应 $S_{Ti}$", color=PALETTE[2])
            ax.set_xticks(x); ax.set_xticklabels(names)
            ax.set_ylabel("Sobol 指数"); ax.set_title(title); ax.legend(); style_ax(ax)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "sobol_barplot.png"), dpi=150)
        plt.close()
        print("\n[INFO] Sobol 条形图已生成")

    result = {
        "climit_scan": {"C_limits": clims.tolist(), "avg_P": P_scan},
        "r_sensitivity": r_results,
        "rapping_sync": {"cv_mean": float(cv.mean()), "ratio_12_mean": float(ratios_12.mean()),
                          "peak_sync": float(peak_sync), "peak_stagger": float(peak_stagger)},
    }
    with open(os.path.join(out_dir, "additional_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent = 2, ensure_ascii = False)
    print(f"\n[INFO] 补充分析完成, 保存至 {out_dir}")



if __name__ == "__main__":
    main()