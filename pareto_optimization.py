"""
多目标优化 Pareto 前沿 — 电耗 P vs 振打瞬时峰值 C_peak
约束法: min P s.t. C_out≤C_limit AND C_peak≤C_peak_max, 扫描 C_peak_max
对高浓度工况画 Pareto 前沿, 给决策者权衡曲线
"""
import sys
import os
import json
import numpy as np
from scipy.optimize import minimize
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


def solve_pareto_point(regime, model, bounds, C_limit, C_peak_max, seed = 42) :
    Tin = regime["mean"]["Temp"]
    Cin = regime["mean"]["C_in"]
    Q = regime["mean"]["Q"]
    params = model["deutsch"]
    power_model = model["power"]

    lb = [bounds[f"U{i}"][0] for i in range(1, 5)] + [bounds[f"T{i}"][0] for i in range(1, 5)]
    ub = [bounds[f"U{i}"][1] for i in range(1, 5)] + [min(bounds[f"T{i}"][1], bounds[f"T_crit{i}"]) for i in range(1, 5)]
    bounds_arr = list(zip(lb, ub))
    np.random.seed(seed)

    def objective(x) :
        return predict_power(power_model, x[:4], T = x[4:8])

    def cons_cout(x) :
        return C_limit - predict_cout(params, Tin, Cin, Q, x[:4], x[4:8])

    def cons_peak(x) :
        return C_peak_max - predict_peak(params, x[4:8], Cin)

    cons = [{"type": "ineq", "fun": cons_cout}, {"type": "ineq", "fun": cons_peak}]
    best = None
    for start in range(8) :
        x0 = np.array([(lb[i] + ub[i]) / 2 for i in range(8)]) if start == 0 else np.array([np.random.uniform(lb[i], ub[i]) for i in range(8)])
        try :
            res = minimize(objective, x0, method="SLSQP", bounds=bounds_arr, constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
            if res.success and cons_cout(res.x) >= -1e-6 and cons_peak(res.x) >= -1e-6 :
                P = predict_power(power_model, res.x[:4], T = res.x[4:8])
                if best is None or P < best["P"]:
                    best = {"P": float(P), "C_peak": float(predict_peak(params, res.x[4:8], Cin)),
                            "U": res.x[:4].tolist(), "T": res.x[4:8].tolist()}
        except Exception :
            pass
    return best


def main() :
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "config", "config.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    csv_path = os.path.join(base_dir, cfg["csv_path"])
    out_dir = os.path.join(base_dir, cfg.get("out_dir", "outputs"))
    seed = cfg.get("seed", 42)
    np.random.seed(seed)

    print("=" * 60)
    print("多目标优化 Pareto 前沿 (电耗 vs 振打峰值)")
    print("=" * 60)

    df = clean_and_impute(load_raw(csv_path))
    bounds = df.attrs["bounds"]
    power_model = fit_power_model(df)
    deutsch_model = fit_deutsch_params(df, bounds)
    model = {"power": power_model, "deutsch": deutsch_model}

    regimes = cluster_regimes(df, k=cfg.get("regime_k", 5), seed=seed)
    C_limit = cfg.get("c_limit", 10.0)
    high_regime = max(regimes["regimes"], key=lambda r: r["mean"]["C_in"])
    print(f"[INFO] 选取高浓度工况{high_regime['id']} (C_in={high_regime['mean']['C_in']:.2f})")

    Cin = high_regime["mean"]["C_in"]
    T_med = [df[f"T{i}_s"].median() for i in range(1, 5)]
    C_peak_min = predict_peak(deutsch_model, T_med, Cin)
    T_ext = [bounds[f"T{i}"][1] for i in range(1, 5)]
    C_peak_max_val = predict_peak(deutsch_model, T_ext, Cin)
    print(f"[INFO] C_peak 范围 : [{C_peak_min:.1f}, {C_peak_max_val:.1f}]")

    peak_grid = np.linspace(50, 300, 14)
    pareto = []
    for cp_max in peak_grid :
        pt = solve_pareto_point(high_regime, model, bounds, C_limit, cp_max, seed = seed)
        if pt :
            pareto.append({"C_peak_max": float(cp_max), "P": pt["P"], "C_peak": pt["C_peak"]})
            print(f"  C_peak_max={cp_max:.1f} -> P={pt['P']:.1f}, C_peak={pt['C_peak']:.1f}")

    fig, ax = plt.subplots(figsize = (10, 7))
    ps = [p["P"] for p in pareto]
    cks = [p["C_peak"] for p in pareto]
    ax.plot(cks, ps, "o-", color=COLOR, markersize=6, linewidth=2)
    ax.set_xlabel(r"振打瞬时峰值 $C_{peak}$ (mg/Nm$^3$)")
    ax.set_ylabel("总电耗 $P$ (kW)")
    ax.set_title(f"Pareto 前沿: 电耗 vs 振打峰值 (工况{high_regime['id']}, $C_{{in}}={Cin:.1f}$)")
    style_ax(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pareto_front.png"), dpi=150)
    plt.close()
    print(f"[INFO] Pareto 前沿图已保存至 {os.path.join(out_dir, 'pareto_front.png')}")

    with open(os.path.join(out_dir, "pareto_front.json"), "w", encoding="utf-8") as f:
        json.dump({"regime_id": high_regime["id"], "Cin": Cin, "pareto_points": pareto}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()