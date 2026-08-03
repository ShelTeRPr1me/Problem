"""kA0 和 beta 敏感性分析：参数扰动对最优电耗和优先级的影响"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yaml
from data_loader.loader import load_raw, clean_and_impute
from modeling.power import fit_power_model, predict_power
from modeling.deutsch import fit_deutsch_params, predict_cout
from optim.regime import cluster_regimes
from optim.solve import solve_all_regimes
from sensitivity.jacobian import numeric_jacobian
from sensitivity.priority import priority_rule

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base, "config/config.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed = cfg.get("seed", 42)
    np.random.seed(seed)
    csv_path = os.path.join(base, cfg["csv_path"]) if not os.path.isabs(cfg["csv_path"]) else cfg["csv_path"]
    df = load_raw(csv_path)
    df = clean_and_impute(df)
    bounds = df.attrs["bounds"]
    power_model = fit_power_model(df)
    deutsch_model = fit_deutsch_params(df, bounds)
    regimes = cluster_regimes(df, k=cfg.get("regime_k", 5), seed=seed)

    # === kA0 敏感性 ===
    kA0_base = deutsch_model["kA_0"]
    kA0_results = {}
    for factor in [0.8, 0.9, 1.0, 1.1, 1.2]:
        dm = dict(deutsch_model)
        dm["kA_0"] = kA0_base * factor
        g = dm.get("g", [1.0, 1.0, 0.9, 0.9])
        dm["kA"] = [dm["kA_0"] * gi for gi in g]
        model = {"power": power_model, "deutsch": dm}
        sol = solve_all_regimes(regimes, model, bounds, C_limit=10.0, multi_start=10, seed=seed)
        avg_P = np.mean([s["sol"]["P"] for s in sol if s["sol"] and s["sol"].get("P")])
        kA0_results[f"{factor:.1f}"] = {"kA0": dm["kA_0"], "avg_P": float(avg_P)}

    # === beta 敏感性（将 beta 放大 1x, 5x, 10x, 50x）===
    beta_base = power_model.get("beta", [0, 0, 0, 0])
    beta_results = {}
    for factor in [1, 5, 10, 50]:
        pm = dict(power_model)
        pm["beta"] = [b * factor for b in beta_base]
        model = {"power": pm, "deutsch": deutsch_model}
        sol = solve_all_regimes(regimes, model, bounds, C_limit=10.0, multi_start=10, seed=seed)
        avg_P = np.mean([s["sol"]["P"] for s in sol if s["sol"] and s["sol"].get("P")])
        # 计算工况0的优先级
        high_sol = next((s["sol"] for s in sol if s["regime"]["id"] == 0), None)
        prio = None
        if high_sol and high_sol.get("U"):
            x0 = high_sol["U"] + high_sol["T"]
            regime_obj = next(s["regime"] for s in sol if s["regime"]["id"] == 0)
            sens = numeric_jacobian(model, x0, regime_obj, bounds, step_ratio=0.01)
            prio = priority_rule(sens)
        beta_results[f"{factor}x"] = {
            "avg_P": float(avg_P),
            "ratio_U": prio["avg_ratio_U"] if prio else None,
            "ratio_T": prio["avg_ratio_T"] if prio else None,
            "priority": prio["priority"] if prio else None,
        }

    results = {"kA0_sensitivity": kA0_results, "beta_sensitivity": beta_results}
    out_path = os.path.join(base, "outputs", "param_sensitivity.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n=== kA0 敏感性 ===")
    for k, v in kA0_results.items():
        print(f"  kA0×{k}: kA0={v['kA0']:.1f}, avg_P={v['avg_P']:.1f}")
    print("\n=== beta 敏感性 ===")
    for k, v in beta_results.items():
        print(f"  beta×{k}: avg_P={v['avg_P']:.1f}, ratio_U={v['ratio_U']:.2f}, ratio_T={v['ratio_T']:.2f}, priority={v['priority']}")

if __name__ == "__main__":
    main()