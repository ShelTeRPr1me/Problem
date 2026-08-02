"""对全部6个工况分别计算性价比，生成条件性优先级规则"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yaml
from data_loader.loader import load_raw, clean_and_impute
from modeling.power import fit_power_model
from modeling.deutsch import fit_deutsch_params
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
    model = {"power": power_model, "deutsch": deutsch_model}

    regimes = cluster_regimes(df, k=cfg.get("regime_k", 5), seed=seed)
    sol_all = solve_all_regimes(regimes, model, bounds, C_limit=cfg.get("c_limit", 10.0),
                                multi_start=cfg.get("multi_start", 10), seed=seed)

    results = []
    for r in sol_all:
        regime = r["regime"]
        sol = r["sol"]
        if not sol or not sol.get("U"):
            continue
        x0 = sol["U"] + sol["T"]
        sens = numeric_jacobian(model, x0, regime, bounds, step_ratio=cfg.get("fd_step_ratio", 0.01))
        prio = priority_rule(sens)
        results.append({
            "regime_id": regime["id"],
            "C_in": regime["mean"]["C_in"],
            "Temp": regime["mean"]["Temp"],
            "avg_ratio_U": prio["avg_ratio_U"],
            "avg_ratio_T": prio["avg_ratio_T"],
            "priority": prio["priority"],
        })

    out_path = os.path.join(base, "outputs", "priority_all_regimes.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n=== 各工况性价比汇总 ===")
    for r in results:
        print(f"工况{r['regime_id']}: C_in={r['C_in']:.2f}, ratio_U={r['avg_ratio_U']:.2f}, "
              f"ratio_T={r['avg_ratio_T']:.2f}, 优先级={r['priority']}")

if __name__ == "__main__":
    main()