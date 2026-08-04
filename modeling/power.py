import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear


def fit_power_model(df) :
    U = np.column_stack([df[f"U{i}_kV"].values ** 2 for i in range(1, 5)])
    P = df["P_total_kW"].values
    k, residuals, rank, sv = np.linalg.lstsq(U, P, rcond = None)
    P_pred = U @ k
    ss_res = np.sum((P - P_pred) ** 2)
    ss_tot = np.sum((P - P.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    cond = np.linalg.cond(U)
    k = np.maximum(k, 0)
    print(f"[INFO] 电耗模型 P = Σk_i*U_i^2 : k = {k}, R2 = {r2:.4f}, cond = {cond:.2f}")

    if r2 < 0.9 :
        print("[WARN] R2<0.9, 尝试扩展模型 P=Σk_i*U_i^2+Σβ_i/T_i+c, 约束 k_i,β_i>=0")
        print("  (物理: 电场能量∝U², 振打电机功耗∝频率=1/T, c吸收固定损耗)")
        X_ext = np.column_stack(
            [df[f"U{i}_kV"].values ** 2 for i in range(1, 5)] +
            [1.0 / df[f"T{i}_s"].values for i in range(1, 5)] +
            [np.ones(len(df))]
        )
        # k_i(前4) >= 0, β_i(中4) >= 1 (振打电机必耗电, 工程下界), c 自由
        lo = np.concatenate([np.zeros(4), np.ones(4), -np.inf * np.ones(1)])
        hi = np.inf * np.ones(9)
        res = lsq_linear(X_ext, P, bounds=(lo, hi), method="trf")
        coef9 = res.x
        P_pred2 = X_ext @ coef9
        r2_ext = 1 - np.sum((P - P_pred2) ** 2) / ss_tot
        k_ext = coef9[:4].tolist()
        beta_ext = coef9[4:8].tolist()
        c_ext = float(coef9[8])
        print(f"[INFO] 扩展模型(含振打, P = ΣkU²+Σβ/T+c) R2 = {r2_ext:.4f}")
        print(f"  k = {k_ext}, beta = {beta_ext}, c = {c_ext:.2f}")
        return {
            "k": k.tolist(), "r2": float(r2), "cond": float(cond),
            "extended": True, "has_strike": True,
            "k_ext": k_ext, "beta_ext": beta_ext, "c_ext": c_ext,
            "r2_ext": float(r2_ext),
        }
    return {"k": k.tolist(), "r2": float(r2), "cond": float(cond), "extended": False}


def predict_power(model, U, T = None) :
    if isinstance(model, dict) :
        if model.get("has_strike", False) and T is not None:
            k_ext = np.array(model["k_ext"])
            beta_ext = np.array(model["beta_ext"])
            c_ext = model["c_ext"]
            return float(sum(k_ext[i] * U[i] ** 2 + beta_ext[i] / T[i] for i in range(4)) + c_ext)
        if model.get("extended", False):
            k_ext = np.array(model["k_ext"])
            c_ext = model["c_ext"]
            beta_ext = np.array(model.get("beta_ext", [0.0] * 4))
            strike = sum(beta_ext[i] / T[i] for i in range(4)) if T is not None else 0.0
            return float(sum(k_ext[i] * U[i] ** 2 for i in range(4)) + c_ext + strike)
        k = model["k"]
    else :
        k = model
    return sum(k[i] * U[i] ** 2 for i in range(4))
