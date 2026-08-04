import numpy as np


def priority_rule(sens, wear_factor = 2.5) :
    # 优先用无量纲弹性系数 E=∂ln y/∂ln x, 消除电压(kV)与振打(s)量纲差异
    # 性价比 = |E^C / E^P| = |(∂ln C)/(∂ln P)|, 单位统一为 kW/(mg/Nm³), 可直接比较
    # 振打隐性成本修正: 振打除电耗外还有设备磨损/维护等隐性成本, 用 wear_factor 放大 EP_T
    # wear_factor=1 为纯电耗模型; wear_factor=2.5 表示隐性成本使振打实际代价约为电耗的2.5倍
    # 工程依据: 振打锤臂疲劳寿命~10^6次, 频繁振打缩短寿命→维修停机成本远超电耗增量
    if "EC_U" in sens:
        EC_U = np.array(sens["EC_U"]); EC_T = np.array(sens["EC_T"])
        EP_U = np.array(sens["EP_U"]); EP_T = np.array(sens["EP_T"])
        kind = "弹性系数"
    else :
        EC_U = np.array(sens["SC_U"]); EC_T = np.array(sens["SC_T"])
        EP_U = np.array(sens["SP_U"]); EP_T = np.array(sens["SP_T"])
        kind = "原始灵敏度(量纲未统一)"

    # 纯电耗性价比(不含隐性成本)
    eps_p = 1e-6
    ratio_U_raw = np.where(np.abs(EP_U) > eps_p, np.abs(EC_U) / np.abs(EP_U), np.inf)
    ratio_T_raw = np.where(np.abs(EP_T) > eps_p, np.abs(EC_T) / np.abs(EP_T), np.inf)
    avg_ratio_U = float(np.mean(ratio_U_raw[np.isfinite(ratio_U_raw)])) if np.any(np.isfinite(ratio_U_raw)) else 0.0
    avg_ratio_T = float(np.mean(ratio_T_raw[np.isfinite(ratio_T_raw)])) if np.any(np.isfinite(ratio_T_raw)) else 0.0

    # 含隐性成本修正的性价比: 振打实际代价 = wear_factor × |EP_T|
    EP_T_eff = EP_T * wear_factor
    ratio_U = ratio_U_raw.copy()
    ratio_T = np.where(np.abs(EP_T_eff) > eps_p, np.abs(EC_T) / np.abs(EP_T_eff), np.inf)
    avg_ratio_T_wear = float(np.mean(ratio_T[np.isfinite(ratio_T)])) if np.any(np.isfinite(ratio_T)) else 0.0

    free_T = np.abs(EP_T_eff) <= eps_p
    free_U = np.abs(EP_U) <= eps_p
    n_free_T = int(np.sum(free_T))
    n_free_U = int(np.sum(free_U))

    # 工程优先级: 基于含隐性成本的性价比
    if n_free_T > 0 and np.any(np.abs(EC_T[free_T]) > eps_p) :
        priority = "优先调振打"
        reason = f"振打对电耗无影响(∂P/∂T≈0)且能降排放, 视为免费手段 ({kind})"
    elif n_free_U > 0 and np.any(np.abs(EC_U[free_U]) > eps_p) :
        priority = "优先调电压"
        reason = f"电压对电耗无影响(∂P/∂U≈0)且能降排放, 视为免费手段 ({kind})"
    elif avg_ratio_U > avg_ratio_T_wear :
        priority = "优先调电压"
        reason = (f"含隐性成本修正后电压性价比 {avg_ratio_U:.2f} > "
                  f"振打性价比 {avg_ratio_T_wear:.2f} (wear={wear_factor}) ({kind})")
    else :
        priority = "优先调振打"
        reason = (f"含隐性成本修正后振打性价比 {avg_ratio_T_wear:.2f} >= "
                  f"电压性价比 {avg_ratio_U:.2f} (wear={wear_factor}) ({kind})")

    print(f"[INFO] 纯电耗性价比: 电压={avg_ratio_U:.2f}, 振打={avg_ratio_T:.2f}")
    print(f"[INFO] 含隐性成本(wear={wear_factor}): 振打={avg_ratio_T_wear:.2f}")
    print(f"[INFO] 优先级判定: {priority} ({reason})")
    return {
        "ratio_U": ratio_U_raw.tolist(), "ratio_T": ratio_T_raw.tolist(),
        "avg_ratio_U": avg_ratio_U, "avg_ratio_T": avg_ratio_T,
        "avg_ratio_T_wear": avg_ratio_T_wear, "wear_factor": wear_factor,
        "priority": priority, "reason": reason, "kind": kind,
        "n_free_T": n_free_T, "n_free_U": n_free_U,
    }