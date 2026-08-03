import sys, json, copy
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, r'D:\WorkSpace\Problem')
from data_loader.loader import load_raw, clean_and_impute
from modeling.deutsch import predict_cout
from modeling.power import fit_power_model, predict_power
from optim.regime import cluster_regimes

DATA_PATH = r'D:\WorkSpace\question\2026年XJTU校赛题目\2026年校赛题目\A\Cement_ESP_Data.csv'
df = clean_and_impute(load_raw(DATA_PATH))
power_model = fit_power_model(df)

# Use pre-fitted params from bayesian_ci.json
base_params = {
    'kA': [287.25, 287.25, 258.52, 258.52],
    'alpha': [0.0023, 0.0021, 0.001, 0.001],
    'T_ref': [233, 233, 445, 445],
    'r': 0.5,
    'kA_0': 287.25,
    'g': [1.0, 1.0, 0.9, 0.9],
}

bounds_info = {
    1: (45.9, 85.1, 157, 290),
    2: (45.4, 85.7, 154, 289),
    3: (38.0, 71.0, 359, 506),
    4: (37.0, 69.8, 358, 510),
}

regimes_df, _ = cluster_regimes(df, n_clusters=6)
regime_means = []
for rid in range(6):
    rd = regimes_df[regimes_df['regime'] == rid]
    regime_means.append({
        'Cin': rd['C_in_gNm3'].mean(),
        'Q': rd['Q_Nm3h'].mean()
    })

T_ref_sets = {
    'short(200/400)': [200, 200, 400, 400],
    'base(233/445)': [233, 233, 445, 445],
    'long(260/490)': [260, 260, 490, 490],
}

results = {}
for label, tref_list in T_ref_sets.items():
    params_copy = copy.deepcopy(base_params)
    params_copy['T_ref'] = tref_list

    regime_results = []
    for rid in range(6):
        rm = regime_means[rid]
        Cin, Q = rm['Cin'], rm['Q']

        def objective(x):
            U = x[:4].tolist()
            T = x[4:].tolist()
            return predict_power(power_model, U, T)

        def constraint(x):
            U = x[:4].tolist()
            T = x[4:].tolist()
            return 10.0 - predict_cout(params_copy, 0, Cin, Q, U, T)

        lb, ub, x0 = [], [], []
        for i in range(1, 5):
            umin, umax, tmin, tmax = bounds_info[i]
            lb.extend([umin, tmin])
            ub.extend([umax, tmax])
            x0.extend([(umin + umax) / 2, tref_list[i - 1]])

        best_P = 1e9
        best_x = None
        np.random.seed(42)
        for trial in range(10):
            if trial == 0:
                x_init = np.array(x0)
            else:
                x_init = np.array([np.random.uniform(lb[j], ub[j]) for j in range(8)])
            try:
                res = minimize(objective, x_init, method='SLSQP',
                             bounds=list(zip(lb, ub)),
                             constraints=[{'type': 'ineq', 'fun': constraint}],
                             options={'maxiter': 500, 'ftol': 1e-9})
                if res.success and constraint(res.x) >= -0.01 and res.fun < best_P:
                    best_P = res.fun
                    best_x = res.x
            except Exception:
                pass

        if best_x is not None:
            regime_results.append({
                'regime': rid, 'Cin': round(Cin, 2),
                'T': [int(round(best_x[4 + j])) for j in range(4)],
                'P': round(best_P, 1)
            })
    results[label] = regime_results

print('\n=== T_ref Sensitivity Results ===')
for label, rr in results.items():
    avg_P = np.mean([r['P'] for r in rr])
    print(f'\n{label}: avg_PB={avg_P:.1f}')
    for r in rr:
        print(f"  R{r['regime']}: T={r['T']}, P={r['P']}")

with open(r'D:\WorkSpace\Problem\outputs\tref_sensitivity.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print('\nSaved.')
