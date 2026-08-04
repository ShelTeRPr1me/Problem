import numpy as np
import pandas as pd
import pytest
from modeling.power import fit_power_model, predict_power


@pytest.fixture
def df_power() :
    np.random.seed(0)
    n = 500
    U = np.random.uniform([40, 40, 30, 30], [80, 80, 70, 70], size = (n, 4))
    T = np.random.uniform([150, 150, 350, 350], [300, 300, 520, 520], size = (n, 4))
    k_true = [0.1, 0.1, 0.05, 0.05]
    beta_true = [500.0, 500.0, 1000.0, 1000.0]
    c_true = 800.0
    P = sum(k_true[i] * U[:, i] ** 2 + beta_true[i] / T[:, i] for i in range(4)) + c_true + np.random.normal(0, 5, n)
    df = pd.DataFrame({
        "U1_kV": U[:, 0], "U2_kV": U[:, 1], "U3_kV": U[:, 2], "U4_kV": U[:, 3],
        "T1_s": T[:, 0], "T2_s": T[:, 1], "T3_s": T[:, 2], "T4_s": T[:, 3],
        "P_total_kW": P,
    })
    return df, k_true, c_true


def test_fit_extended_r2_high(df_power) :
    df, _, _ = df_power
    model = fit_power_model(df)
    assert model["extended"], "有截距应触发扩展模型"
    assert model["r2_ext"] > 0.9


def test_fit_k_nonneg(df_power) :
    df, _, _ = df_power
    model = fit_power_model(df)
    for k in model["k_ext"]:
        assert k >= 0, "物理约束 k_i>=0"


def test_fit_beta_positive(df_power) :
    df, _, _ = df_power
    model = fit_power_model(df)
    if model.get("has_strike"):
        for b in model["beta_ext"]:
            assert b >= 1.0, "物理约束 beta_i>=1 (振打电机必耗电)"


def test_predict_matches_fit(df_power) :
    df, _, _ = df_power
    model = fit_power_model(df)
    row = df.iloc[0]
    U = [row["U1_kV"], row["U2_kV"], row["U3_kV"], row["U4_kV"]]
    T = [row["T1_s"], row["T2_s"], row["T3_s"], row["T4_s"]]
    P_pred = predict_power(model, U, T = T)
    assert abs(P_pred - row["P_total_kW"]) < 50.0


def test_predict_extended_formula(df_power) :
    df, k_true, c_true = df_power
    model = fit_power_model(df)
    U = [60.0, 60.0, 50.0, 50.0]
    T = [200.0, 200.0, 400.0, 400.0]
    P = predict_power(model, U, T = T)
    expected = sum(model["k_ext"][i] * U[i] ** 2 + model["beta_ext"][i] / T[i] for i in range(4)) + model["c_ext"]
    assert abs(P - expected) < 1e-6
