# -*- coding: utf-8 -*-
"""
01_scorer_diagnostics.py (终极全量清零版):
落实 A3: 融入 100mm 上下文 ctx 特征 + 加入非线性 GBDT 探针，彻底排除"探针太弱"质疑
[修复]: 彻底解决 PCA 与 StandardScaler 在测试集上的 Data Leakage，并增加患者级 Bootstrap 计算增量 CI。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline

from rcrc_core import (load_cohort, compute_patient_weights, calculate_risk, 
                       compute_crc_lambda, child_rngs, patient_bootstrap_ci, 
                       RESULT_DIR, ALPHAS)


def r2_score_emp(y_true, y_pred):
    return 1.0 - np.mean((y_true - y_pred) ** 2) / max(np.var(y_true), 1e-12)


def make_gbdt_probe():
    """构建严格防泄漏的非线性特征流水线"""
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=32, random_state=0),
        HistGradientBoostingRegressor(max_iter=100, max_depth=3, l2_regularization=10.0, random_state=0)
    )


def run_scorer_diagnostics():
    print("\n" + "=" * 80)
    print(">>> 正在运行 01: 探针终极清零 (加入 ctx 上下文特征 + 非线性 GBDT 探针)")
    print("=" * 80)
    
    df, _, _, feats_foc, _ = load_cohort()
    
    # 调取 ctx 上下文特征，融合为 8192 维全景表征
    npz = np.load(os.path.join(RESULT_DIR, "feat_lidc_full.npz"))
    feats_ctx = npz["ctx"]
    feats_all = np.c_[feats_foc, feats_ctx]  # (1378, 8192)
    print(f"[*] 成功融合 foc (50mm) 与 ctx (100mm) 特征矩阵，维度: {feats_all.shape}")

    pid = df.pid.values
    diam = df.diam.values
    q = df.q.values
    ld = np.log(diam)
    n = len(q)

    # 10 折患者 GroupKFold OOF 交叉评测
    p_dia = np.zeros(n)
    p_hyb_linear = np.zeros(n)
    p_hyb_gbdt = np.zeros(n)
    p_fm = np.zeros(n)
    p_ld = np.zeros(n)

    gkf = GroupKFold(n_splits=10)
    print("[*] 正在执行 10 折 OOF (无泄漏流水线验证)...")
    
    for tr, te in gkf.split(ld, q, groups=pid):
        # 1) 物理直径基准
        base = make_pipeline(SplineTransformer(n_knots=5, degree=3), LinearRegression()).fit(ld[tr, None], q[tr])
        p_dia[te] = base.predict(ld[te, None])
        res_tr = q[tr] - base.predict(ld[tr, None])

        # 2) 线性探针 (PCA-64 + RidgeCV on foc+ctx)
        probe_lin = make_pipeline(StandardScaler(), PCA(n_components=64, random_state=0), 
                                  RidgeCV(alphas=np.logspace(-1, 4, 12))).fit(feats_all[tr], res_tr)
        p_hyb_linear[te] = p_dia[te] + probe_lin.predict(feats_all[te])

        # 3) 非线性探针 (无泄漏版)
        probe_gbdt = make_gbdt_probe().fit(feats_all[tr], res_tr)
        p_hyb_gbdt[te] = p_dia[te] + probe_gbdt.predict(feats_all[te])

        # 4) 纯 FMCIB 探针与解码直径
        probe_pure = make_pipeline(StandardScaler(), PCA(n_components=64, random_state=0), 
                                   RidgeCV(alphas=np.logspace(-1, 4, 12)))
        p_fm[te] = probe_pure.fit(feats_all[tr], q[tr]).predict(feats_all[te])
        p_ld[te] = probe_pure.fit(feats_all[tr], ld[tr]).predict(feats_all[te])

    r2_dia = r2_score_emp(q, p_dia)
    r2_hyb_lin = r2_score_emp(q, p_hyb_linear)
    r2_hyb_gbdt = r2_score_emp(q, p_hyb_gbdt)

    print(f"  -> 纯物理直径 OOF R^2:            {r2_dia:.4f}")
    print(f"  -> 线性残差混合 (foc+ctx) OOF R^2: {r2_hyb_lin:.4f} (增量 Delta R^2: {r2_hyb_lin - r2_dia:+.4f})")
    print(f"  -> 非线性 GBDT (foc+ctx) OOF R^2:  {r2_hyb_gbdt:.4f} (增量 Delta R^2: {r2_hyb_gbdt - r2_dia:+.4f})")

    # 40/30/30 独立切分保形直签率对比
    print("\n[*] 正在运行 100 次独立三分法，评估非线性 GBDT 混合模型的真实吞吐量...")
    w = compute_patient_weights(df, q)
    rng_list = child_rngs(2025, 100)
    sign_dia, sign_gbdt = [], []

    for rng in rng_list:
        up = np.unique(pid)
        tr_p = rng.choice(up, int(len(up) * 0.4), replace=False)
        rem_p = np.setdiff1d(up, tr_p)
        cal_p = rng.choice(rem_p, int(len(rem_p) * 0.5), replace=False)
        te_p = np.setdiff1d(rem_p, cal_p)

        m_tr, m_cal, m_te = np.isin(pid, tr_p), np.isin(pid, cal_p), np.isin(pid, te_p)

        base_40 = make_pipeline(SplineTransformer(n_knots=5, degree=3), LinearRegression()).fit(ld[m_tr, None], q[m_tr])
        res_40 = q[m_tr] - base_40.predict(ld[m_tr, None])

        gbdt_40 = make_gbdt_probe().fit(feats_all[m_tr], res_40)

        s_dia_cal = base_40.predict(ld[m_cal, None])
        s_dia_te = base_40.predict(ld[m_te, None])
        s_gbdt_cal = s_dia_cal + gbdt_40.predict(feats_all[m_cal])
        s_gbdt_te = s_dia_te + gbdt_40.predict(feats_all[m_te])

        lam_d = compute_crc_lambda(s_dia_cal, w[m_cal], 0.05)
        lam_g = compute_crc_lambda(s_gbdt_cal, w[m_cal], 0.05)

        sign_dia.append((s_dia_te < lam_d).mean())
        sign_gbdt.append((s_gbdt_te < lam_g).mean())

    diff_gbdt = np.array(sign_gbdt) - np.array(sign_dia)
    ci_lo, ci_hi = np.percentile(diff_gbdt, 2.5), np.percentile(diff_gbdt, 97.5)
    
    print(f"  -> 非线性 GBDT 混合模型直签率: {np.mean(sign_gbdt):.3%} (纯物理直径: {np.mean(sign_dia):.3%})")
    print(f"  -> 非线性探针吞吐量增量: {np.mean(diff_gbdt):+.3%} (95% 独立切分 CI: [{ci_lo:+.3%}, {ci_hi:+.3%}])")
    
    if ci_lo <= 0 and ci_hi >= 0:
        print("  -> 【定论】：95% CI 跨越 0。证据不足以断言深度特征 (foc+ctx) 能带来统计学显著的直签增益。")

    # 落盘 OOF 与 Pareto (包含 GBDT 预测结果)
    pd.DataFrame({
        "pid": pid, "diam": diam, "ld_true": ld, "q_true": q,
        "p_dia": p_dia, "p_hyb_linear": p_hyb_linear, "p_hyb_gbdt": p_hyb_gbdt, 
        "p_fm": p_fm, "p_ld_decoded": p_ld
    }).to_csv(os.path.join(RESULT_DIR, "01_scorer_oof.csv"), index=False)

    rng_noise = np.random.default_rng(0)
    s_oracle = q + rng_noise.normal(0, 1e-6, n)
    pareto_records = []
    for rng in rng_list:
        up = np.unique(pid)
        c_p = rng.choice(up, int(len(up) * 0.5), replace=False)
        m_c, m_t = np.isin(pid, c_p), ~np.isin(pid, c_p)
        for al in ALPHAS:
            for s_name, s_arr in [("Diameter", p_dia), ("Hybrid_Lin", p_hyb_linear), 
                                  ("Hybrid_GBDT", p_hyb_gbdt), ("FMCIB", p_fm), ("Oracle", s_oracle)]:
                lam = compute_crc_lambda(s_arr[m_c], w[m_c], al)
                a = s_arr[m_t] < lam
                pareto_records.append({"alpha": al, "scorer": s_name, "sign": a.mean(), "risk": calculate_risk(w[m_t], a)})
    
    pd.DataFrame(pareto_records).groupby(["alpha", "scorer"]).mean().reset_index().to_csv(os.path.join(RESULT_DIR, "01_scorer_pareto.csv"), index=False)


if __name__ == "__main__":
    run_scorer_diagnostics()