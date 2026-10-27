# -*- coding: utf-8 -*-
"""
02_definition_audit.py: 读者定义泄漏矩阵审计、tau 敏感性测试与 K=4 全读者面板真实泄漏分析
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from rcrc_core import (load_cohort, get_risk_definitions, compute_patient_weights, 
                       calculate_risk, compute_crc_lambda, child_rngs, RESULT_DIR)


def run_definition_audit(n_subsamples=200):
    print("\n" + "=" * 80)
    print(">>> 正在运行 02: 读者定义泄漏矩阵 (K=3 抽样 / tau 敏感性 / K=4 全量面板)")
    print("=" * 80)
    
    df, _, _, feats, M = load_cohort()
    pid = df.pid.values
    diam = df.diam.values
    
    # 1. tau=4 基准下的固定 K=3 抽样泄漏矩阵
    M_binary = np.where(np.isnan(M), np.nan, (M >= 4).astype(float))
    defs_names = ["any", "majority", "mean", "unanimous"]
    risk_records = {c: {e: [] for e in defs_names} for c in defs_names}
    sign_records = {c: [] for c in defs_names}

    for s_idx in range(n_subsamples):
        rng_k = np.random.default_rng(s_idx + 1000)
        Rind_k3 = np.full((len(df), 3), np.nan)
        for i, row in enumerate(M_binary):
            valid = row[~np.isnan(row)]
            Rind_k3[i] = rng_k.choice(valid, 3, replace=False)
        
        D_k3 = get_risk_definitions(Rind_k3)
        W_k3 = {k: compute_patient_weights(df, v) for k, v in D_k3.items()}

        up = np.unique(pid)
        c_p = rng_k.choice(up, int(len(up) * 0.5), replace=False)
        m_cal, m_te = np.isin(pid, c_p), ~np.isin(pid, c_p)

        for cal_name in defs_names:
            lam = compute_crc_lambda(diam[m_cal], W_k3[cal_name][m_cal], 0.05)
            auto_sign = diam[m_te] < lam
            sign_records[cal_name].append(auto_sign.mean())
            for ev_name in defs_names:
                r = calculate_risk(W_k3[ev_name][m_te], auto_sign)
                risk_records[cal_name][ev_name].append(r)

    mean_matrix = pd.DataFrame(index=defs_names, columns=defs_names)
    sd_matrix = pd.DataFrame(index=defs_names, columns=defs_names)
    for c in defs_names:
        for e in defs_names:
            arr = np.array(risk_records[c][e])
            mean_matrix.loc[c, e] = np.mean(arr)
            sd_matrix.loc[c, e] = np.std(arr, ddof=1)

    df_signoff = pd.DataFrame({
        "cal": defs_names,
        "sign_mean": [np.mean(sign_records[c]) for c in defs_names],
        "sign_sd": [np.std(sign_records[c], ddof=1) for c in defs_names]
    })

    mean_matrix.to_csv(os.path.join(RESULT_DIR, "02_definition_matrix.csv"))
    sd_matrix.to_csv(os.path.join(RESULT_DIR, "02_definition_matrix_sd.csv"))
    df_signoff.to_csv(os.path.join(RESULT_DIR, "02_definition_signoff.csv"), index=False)
    print(f"[+] K=3 泄漏矩阵与直签率均值及 SD 已落盘")

    # 2. tau 敏感性分析 (tau in 3, 4, 5)
    print("\n[*] 正在运行 tau 敏感性分析并落盘...")
    tau_records = []
    for tau_test in [3, 4, 5]:
        Rind_t = np.where(np.isnan(M), np.nan, (M >= tau_test).astype(float))
        D_t = get_risk_definitions(Rind_t)
        W_t = {k: compute_patient_weights(df, v) for k, v in D_t.items()}
        
        rng_list = child_rngs(tau_test * 100, 100)
        t_risks_any, t_risks_maj = [], []
        for rng in rng_list:
            up = np.unique(pid)
            c_p = rng.choice(up, int(len(up) * 0.5), replace=False)
            m_cal, m_te = np.isin(pid, c_p), ~np.isin(pid, c_p)
            
            lam = compute_crc_lambda(diam[m_cal], W_t["majority"][m_cal], 0.05)
            a = diam[m_te] < lam
            t_risks_any.append(calculate_risk(W_t["any"][m_te], a))
            t_risks_maj.append(calculate_risk(W_t["majority"][m_te], a))
            
        r_any_m = np.mean(t_risks_any)
        r_maj_m = np.mean(t_risks_maj)
        tau_records.append({
            "tau": tau_test,
            "majority_prevalence": np.nanmean(D_t["majority"]),
            "cal_majority_eval_any_risk": r_any_m,
            "cal_majority_eval_maj_risk": r_maj_m,
            "excess_leakage_ratio": r_any_m / 0.05
        })

    df_tau = pd.DataFrame(tau_records)
    tau_csv = os.path.join(RESULT_DIR, "02_tau_sensitivity.csv")
    df_tau.to_csv(tau_csv, index=False)
    print(f"[+] tau 敏感性结果已落盘 -> {tau_csv}")

    # 3. K=4 全读者真实子集面板泄漏分析 (无 NaN 切片)
    print("\n[真实数据分析: K=4 全读者面板未稀释泄漏分析]")
    mask_4r = (df.n_readers == 4).values
    df_4r = df[mask_4r].reset_index(drop=True)
    
    # 严格切取前 4 列并验证
    M_4r = M[mask_4r, :4]
    assert not np.isnan(M_4r).any(), "在 n_readers == 4 子集中存在意外的 NaN 打分"
    
    M_binary_4r = (M_4r >= 4).astype(float)
    y_any_4r = M_binary_4r.max(axis=1)
    y_maj_4r = (M_binary_4r.mean(axis=1) >= 0.5).astype(float)
    
    w_any_4r = compute_patient_weights(df_4r, y_any_4r)
    w_maj_4r = compute_patient_weights(df_4r, y_maj_4r)
    diam_4r = df_4r.diam.values
    pid_4r = df_4r.pid.values

    rng_4r = np.random.default_rng(2026)
    k4_leakage = []
    up_4r = np.unique(pid_4r)
    for _ in range(200):
        c_p = rng_4r.choice(up_4r, int(len(up_4r) * 0.5), replace=False)
        m_cal, m_te = np.isin(pid_4r, c_p), ~np.isin(pid_4r, c_p)
        lam_maj = compute_crc_lambda(diam_4r[m_cal], w_maj_4r[m_cal], 0.05)
        if np.isfinite(lam_maj):
            r_any = calculate_risk(w_any_4r[m_te], diam_4r[m_te] < lam_maj)
            k4_leakage.append(r_any / 0.05)

    med_k4 = np.median(k4_leakage)
    lo_k4, hi_k4 = np.percentile(k4_leakage, [2.5, 97.5])
    print(f"  -> K=4 读者结节数: {len(df_4r)} 颗 (涵盖 {len(up_4r)} 例患者)")
    print(f"  -> K=4 面板下 Majority 校准对 Any 评估的泄漏倍数: 中位数 {med_k4:.2f}x (95% CI: [{lo_k4:.2f}x, {hi_k4:.2f}x])")
    
    pd.DataFrame([{
        "experiment": "K=4_reader_panel", "n_nodules": len(df_4r), "n_patients": len(up_4r),
        "median_leak_ratio": med_k4, "ci_2.5": lo_k4, "ci_97.5": hi_k4
    }]).to_csv(os.path.join(RESULT_DIR, "02_k4_panel_leakage.csv"), index=False)
    print(f"[+] K=4 真实面板泄漏分析已落盘 -> {os.path.join(RESULT_DIR, '02_k4_panel_leakage.csv')}")


if __name__ == "__main__":
    run_definition_audit()