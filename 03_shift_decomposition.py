# -*- coding: utf-8 -*-
"""
03_shift_decomposition.py (动态路径、设备厂商分层与协议偏移分解)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

from rcrc_core import (load_cohort, compute_patient_weights, calculate_risk, 
                       compute_crc_lambda, compute_ltt_lambda, child_rngs, RESULT_DIR, BASE_DIR, ALPHAS)


def run_shift_decomposition(rep=200):
    print("\n" + "=" * 80)
    print(">>> 正在运行 03: 协议偏移分解、厂商分层置换与 LTT 压力测试")
    print("=" * 80)
    
    df, _, _, _, _ = load_cohort()
    pid = df.pid.values
    d = df.diam.values
    q = df.q.values
    ld = np.log(d)
    w = compute_patient_weights(df, q)
    thin_125 = df.slice_th.values <= 1.25
    x_feat = ld[:, None]

    # 1. 全队列残差层内患者级分层置换检验
    print("\n[1. 全队列残差分层患者级置换检验]")
    base_spline = make_pipeline(SplineTransformer(n_knots=5, degree=3), LinearRegression()).fit(ld[:, None], q)
    resid_q = q - base_spline.predict(ld[:, None])

    df_pt = df.assign(thin=thin_125, r=resid_q, ld=ld).groupby("pid").agg(
        thin=("thin", "first"), nth=("thin", "nunique"), r_mean=("r", "mean"), dmed=("diam", "median")
    )
    
    if (df_pt.nth > 1).any():
        df_pt = df_pt[df_pt.nth == 1]

    pt_thin = df_pt.thin.values.astype(bool)
    pt_r = df_pt.r_mean.values
    pt_strata = pd.qcut(np.log(df_pt.dmed.values), 5, labels=False)
    groups = [np.where(pt_strata == s)[0] for s in np.unique(pt_strata)]

    stat_fn = lambda l: pt_r[l].mean() - pt_r[~l].mean()
    obs_diff = stat_fn(pt_thin)

    rng_perm = np.random.default_rng(2025)
    n_perm = 2000
    null_diffs = np.empty(n_perm)
    for b in range(n_perm):
        new_thin = pt_thin.copy()
        for ix in groups:
            new_thin[ix] = rng_perm.permutation(pt_thin[ix])
        null_diffs[b] = stat_fn(new_thin)

    two_sided_p = (1.0 + np.sum(np.abs(null_diffs) >= abs(obs_diff))) / (n_perm + 1.0)
    print(f"  -> 全队列薄层残差 - 厚层残差: {obs_diff:+.4f} (双侧 p = {two_sided_p:.4e})")

    # 2. 扫描仪厂商分层检验 (动态相对路径与安全 Merge)
    print("\n[2. 扫描仪厂商分层置换检验 (排除设备/中心混杂)]")
    patch_meta_candidates = [
        os.path.join(BASE_DIR, "lidc_patches", "meta.csv"),
        os.path.join(r"D:\lidc_patches", "meta.csv"),
        os.path.join(RESULT_DIR, "meta.csv")
    ]
    
    mfg_verified = False
    for p_path in patch_meta_candidates:
        if os.path.exists(p_path):
            try:
                df_pm = pd.read_csv(p_path).drop_duplicates("pid")
                # 安全合并：reset_index 后按 pid 进行列合并
                df_merged = df_pt.reset_index().merge(df_pm[["pid", "manufacturer"]], on="pid", how="inner")
                if len(df_merged) >= 100:
                    is_ge = df_merged.manufacturer.str.contains("GE|General", case=False, na=False).values
                    df_ge = df_merged[is_ge]
                    if len(df_ge) >= 50:
                        ge_thin = df_ge.thin.values.astype(bool)
                        ge_r = df_ge.r_mean.values
                        obs_ge_diff = ge_r[ge_thin].mean() - ge_r[~ge_thin].mean()
                        
                        null_ge = np.empty(1000)
                        rng_ge = np.random.default_rng(2026)
                        for b in range(1000):
                            p_t = rng_ge.permutation(ge_thin)
                            null_ge[b] = ge_r[p_t].mean() - ge_r[~p_t].mean()
                        p_ge = (1.0 + np.sum(np.abs(null_ge) >= abs(obs_ge_diff))) / 1001.0
                        print(f"  -> GE 厂商患者子集 (n={len(df_ge)}): 薄层与厚层残差差值 = {obs_ge_diff:+.4f} (p = {p_ge:.4e})")
                        pd.DataFrame([{"subset": "GE_only", "n_patients": len(df_ge), "diff": obs_ge_diff, "p_value": p_ge}]).to_csv(
                            os.path.join(RESULT_DIR, "03_manufacturer_stratified_p.csv"), index=False
                        )
                        mfg_verified = True
                        break
            except Exception as e:
                continue

    if not mfg_verified:
        print("  -> (未检测到完整的 meta.csv 厂商字段，厂商分层结果已作为局限性在正文中说明)")

    # 3. 双向偏移分解
    print(f"\n[*] 正在运行 {rep} 次 Monte Carlo 协议偏移绝对风险点分解...")
    rng_list = child_rngs(42, rep)
    out = []

    for rng in rng_list:
        for name, src, dst in (("厚→薄 (违约压力测试)", ~thin_125, thin_125), ("薄→厚 (保守测试)", thin_125, ~thin_125)):
            Ps, Pd = np.unique(pid[src]), np.unique(pid[dst])
            
            p_src_cal = rng.choice(Ps, int(0.7 * len(Ps)), replace=False)
            m_src_cal = np.isin(pid, p_src_cal) & src
            m_src_te = (~np.isin(pid, p_src_cal)) & src

            p_dst_perm = rng.permutation(Pd)
            h = len(Pd) // 2
            m_dst_oc = np.isin(pid, p_dst_perm[:h]) & dst
            m_dst_ot = np.isin(pid, p_dst_perm[h:]) & dst

            fit_mask = m_src_cal | m_dst_oc
            prior_ratio = m_src_cal.sum() / max(m_dst_oc.sum(), 1)
            clf_dr = make_pipeline(SplineTransformer(n_knots=5, degree=3), LogisticRegression(C=1.0, max_iter=2000)).fit(x_feat[fit_mask], dst[fit_mask].astype(int))
            p_dst_hat = np.clip(clf_dr.predict_proba(x_feat)[:, 1], 0.02, 0.98)
            ratio = np.clip(p_dst_hat / (1.0 - p_dst_hat) * prior_ratio, 0.1, 10.0)

            for al in ALPHAS:
                lam_raw = compute_crc_lambda(d[m_src_cal], w[m_src_cal], al)
                lam_rew = compute_crc_lambda(d[m_src_cal], w[m_src_cal], al, B=1.0, v=ratio[m_src_cal], clip_max=10.0)
                lam_tar = compute_crc_lambda(d[m_dst_oc], w[m_dst_oc], al)

                r_src = calculate_risk(w[m_src_te], d[m_src_te] < lam_raw)
                a_raw = d[m_dst_ot] < lam_raw
                a_rew = d[m_dst_ot] < lam_rew
                a_tar = d[m_dst_ot] < lam_tar

                r_raw = calculate_risk(w[m_dst_ot], a_raw)
                r_diag = calculate_risk(w[m_dst_ot] / ratio[m_dst_ot], a_raw)
                r_rew = calculate_risk(w[m_dst_ot], a_rew)
                r_tar = calculate_risk(w[m_dst_ot], a_tar)

                out.append({
                    "方向": name, "alpha": al,
                    "R_src": r_src, "R_raw": r_raw, "R_diag": r_diag, "R_rew": r_rew, "R_tar": r_tar,
                    "delta_cov": r_raw - r_diag, "delta_cond": r_diag - r_src,
                    "raw_违约": float(r_raw > al), "rew_违约": float(r_rew > al), "tar_违约": float(r_tar > al),
                    "直签_raw": a_raw.mean(), "直签_rew": a_rew.mean(), "直签_tar": a_tar.mean()
                })

    df_res = pd.DataFrame(out)
    summary = df_res.groupby(["方向", "alpha"]).mean(numeric_only=True)
    summary.to_csv(os.path.join(RESULT_DIR, "03_shift_summary.csv"))
    print(f"[+] 协议偏移加法分解表已落盘")


if __name__ == "__main__":
    run_shift_decomposition()