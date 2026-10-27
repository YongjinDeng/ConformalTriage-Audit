# -*- coding: utf-8 -*-
"""
08_strict_statistics.py: 患者级 Bootstrap CI、份额稳健化、PAC 嵌套留出与全片免审敏感性分析
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

from rcrc_core import (load_cohort, get_risk_definitions, compute_patient_weights, calculate_risk,
                       compute_crc_lambda, compute_ltt_lambda, child_rngs, sample_k_readers,
                       binarize_ratings, expand_patients, RESULT_DIR, ALPHAS, CANON_K3_SEED)


def run_strict_statistics(n_boot1=1000, n_boot2=500, inner=5, n_cert=100, perms=100):
    print("\n" + "=" * 80)
    print(">>> 08: 患者级 Bootstrap CI / 份额稳健化 / PAC 嵌套留出 / 临床全片免审敏感性")
    print("=" * 80)

    df, _, _, _, M = load_cohort()
    pid, diam, q = df.pid.values, df.diam.values, df.q.values
    ld = np.log(diam)
    thin = df.slice_th.values <= 1.25
    Mb = binarize_ratings(M)
    up, inv = np.unique(pid, return_inverse=True)
    n_pts = len(up)
    ids_by_pt = [np.where(inv == i)[0] for i in range(n_pts)]
    n_p = df.groupby("pid").pid.transform("size").values.astype(float)
    w = q / n_p
    out_rows = []

    def split_boot(rng, pts, frac):
        draw = rng.choice(pts, len(pts), replace=True)
        u, c = np.unique(draw, return_counts=True)
        perm = rng.permutation(len(u))
        k = int(frac * len(u))
        a, b = perm[:k], perm[k:]
        return expand_patients(ids_by_pt, u[a], c[a]), expand_patients(ids_by_pt, u[b], c[b])

    # 1. Majority -> Any 定义泄漏倍数 Bootstrap CI
    print("\n[1. Majority -> Any 定义泄漏倍数 (α=0.05) 患者级 Bootstrap 95% CI]")
    rng = np.random.default_rng(42)
    leak_any, leak_maj = [], []
    all_pts = np.arange(n_pts)
    for _ in range(n_boot1):
        Rind = sample_k_readers(Mb, rng)
        w_any = Rind.max(axis=1) / n_p
        w_maj = (Rind.mean(axis=1) >= 0.5).astype(float) / n_p
        i_cal, i_te = split_boot(rng, all_pts, 0.5)
        lam = compute_crc_lambda(diam[i_cal], w_maj[i_cal], 0.05)
        if np.isfinite(lam):
            leak_any.append(calculate_risk(w_any[i_te], diam[i_te] < lam) / 0.05)
            leak_maj.append(calculate_risk(w_maj[i_te], diam[i_te] < lam) / 0.05)
    lo, hi = np.percentile(leak_any, [2.5, 97.5])
    print(f"  -> Any 评估泄漏倍数: 中位数 {np.median(leak_any):.2f}x, 95% CI [{lo:.2f}x, {hi:.2f}x]")
    out_rows.append(dict(item="leak_any_ratio", median=np.median(leak_any), lo=lo, hi=hi))

    # 2. 厚→薄 直径可解释份额 Bootstrap CI
    print("\n[2. 厚→薄 直径可解释份额 (α=0.05) 患者级 Bootstrap 95% CI]")
    x = ld[:, None]
    thick_pts, thin_pts = np.unique(inv[~thin]), np.unique(inv[thin])

    def shift_once(rng, alpha=0.05):
        i_sc, i_st = split_boot(rng, thick_pts, 0.7)
        i_oc, i_ot = split_boot(rng, thin_pts, 0.5)
        i_sc, i_st = i_sc[~thin[i_sc]], i_st[~thin[i_st]]
        i_oc, i_ot = i_oc[thin[i_oc]], i_ot[thin[i_ot]]
        if min(len(i_sc), len(i_st), len(i_oc), len(i_ot)) < 20:
            return None
        clf = make_pipeline(SplineTransformer(n_knots=5, degree=3), LogisticRegression(C=1.0, max_iter=2000)).fit(
            np.r_[x[i_sc], x[i_oc]], np.r_[np.zeros(len(i_sc)), np.ones(len(i_oc))])
        p = np.clip(clf.predict_proba(x[i_ot])[:, 1], 0.02, 0.98)
        ratio = np.clip(p / (1.0 - p) * len(i_sc) / len(i_oc), 0.1, 10.0)
        lam = compute_crc_lambda(diam[i_sc], w[i_sc], alpha)
        if not np.isfinite(lam):
            return None
        a = diam[i_ot] < lam
        return (calculate_risk(w[i_st], diam[i_st] < lam), calculate_risk(w[i_ot], a),
                calculate_risk(w[i_ot] / ratio, a))

    rng2 = np.random.default_rng(4242)
    ex_l, dc_l, sh_l = [], [], []
    for _ in range(n_boot2):
        vals = [v for v in (shift_once(rng2) for _ in range(inner)) if v is not None]
        if not vals:
            continue
        r_src, r_raw, r_diag = np.mean(vals, axis=0)
        ex, dc = r_raw - r_src, r_raw - r_diag
        ex_l.append(ex); dc_l.append(dc)
        sh_l.append(dc / ex if ex > 0.01 else np.nan)
    sh = np.array(sh_l)
    ok = ~np.isnan(sh)
    lo_s, hi_s = np.percentile(sh[ok], [2.5, 97.5])
    lo_e, hi_e = np.percentile(ex_l, [2.5, 97.5])
    lo_d, hi_d = np.percentile(dc_l, [2.5, 97.5])
    print(f"  -> 总超额风险 R_raw-R_src : 中位数 {np.median(ex_l):.4f}, 95% CI [{lo_e:.4f}, {hi_e:.4f}]")
    print(f"  -> 直径可解释部分 Δcov    : 中位数 {np.median(dc_l):.4f}, 95% CI [{lo_d:.4f}, {hi_d:.4f}]")
    print(f"  -> 份额 Δcov/超额         : 中位数 {np.median(sh[ok]):.1%}, 95% CI [{lo_s:.1%}, {hi_s:.1%}]")
    out_rows += [dict(item="excess_risk", median=np.median(ex_l), lo=lo_e, hi=hi_e),
                 dict(item="delta_cov", median=np.median(dc_l), lo=lo_d, hi=hi_d),
                 dict(item="diam_share", median=np.median(sh[ok]), lo=lo_s, hi=hi_s)]

    # 3. PAC-LTT 嵌套留出验证
    print("\n[3. PAC-LTT 嵌套留出验证 (δ=0.10)]")
    D = get_risk_definitions(sample_k_readers(Mb, np.random.default_rng(CANON_K3_SEED)))
    grid_T = np.arange(4.0, 16.5, 0.5)
    splits = []
    for r in child_rngs(999, n_cert):
        perm = r.permutation(n_pts)
        f = np.zeros(n_pts, bool)
        f[perm[: n_pts // 2]] = True
        splits.append(f[inv])
    rows = []
    for name in ("majority", "mean", "any"):
        wd = compute_patient_weights(df, D[name])
        for al in ALPHAS:
            cert = viol = 0
            for k, m_cal in enumerate(splits):
                m_te = ~m_cal
                lam = compute_ltt_lambda(diam[m_cal], wd[m_cal], pid[m_cal], al, 0.10, grid_T, perms=perms, seed=k)
                if lam == -np.inf:
                    continue
                cert += 1
                viol += calculate_risk(wd[m_te], diam[m_te] < lam) > al
            rows.append(dict(definition=name, alpha=al, n_splits=n_cert, certified=cert,
                             empty=n_cert - cert, holdout_violation=viol, valid=cert - viol))
    df_c = pd.DataFrame(rows)
    df_c.to_csv(os.path.join(RESULT_DIR, "08_pac_nested_validation.csv"), index=False)
    pd.DataFrame(out_rows).to_csv(os.path.join(RESULT_DIR, "08_bootstrap_summary.csv"), index=False)

    # 4. 临床工作流“全片完全免审”与“患者真漏诊”敏感性分析
    print("\n[4. 临床工作流全片完全免审 (Patient-Level Complete Sign-off) 敏感性分析]")
    y_any = D["any"]
    pt_rows = []
    for al in [0.05, 0.10]:
        full_sign_rates = []
        patient_miss_rates = []
        for r_idx in range(50):
            rng_pt = np.random.default_rng(r_idx + 800)
            perm_p = rng_pt.permutation(n_pts)
            h = n_pts // 2
            cal_p, te_p = perm_p[:h], perm_p[h:]
            m_cal = np.isin(inv, cal_p)
            m_te = np.isin(inv, te_p)
            
            lam = compute_crc_lambda(diam[m_cal], (y_any / n_p)[m_cal], al)
            if not np.isfinite(lam): continue
            
            df_te = df[m_te].copy()
            df_te["is_susp"] = y_any[m_te]
            df_te["is_signed"] = diam[m_te] < lam
            
            # 患者级全片免审逻辑：仅当患者体内所有结节均 < lam 时，该患者才被完全直签放行
            pt_summary = df_te.groupby("pid").agg(
                has_susp=("is_susp", lambda x: (x > 0).any()),
                all_signed=("is_signed", "all") # 全片直签
            )
            # 患者级真漏诊：患者体内有可疑结节，但被全片放行
            pt_summary["missed_patient"] = pt_summary.has_susp & pt_summary.all_signed
            
            full_sign_rates.append(pt_summary.all_signed.mean())
            # 在有可疑结节的患者中，被完全漏诊放行的患者比例
            susp_pts = pt_summary[pt_summary.has_susp]
            pt_miss = susp_pts.missed_patient.mean() if len(susp_pts) else 0.0
            patient_miss_rates.append(pt_miss)
            
        mean_sign = np.mean(full_sign_rates)
        mean_miss = np.mean(patient_miss_rates)
        print(f"  -> Any 规则 (α={al:.2f}) | 全片完全直签免审率: {mean_sign:.1%} | 患者级真实漏诊率: {mean_miss:.4f}")
        pt_rows.append({"alpha": al, "patient_full_signoff_rate": mean_sign, "patient_true_miss_rate": mean_miss})

    pd.DataFrame(pt_rows).to_csv(os.path.join(RESULT_DIR, "08_patient_binary_sensitivity.csv"), index=False)
    print(f"[+] 临床全片免审敏感性分析已落盘 -> {os.path.join(RESULT_DIR, '08_patient_binary_sensitivity.csv')}")


if __name__ == "__main__":
    run_strict_statistics()