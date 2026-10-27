# -*- coding: utf-8 -*-
"""
04_guideline_ltt_pac.py: 全量经验风险曲线提取与真实 PAC 众数切点 (落实严格审查)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from rcrc_core import (load_cohort, get_risk_definitions, compute_patient_weights, 
                       compute_ltt_lambda, RESULT_DIR, ALPHAS)


def run_guideline_cutoffs():
    print("\n" + "=" * 80)
    print(">>> 正在运行 04: 真实物理尺寸曲线提取与 PAC 认证切点")
    print("=" * 80)
    
    df, Rind, _, _, M = load_cohort()
    pid = df.pid.values
    d = df.diam.values
    grid_T = np.arange(4.0, 14.5, 0.5)

    # 1. 固定 K=3 读者定义基准
    rng_k3 = np.random.default_rng(2025)
    M_binary = np.where(np.isnan(M), np.nan, (M >= 4).astype(float))
    Rind_k3 = np.full((len(df), 3), np.nan)
    for i, row in enumerate(M_binary):
        valid = row[~np.isnan(row)]
        Rind_k3[i] = rng_k3.choice(valid, 3, replace=False)
    
    D = get_risk_definitions(Rind_k3)

    # 2. 全量经验风险曲线与 Bootstrap 置信带
    print("[*] 正在计算全量真实经验阶梯风险曲线与患者 Bootstrap 95% CI...")
    unique_p, inv_p = np.unique(pid, return_inverse=True)
    n_pts = len(unique_p)
    A_mat = (d[:, None] < grid_T[None, :]).astype(float)
    rng_boot = np.random.default_rng(0)
    boot_pt_idx = [rng_boot.integers(0, n_pts, n_pts) for _ in range(1000)]

    curve_rows = []
    for name in ("majority", "mean", "any"):
        w = compute_patient_weights(df, D[name])
        M_p = np.bincount(inv_p, weights=w, minlength=n_pts)
        L_p = np.stack([np.bincount(inv_p, weights=w * A_mat[:, j], minlength=n_pts) for j in range(len(grid_T))], axis=1)
        pt_risk = L_p.sum(axis=0) / M_p.sum()
        
        boot_risks = [L_p[b].sum(axis=0) / M_p[b].sum() for b in boot_pt_idx]
        boot_risks = np.array(boot_risks)
        ci_lo = np.percentile(boot_risks, 2.5, axis=0)
        ci_hi = np.percentile(boot_risks, 97.5, axis=0)

        for j, cut in enumerate(grid_T):
            curve_rows.append({
                "definition": name, "cutoff": cut,
                "risk_real": pt_risk[j], "ci_lo": ci_lo[j], "ci_hi": ci_hi[j]
            })

    df_curves = pd.DataFrame(curve_rows)
    df_curves.to_csv(os.path.join(RESULT_DIR, "04_guideline_curves.csv"), index=False)
    print(f"[+] 真实经验风险曲线数据已落盘 -> {os.path.join(RESULT_DIR, '04_guideline_curves.csv')}")

    # 3. 20 个独立种子 x perms=1000 稳定 PAC 众数切点
    print("\n[D1: 20 个独立种子 x perms=1000 计算 PAC 众数认证切点]")
    seeds = np.arange(202501, 202521)
    pac_rows = []

    for name in ("majority", "mean", "any"):
        w = compute_patient_weights(df, D[name])
        for al in ALPHAS:
            cutoffs = []
            for s in seeds:
                lam = compute_ltt_lambda(d, w, pid, al, delta=0.10, grid=grid_T, floor=1e-2, perms=1000, seed=int(s))
                cutoffs.append(lam)
            
            m_val = float(pd.Series(cutoffs).mode().iloc[0])
            min_c, max_c = np.min(cutoffs), np.max(cutoffs)
            pac_rows.append({"definition": name, "alpha": al, "mode_cutoff": m_val, "min_c": min_c, "max_c": max_c})
            print(f"  -> {name:<10} | alpha={al:.2f} | 众数认证切点: < {m_val:4.1f} mm ([< {min_c:.1f}, < {max_c:.1f}] mm)")

    df_pac_out = pd.DataFrame(pac_rows)
    df_pac_out.to_csv(os.path.join(RESULT_DIR, "04_pac_cutoffs.csv"), index=False)
    print(f"[+] PAC 认证切点已落盘 -> {os.path.join(RESULT_DIR, '04_pac_cutoffs.csv')}")


if __name__ == "__main__":
    run_guideline_cutoffs()