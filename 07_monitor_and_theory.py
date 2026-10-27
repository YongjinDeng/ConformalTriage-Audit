# -*- coding: utf-8 -*-
"""
07_monitor_and_theory.py (终极稳定无报错版):
修复 Matplotlib 字符串转义错误，增加 n_min 理论界限指示线
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rcrc_core import (load_cohort, get_risk_definitions, compute_patient_weights, calculate_risk,
                       compute_crc_lambda, betting_path, sample_k_readers, binarize_ratings,
                       child_rngs, RESULT_DIR, CANON_K3_SEED)

FIG_DIR = os.path.join(RESULT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

ALPHA, DELTA, FLOOR = 0.05, 0.10, 1e-3
AUDIT_SIZES = [20, 50, 100, 150, 200, 300]
LOCAL_N = [20, 40, 60, 80, 100, 150, 200]


def run_monitor_and_theory(n_trials=1000, n_rep_local=200):
    print("\n" + "=" * 80)
    print(">>> 07 (JBHI 规范版): Anytime-Valid 报警监控 / 本地重校准 / 定义序关系")
    print("=" * 80)

    df, _, _, _, M = load_cohort()
    pid, diam, q = df.pid.values, df.diam.values, df.q.values
    thin = df.slice_th.values <= 1.25
    w = compute_patient_weights(df, q)
    up, inv = np.unique(pid, return_inverse=True)
    n_pts = len(up)

    def mask_of(ids):
        f = np.zeros(n_pts, bool)
        f[ids] = True
        return f[inv]

    def pts_in(dm):
        return np.unique(inv[dm])

    thick_ids, thin_ids = pts_in(~thin), pts_in(thin)
    print(f"[*] 厚层患者 {len(thick_ids)} / 薄层患者 {len(thin_ids)} / 跨层厚患者 {len(np.intersect1d(thick_ids, thin_ids))}")

    def patient_stats(dm, lam):
        a = dm & (diam < lam)
        l = np.bincount(inv, weights=w * a, minlength=n_pts)
        m = np.bincount(inv, weights=w * dm, minlength=n_pts)
        return l, m

    def pool_risk(l, m, pool):
        return l[pool].sum() / max(m[pool].sum(), 1e-12)

    def alarm_runmax(l, m, pool, n_max, rng):
        ids = rng.choice(pool, n_max, replace=True)           
        U = np.clip(l[ids] - ALPHA * m[ids] + ALPHA, 0.0, 1.0)  
        return np.maximum.accumulate(betting_path(U, ALPHA, floor=FLOOR, direction="up"))

    # 1. 在线监控仿真 (含 H0 池、H1 漂移池及临界 Boundary 流)
    print("\n[1. 在线监控: 校准(厚层一半) / H0 监控池(厚层另一半) / H1 监控池(薄层漂移)]")
    thr, max_n = 1.0 / DELTA, max(AUDIT_SIZES)
    rec, n_vacuous = [], 0
    boundary_alarms = []

    for rng in child_rngs(2025, n_trials):
        perm = rng.permutation(thick_ids)
        h = len(perm) // 2
        m_cal = mask_of(perm[:h]) & ~thin
        lam = compute_crc_lambda(diam[m_cal], w[m_cal], ALPHA)
        if not np.isfinite(lam):
            n_vacuous += 1
            continue
        dm0 = mask_of(perm[h:]) & ~thin
        pool0 = pts_in(dm0)
        l0, m0 = patient_stats(dm0, lam)
        l1, m1 = patient_stats(thin, lam)
        R0, R1 = pool_risk(l0, m0, pool0), pool_risk(l1, m1, thin_ids)
        rm0 = alarm_runmax(l0, m0, pool0, max_n, rng)
        rm1 = alarm_runmax(l1, m1, thin_ids, max_n, rng)
        for n in AUDIT_SIZES:
            rec.append(dict(n=n, R0=R0, R1=R1, alarm0=bool(rm0[n - 1] >= thr), alarm1=bool(rm1[n - 1] >= thr)))

        # 临界流检验 (Boundary stream where E[U] = alpha = 0.05)
        U_boundary = rng.binomial(1, ALPHA, max_n).astype(float)
        rm_b = np.maximum.accumulate(betting_path(U_boundary, ALPHA, floor=FLOOR, direction="up"))
        boundary_alarms.append(rm_b[-1] >= thr)

    R = pd.DataFrame(rec)
    boundary_fa_rate = np.mean(boundary_alarms)
    print(f"  (空认证试验 {n_vacuous}/{n_trials}; 厚层监控池平均风险 {R.R0.mean():.4f}; 薄层(漂移)平均风险 {R.R1.mean():.4f})")
    print(f"  -> 【临界边界流 E[U]=α=0.05】 实测误报率: {boundary_fa_rate:.1%} (严格受限于 δ=10%)")

    rows = []
    for n, g in R.groupby("n"):
        h0, h1 = g[g.R0 <= ALPHA], g[g.R1 > ALPHA]
        fa = h0.alarm0.mean() if len(h0) else np.nan
        pw = h1.alarm1.mean() if len(h1) else np.nan
        rows.append(dict(n=n, n_H0_true=len(h0), false_alarm=fa, n_H1=len(h1), power=pw,
                         alarm_rate_whole_H0pool=g.alarm0.mean()))
        print(f"  -> N={n:<4d} 患者 | H0 误报率 {fa:6.1%} | H1 漂移检出功效 {pw:6.1%}")
    df_mon = pd.DataFrame(rows)
    df_mon.to_csv(os.path.join(RESULT_DIR, "07_monitor_power.csv"), index=False)

    # 2. 本地重校准需求曲线
    print("\n[2. 目标域(薄层)本地重校准需求曲线 (α=0.05, CRC)]")
    rows2 = []
    for N in LOCAL_N:
        if len(thin_ids) - N < 30:
            continue
        risks, signs, vac = [], [], 0
        for rng in child_rngs(77 + N, n_rep_local):
            perm = rng.permutation(thin_ids)
            m_cal, m_te = mask_of(perm[:N]) & thin, mask_of(perm[N:]) & thin
            lam = compute_crc_lambda(diam[m_cal], w[m_cal], ALPHA)
            if lam == -np.inf:
                vac += 1
                risks.append(0.0)
                signs.append(0.0)
                continue
            a = diam[m_te] < lam
            risks.append(calculate_risk(w[m_te], a))
            signs.append(a.mean())
        rows2.append(dict(N=N, risk=np.mean(risks), sign=np.mean(signs),
                          viol=np.mean(np.array(risks) > ALPHA), vacuous=vac / n_rep_local))
        r = rows2[-1]
        print(f"  -> N={N:<4d} | 测试风险 {r['risk']:.4f} | 直签率 {r['sign']:.1%} | 空认证率 {r['vacuous']:.1%}")
    df_loc = pd.DataFrame(rows2)
    df_loc.to_csv(os.path.join(RESULT_DIR, "07_local_recal_curve.csv"), index=False)

    # ------------------------------------------------------------------
    # 3. 绘制更新后的 Figure 7 (规范双轴图例合并，彻底消除文字遮挡)
    # ------------------------------------------------------------------
    print("\n[*] 正在更新 Figure 7 并落盘...")
    fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.5))

    # 7a: 时序监控功效 (图例移至左上方空白区域)
    a = ax[0]
    l1 = a.plot(df_mon.n, df_mon.power, "o-", color="#d62728", lw=1.8, label="H1 power (thick→thin drift)")[0]
    l2 = a.plot(df_mon.n, df_mon.false_alarm, "s--", color="#1f77b4", lw=1.8, label="H0 false-alarm (compliant stream)")[0]
    l3 = a.axhline(DELTA, color="gray", ls="-.", lw=1.0, label=f"Significance budget δ = {DELTA:.2f}")
    a.set_xlabel("Audited patients N")
    a.set_ylabel("Alarm probability")
    a.set_title("(a) Anytime-valid betting monitor performance")
    a.set_ylim(-0.02, 0.65)
    a.grid(ls=":", alpha=0.6)
    a.legend(loc="upper left", framealpha=0.9)

    # 7b: 本地重校准双轴合并图例，移除遮挡文本框
    b = ax[1]
    b2 = b.twinx()

    line_risk = b.plot(df_loc.N, df_loc.risk, "o-", color="#d62728", lw=1.8, label="Test risk (pw-FNR)")[0]
    line_alpha = b.axhline(ALPHA, color="red", ls="-.", lw=1.0, label=f"Nominal α = {ALPHA:.2f}")
    line_nmin = b.axvline(82.3, color="blue", ls=":", lw=1.5, label="Analytical bound n_min ≈ 82.3")

    line_sign = b2.plot(df_loc.N, df_loc.sign * 100, "s--", color="#2ca02c", lw=1.8, label="Sign-off rate (%)")[0]

    b.set_xlabel("Labelled target patients N")
    b.set_ylabel("Patient-weighted risk (pw-FNR)", color="#d62728")
    b.set_ylim(-0.002, 0.065)
    b2.set_ylabel("Autonomous sign-off rate (%)", color="#2ca02c")
    b2.set_ylim(-1.0, 25.0)

    # 合并左轴与右轴的 Handles 和 Labels 至同一个图例框
    handles = [line_risk, line_alpha, line_nmin, line_sign]
    labels = [h.get_label() for h in handles]
    b.legend(handles, labels, loc="upper left", framealpha=0.9, fontsize=8.5)

    b.set_title("(b) Local target recalibration trade-off")
    b.grid(ls=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "Fig7_Monitor_and_Theory.png"), dpi=300)
    plt.close()
    print(f"[+] Figure 7 已重新生成并保存至 -> {FIG_DIR}")

if __name__ == "__main__":
    run_monitor_and_theory()