# -*- coding: utf-8 -*-
"""
06_report_figures.py (终极图表生成脚本 - 严格对齐 JBHI 规范与正文叙述)
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 8.5

try:
    from rcrc_core import RESULT_DIR
except ImportError:
    RESULT_DIR = os.environ.get("RCRC_BASE_DIR", r"D:\0临床科研\生物视觉图网络用于分类") + "\\result"

FIG_DIR = os.path.join(RESULT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def plot_fig2():
    print("[*] 正在绘制 Figure 2 (共识泄漏矩阵与吞吐量差异)...")
    df_mean = pd.read_csv(os.path.join(RESULT_DIR, "02_definition_matrix.csv"), index_col=0)
    df_sd = pd.read_csv(os.path.join(RESULT_DIR, "02_definition_matrix_sd.csv"), index_col=0)
    df_sign = pd.read_csv(os.path.join(RESULT_DIR, "02_definition_signoff.csv"))

    labels = ["Any", "Majority", "Mean", "Unanimous"]
    fig = plt.figure(figsize=(11.5, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    annot = np.array([[f"{m:.3f}\n±{s:.3f}" for m, s in zip(row_m, row_s)] 
                      for row_m, row_s in zip(df_mean.values, df_sd.values)])
    
    sns.heatmap(df_mean.values, annot=annot, fmt="", cmap="YlOrRd", ax=ax1,
                vmin=0.0, vmax=0.35, cbar_kws={'label': 'Empirical Ratio Risk'},
                linewidths=1.2, linecolor='white')
    for i in range(4):
        ax1.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='blue', lw=2.0, linestyle='--'))
    ax1.set_xticklabels(labels)
    ax1.set_yticklabels(labels, rotation=0)
    ax1.set_xlabel("Evaluation Objective")
    ax1.set_ylabel("Calibration Target Policy")
    ax1.set_title("(a) Cross-Definition Leakage Matrix (K=3 Readers, α=0.05)")

    ax2 = fig.add_subplot(gs[1])
    y_pos = np.arange(len(labels))
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
    sign_rates = df_sign["sign_mean"].values * 100
    sign_sds = df_sign["sign_sd"].values * 100

    ax2.barh(y_pos, sign_rates, xerr=sign_sds, align='center', color=colors, alpha=0.85, capsize=4, edgecolor='black', lw=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels)
    ax2.invert_yaxis()
    ax2.set_xlabel("Autonomous Sign-off Rate (%)")
    ax2.set_title("(b) Triage Throughput Disparity")
    ax2.axvline(sign_rates[0], color='#1f77b4', linestyle=':', lw=1.2, label='Any (Envelope) Baseline')
    ax2.legend(loc='lower right')
    ax2.set_xlim(0, 72)

    for i, v in enumerate(sign_rates):
        ax2.text(v + sign_sds[i] + 1.5, i, f"{v:.1f}%", va='center', fontsize=8.5, weight='bold')

    plt.savefig(os.path.join(FIG_DIR, "Fig2_Definition_Leakage.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_fig3():
    print("[*] 正在绘制 Figure 3 (探针基准与尺寸解码)...")
    df_oof = pd.read_csv(os.path.join(RESULT_DIR, "01_scorer_oof.csv"))
    df_pareto = pd.read_csv(os.path.join(RESULT_DIR, "01_scorer_pareto.csv"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))

    alphas = sorted(df_pareto.alpha.unique())
    p_pivot = df_pareto.pivot(index="alpha", columns="scorer", values="sign") * 100

    ax1.plot(alphas, p_pivot["Oracle"], 'k*--', lw=1.5, markersize=8, label="Theoretical Ceiling (Oracle: q)")
    ax1.plot(alphas, p_pivot["Diameter"], 'o-', color='#1f77b4', lw=2.2, label="Physical Caliber (Diameter)")
    ax1.plot(alphas, p_pivot["Hybrid_GBDT"], 'D-', color='#ff7f0e', lw=2.0, label="Residual GBDT Hybrid (Diam + FMCIB)")
    ax1.plot(alphas, p_pivot["FMCIB"], '^--', color='#d62728', lw=1.8, label="Deep Foundation Probe (FMCIB)")

    ax1.set_xlabel("Nominal Risk Tolerance (α)")
    ax1.set_ylabel("Autonomous Sign-off Rate (%)")
    ax1.set_title("(a) Triage Throughput Pareto Frontier")
    ax1.set_xticks(alphas)
    ax1.set_xticklabels([f"{int(a*100)}%" for a in alphas])
    ax1.set_ylim(0, 75)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower right', framealpha=0.9)

    ld_true = df_oof["ld_true"].values
    ld_pred = df_oof["p_ld_decoded"].values
    r2_real = 1.0 - np.mean((ld_true - ld_pred) ** 2) / np.var(ld_true)
    sp_real = pd.Series(ld_true).corr(pd.Series(ld_pred), method="spearman")
    pe_real = pd.Series(ld_true).corr(pd.Series(ld_pred), method="pearson")

    ax2.scatter(ld_true, ld_pred, color='#7570b3', alpha=0.35, s=18, edgecolors='none')
    m_fit, b_fit = np.polyfit(ld_true, ld_pred, 1)
    x_line = np.linspace(min(ld_true), max(ld_true), 100)
    ax2.plot(x_line, m_fit * x_line + b_fit, 'r-', lw=2.0, 
             label=f"10-fold OOF Fit (R² = {r2_real:.3f}, ρ = {sp_real:.3f}, r = {pe_real:.3f})")

    ax2.set_xlabel("True log(Nodule Diameter, mm)")
    ax2.set_ylabel("FMCIB OOF Decoded log(Diameter, mm)")
    ax2.set_title("(b) FMCIB Implicitly Encodes Caliber (10-fold OOF)")
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "Fig3_Scorer_Pareto.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_fig4():
    print("[*] 正在绘制 Figure 4 (PAC 认证切点与双指南对齐)...")
    df_curves = pd.read_csv(os.path.join(RESULT_DIR, "04_guideline_curves.csv"))
    df_pac = pd.read_csv(os.path.join(RESULT_DIR, "04_pac_cutoffs.csv"))

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    colors = {"any": "#d95f02", "mean": "#7570b3", "majority": "#1b9e77"}
    labels = {"any": "Any Reader Suspicion", "mean": "Mean Reader Suspicion", "majority": "Majority Vote Binary"}

    for name in ("any", "mean", "majority"):
        sub = df_curves[df_curves.definition == name].sort_values("cutoff")
        ax.plot(sub.cutoff, sub.risk_real, color=colors[name], lw=2.2, label=labels[name])
        ax.fill_between(sub.cutoff, sub.ci_lo, sub.ci_hi, color=colors[name], alpha=0.15)

    pac_005 = df_pac[df_pac.alpha == 0.05].set_index("definition")["mode_cutoff"].to_dict()
    
    for name, c_val in [('any', '#d95f02'), ('mean', '#7570b3'), ('majority', '#1b9e77')]:
        if not np.isnan(pac_005.get(name, np.nan)):
            ax.axvline(pac_005[name], color=c_val, linestyle='--', lw=1.5, label=f"{name.capitalize()} PAC (<{pac_005[name]:.1f} mm)")

    # 修正审稿人意见：指南双线准确对齐
    ax.axvline(6.0, color='black', linestyle=':', lw=1.8)
    ax.text(6.08, 0.42, "6.0 mm (Lung-RADS Cat 2 &\nFleischner low-risk lower bound)", 
            rotation=0, fontsize=8.0, weight='bold', color='black',
            bbox=dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor="black", lw=0.8, alpha=0.9))
    
    ax.axvline(8.0, color='black', linestyle=':', lw=1.8)
    ax.text(8.08, 0.35, "8.0 mm Escalation Boundary\n(Both Guidelines: Short-interval/PET)", 
            rotation=0, fontsize=8.0, weight='bold', color='black',
            bbox=dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor="black", lw=0.8, alpha=0.9))

    ax.axhline(0.05, color='red', linestyle='-.', lw=1.2, label='Nominal α = 5%')
    ax.axhline(0.10, color='gray', linestyle='-.', lw=1.0, label='Nominal α = 10%')

    ax.set_xlim(4.0, 13.0)
    ax.set_ylim(0, 0.72)
    ax.set_xlabel("Nodule Physical Caliber Threshold (mm)")
    ax.set_ylabel("Empirical Ratio Risk (Missed Suspicion pw-FNR)")
    ax.set_title("Physical Caliber Benchmark & Finite-Sample PAC Certification", pad=12)
    ax.legend(loc='upper left', framealpha=0.9, fontsize=8.5)
    ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "Fig4_Guideline_Cutoffs.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_fig5():
    print("[*] 正在绘制 Figure 5 (双向瀑布加法分解与协议迁移)...")
    df_sum = pd.read_csv(os.path.join(RESULT_DIR, "03_shift_summary.csv"))
    sub_005 = df_sum[df_sum.alpha == 0.05].set_index("方向")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))

    r_src = sub_005.loc["厚→薄 (违约压力测试)", "R_src"]
    r_raw = sub_005.loc["厚→薄 (违约压力测试)", "R_raw"]
    r_diag = sub_005.loc["厚→薄 (违约压力测试)", "R_diag"]
    r_rew = sub_005.loc["厚→薄 (违约压力测试)", "R_rew"]
    r_tar = sub_005.loc["厚→薄 (违约压力测试)", "R_tar"]
    
    d_cond = r_diag - r_src
    d_cov = r_raw - r_diag
    r_excess = r_raw - r_src
    cond_share = d_cond / r_excess if r_excess > 0.01 else 0.0
    cov_share = d_cov / r_excess if r_excess > 0.01 else 0.0

    stages = ["Source R_src", f"+ Cond Shift\n(+{d_cond:.3f})", f"+ Cov Shift\n(+{d_cov:.3f})", "Transfer R_raw", "Reweighted\n(R_rew)", "Target Local R_tar"]
    risks = [r_src, r_diag, r_raw, r_raw, r_rew, r_tar]
    colors = ['#377eb8', '#e41a1c', '#ff7f00', '#e41a1c', '#984ea3', '#4daf4a']

    ax1.bar(range(len(stages)), risks, color=colors, alpha=0.85, edgecolor='black', lw=0.8)
    ax1.axhline(0.05, color='black', linestyle='--', lw=1.5, label='Tolerance α=5%')
    ax1.set_ylabel("Evaluated Risk on Thin Slices")
    ax1.set_title("(a) Additive Risk Waterfall (Thick → Thin, α=0.05)")
    ax1.set_xticks(range(len(stages)))
    ax1.set_xticklabels(stages, rotation=30, ha='right', fontsize=8.5)
    ax1.legend(loc='upper left')
    ax1.set_ylim(0, max(risks) * 1.30)
    
    ax1.text(1, r_diag + 0.004, f"{cond_share:.1%} Cond", ha='center', fontsize=8, weight='bold', color='#e41a1c')
    ax1.text(2, r_raw + 0.004, f"{cov_share:.1%} Cov", ha='center', fontsize=8, weight='bold', color='#ff7f00')

    sign_raw_tt = sub_005.loc["厚→薄 (违约压力测试)", "直签_raw"] * 100
    sign_tar_tt = sub_005.loc["厚→薄 (违约压力测试)", "直签_tar"] * 100
    sign_raw_tb = sub_005.loc["薄→厚 (保守测试)", "直签_raw"] * 100
    sign_tar_tb = sub_005.loc["薄→厚 (保守测试)", "直签_tar"] * 100

    categories = ["Thick → Thin\n(Stress Test)", "Thin → Thick\n(Conservative Test)"]
    x = np.arange(len(categories))
    w = 0.35
    ax2.bar(x - w/2, [sign_raw_tt, sign_raw_tb], w, label='Raw Uncalibrated Transfer', color='#e41a1c', alpha=0.85, edgecolor='black')
    ax2.bar(x + w/2, [sign_tar_tt, sign_tar_tb], w, label='Target Locally Calibrated', color='#4daf4a', alpha=0.85, edgecolor='black')

    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.set_ylabel("Autonomous Sign-off Rate (%)")
    ax2.set_title("(b) Bidirectional Protocol Asymmetry")
    ax2.legend(loc='upper right')
    max_h = max([sign_raw_tt, sign_tar_tt, sign_raw_tb, sign_tar_tb])
    ax2.set_ylim(0, max_h * 1.35)

    r_raw_tt = sub_005.loc['厚→薄 (违约压力测试)', 'R_raw']
    v_raw_tt = sub_005.loc['厚→薄 (违约压力测试)', 'raw_违约'] * 100
    r_tar_tt = sub_005.loc['厚→薄 (违约压力测试)', 'R_tar']
    v_tar_tt = sub_005.loc['厚→薄 (违约压力测试)', 'tar_违约'] * 100

    r_raw_tb = sub_005.loc['薄→厚 (保守测试)', 'R_raw']
    r_tar_tb = sub_005.loc['薄→厚 (保守测试)', 'R_tar']
    v_tar_tb = sub_005.loc['薄→厚 (保守测试)', 'tar_违约'] * 100

    # 上移文字 padding 防止重叠
    pad = max_h * 0.04
    ax2.text(0 - w/2, sign_raw_tt + pad, f"Raw R={r_raw_tt:.3f}\n(Vio {v_raw_tt:.1f}%)", ha='center', fontsize=7.5, color='#d62728', weight='bold')
    ax2.text(0 + w/2, sign_tar_tt + pad, f"Local R={r_tar_tt:.3f}\n(Vio {v_tar_tt:.1f}%)", ha='center', fontsize=7.5, color='#2ca02c', weight='bold')

    ax2.text(1 - w/2, sign_raw_tb + pad, f"Raw R={r_raw_tb:.3f}\n(Vio 0.0%)", ha='center', fontsize=7.5, color='#d62728', weight='bold')
    ax2.text(1 + w/2, sign_tar_tb + pad, f"Local R={r_tar_tb:.3f}\n(Vio {v_tar_tb:.1f}%)", ha='center', fontsize=7.5, color='#2ca02c', weight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "Fig5_Protocol_Shift.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_fig6():
    print("[*] 正在绘制 Figure 6 (大样本渐近紧致性与 PAC 税)...")
    df_sim = pd.read_csv(os.path.join(RESULT_DIR, "05_simulation_summary.csv"))
    
    crc_300 = df_sim[df_sim.method == "CRC (n=300)"].sort_values("alpha")["总体风险均值"].values
    crc_3000 = df_sim[df_sim.method == "CRC (n=3000)"].sort_values("alpha")["总体风险均值"].values
    sign_crc = df_sim[df_sim.method == "CRC (n=300)"].sort_values("alpha")["直签吞吐率"].values * 100
    sign_ltt = df_sim[df_sim.method == "LTT (n=300)"].sort_values("alpha")["直签吞吐率"].values * 100

    vio_crc = df_sim.loc[df_sim.method == "CRC (n=300)", "真实违约概率"].values * 100
    vio_ltt = df_sim.loc[df_sim.method == "LTT (n=300)", "真实违约概率"].values * 100

    alphas = np.array([0.05, 0.10, 0.20])
    gap_300 = np.mean(np.abs(alphas - crc_300))
    gap_3000 = np.mean(np.abs(alphas - crc_3000))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.5))

    ax1.plot(alphas, alphas, 'k--', lw=1.5, label='Nominal Target α')
    ax1.plot(alphas, crc_300, 'o-', color='#377eb8', lw=1.8, label='CRC Small Sample (n=300)')
    ax1.plot(alphas, crc_3000, 's-', color='#e41a1c', lw=2.0, label='CRC Asymptotically Tight (n=3000)')

    ax1.set_xlabel("Nominal Risk Tolerance (α)")
    ax1.set_ylabel("Empirical Mean Risk on N=200,000")
    ax1.set_title(f"(a) CRC Convergence (Mean Gap: {gap_300:.4f} → {gap_3000:.4f})")
    ax1.set_xticks(alphas)
    ax1.set_xticklabels(["0.05", "0.10", "0.20"])
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.6)

    penalties = (sign_crc - sign_ltt) / sign_crc * 100
    x = np.arange(len(alphas))
    w = 0.35
    
    lab_crc = f"CRC (Expectation Bound, obs. Vio {vio_crc.min():.0f}%–{vio_crc.max():.0f}%)"
    lab_ltt = f"LTT (PAC δ=0.10, obs. Vio {vio_ltt.min():.0f}%–{vio_ltt.max():.0f}%)"
    
    ax2.bar(x - w/2, sign_crc, w, label=lab_crc, color='#377eb8', alpha=0.85, edgecolor='black')
    ax2.bar(x + w/2, sign_ltt, w, label=lab_ltt, color='#ff7f0e', alpha=0.85, edgecolor='black')

    ax2.set_xticks(x)
    ax2.set_xticklabels(["α = 0.05", "α = 0.10", "α = 0.20"])
    ax2.set_ylabel("Autonomous Sign-off Rate (%)")
    ax2.set_title("(b) The 'PAC Tax': Throughput Cost of Bounding Violations")
    ax2.legend(loc='lower right')
    ax2.set_ylim(0, max(np.concatenate([sign_crc, sign_ltt])) * 1.3)

    for i in range(len(alphas)):
        ax2.text(i + w/2, sign_ltt[i] + 2.0, f"-{penalties[i]:.1f}%", ha='center', fontsize=8.5, weight='bold', color='#d62728')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "Fig6_PAC_Tradeoff.png"), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    plot_fig2()
    plot_fig3()
    plot_fig4()
    plot_fig5()
    plot_fig6()
    print(f"\n[+] 全部图表重新绘制生成完毕！保存于: {FIG_DIR}")