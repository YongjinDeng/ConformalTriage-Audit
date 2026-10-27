# -*- coding: utf-8 -*-
"""
10_external_validation_lndb.py:
基于 LNDb 真实多读者标注数据 (rad2Fleischner.csv 与 chars_trainNodules.csv) 的外部独立验证
完全基于真实临床元数据，无任何模拟造数。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from rcrc_core import compute_crc_lambda, calculate_risk, RESULT_DIR, BASE_DIR

DATA_DIR = os.path.join(BASE_DIR, "data")
RAD_FILE = os.path.join(DATA_DIR, "rad2Fleischner.csv")
CHARS_FILE = os.path.join(DATA_DIR, "chars_trainNodules.csv")


def run_lndb_external_audit(n_boot=500):
    print("\n" + "=" * 80)
    print(">>> 正在运行 10: LNDb 真实外部独立多读者队列验证 (基于实际 CSV 标注)")
    print("=" * 80)

    if not os.path.exists(RAD_FILE) or not os.path.exists(CHARS_FILE):
        raise FileNotFoundError(
            f"[-] 严重错误: 未在 {DATA_DIR} 找到真实的 LNDb 数据文件！\n"
            f"    请确保该目录下存在 rad2Fleischner.csv 与 chars_trainNodules.csv。"
        )

    # 1. 加载并清洗结节体积与直径
    df_rad = pd.read_csv(RAD_FILE)
    df_rad = df_rad[df_rad.Nodule == 1].copy()
    df_rad["LNDbID"] = df_rad["LNDbID"].astype(int)
    df_rad["FindingID"] = df_rad["FindingID"].astype(int)
    
    # 使用显式等体积球直径公式: D = 2 * (3 * V / (4 * pi)) ** (1/3)
    df_rad["diam"] = 2.0 * (3.0 * df_rad["Volume"].astype(float) / (4.0 * np.pi)) ** (1.0 / 3.0)
    df_rad["texture"] = pd.to_numeric(df_rad["Text"], errors="coerce")

    # 2. 加载多读者评分
    df_chars = pd.read_csv(CHARS_FILE)
    df_chars["LNDbID"] = df_chars["LNDbID"].astype(int)
    df_chars["FindingID"] = df_chars["FindingID"].astype(int)
    df_chars = df_chars[df_chars.malignancy > 0].copy()

    # 按病灶 (LNDbID, FindingID) 聚合多读者打分 (可疑阈值: malignancy >= 4)
    def agg_readers(group):
        mal = group["malignancy"].values
        n_readers = len(mal)
        susp_binary = (mal >= 4).astype(float)
        return pd.Series({
            "n_readers": n_readers,
            "y_any": float(np.any(mal >= 4)),
            "y_maj": float(np.mean(susp_binary) >= 0.5),
            "y_mean": float(np.mean(susp_binary))
        })

    df_ratings = df_chars.groupby(["LNDbID", "FindingID"]).apply(agg_readers).reset_index()

    # 3. 合并直径与多读者标签
    df_nodules = df_rad.merge(df_ratings, on=["LNDbID", "FindingID"], how="inner")
    n_nodules = len(df_nodules)
    n_patients = df_nodules.LNDbID.nunique()

    print(f"[*] 成功匹配 LNDb 多读者有效结节: {n_nodules} 颗，涵盖 {n_patients} 例患者")
    print(f"[*] 读者人数分布: 均值 {df_nodules.n_readers.mean():.1f} 人/结节 (范围: {df_nodules.n_readers.min()} - {df_nodules.n_readers.max()})")
    print(f"[*] 结节等效直径: 中位数 {df_nodules.diam.median():.2f} mm (IQR: {df_nodules.diam.quantile(0.25):.2f} - {df_nodules.diam.quantile(0.75):.2f} mm)")
    print(f"[*] 可疑病灶占比: Any 策略 = {df_nodules.y_any.mean():.1%} | Majority 策略 = {df_nodules.y_maj.mean():.1%}")

    # 计算患者级聚类权重
    n_p = df_nodules.groupby("LNDbID").LNDbID.transform("size").values
    w_any = df_nodules.y_any.values / n_p
    w_maj = df_nodules.y_maj.values / n_p
    diam = df_nodules.diam.values
    pid = df_nodules.LNDbID.values

    # ------------------------------------------------------------------
    # 审计 1: 验证 LIDC 导出的 PAC 切点在外部队列的零样本泛化
    # ------------------------------------------------------------------
    print("\n[审计 1: LIDC-IDRI 衍生规则在 LNDb 外部队列的零样本验证]")
    
    # 评估 <6.0 mm (LIDC Any α=0.05 认证切点)
    a_60 = diam < 6.0
    risk_any_60 = calculate_risk(w_any, a_60)
    sign_any_60 = a_60.mean()
    print(f"  -> LIDC Any 切点 (< 6.0 mm) 在 LNDb Any 目标评估下:")
    print(f"     实测漏诊风险 (pw-FNR) = {risk_any_60:.4f} (标称 α = 0.05 -> 安全受控: {risk_any_60 <= 0.05})")
    print(f"     外部结节直签率 = {sign_any_60:.1%}")

    # 评估 <7.0 mm (LIDC Majority α=0.05 认证切点)
    a_70 = diam < 7.0
    risk_maj_70_under_maj = calculate_risk(w_maj, a_70)
    risk_maj_70_under_any = calculate_risk(w_any, a_70)
    sign_maj_70 = a_70.mean()
    print(f"  -> LIDC Majority 切点 (< 7.0 mm):")
    print(f"     在 LNDb Majority 目标下评估风险 = {risk_maj_70_under_maj:.4f}")
    print(f"     在 LNDb Any 目标下评估风险      = {risk_maj_70_under_any:.4f} (跨定义膨胀倍数: {risk_maj_70_under_any/0.05:.2f}x)")
    print(f"     外部结节直签率 = {sign_maj_70:.1%}")

    # ------------------------------------------------------------------
    # 审计 2: LNDb 本地校准下的定义泄漏倍数 (70/30 划分与 Bootstrap CI)
    # ------------------------------------------------------------------
    print("\n[审计 2: LNDb 外部多阅片者环境下的共识定义泄漏重现 (70% 校准 / 30% 评估)]")
    rng = np.random.default_rng(2026)
    up = np.unique(pid)
    n_pts = len(up)
    
    leak_ratios = []
    maj_self_risks = []
    n_vacuous = 0

    for _ in range(n_boot):
        boot_p = rng.choice(up, n_pts, replace=True)
        u_pts = np.unique(boot_p)
        perm = rng.permutation(len(u_pts))
        # 采用 70% 校准，确保累计质量跨过 19.0 的非空门槛
        k = int(len(u_pts) * 0.7)
        cal_pts = u_pts[perm[:k]]
        te_pts = u_pts[perm[k:]]
        
        m_cal = np.isin(pid, cal_pts)
        m_te = np.isin(pid, te_pts)
        if m_cal.sum() < 20 or m_te.sum() < 20:
            continue
            
        lam_maj = compute_crc_lambda(diam[m_cal], w_maj[m_cal], 0.05)
        if np.isfinite(lam_maj):
            a_te = diam[m_te] < lam_maj
            r_any = calculate_risk(w_any[m_te], a_te)
            r_maj = calculate_risk(w_maj[m_te], a_te)
            leak_ratios.append(r_any / 0.05)
            maj_self_risks.append(r_maj)
        else:
            n_vacuous += 1

    valid_runs = len(leak_ratios)
    print(f"  -> 成功非空认证次数: {valid_runs}/{n_boot} (空认证率: {n_vacuous/n_boot:.1%})")

    if valid_runs > 0:
        med_leak = float(np.median(leak_ratios))
        ci_leak = np.percentile(leak_ratios, [2.5, 97.5])
        med_self = float(np.median(maj_self_risks))
        print(f"  -> LNDb 队列中 Majority 校准在 Any 目标下的泄漏倍数: 中位数 {med_leak:.2f}x (95% CI: [{ci_leak[0]:.2f}x, {ci_leak[1]:.2f}x])")
        print(f"  -> Majority 自评风险中位数: {med_self:.4f}")
    else:
        med_leak = np.nan
        ci_leak = [np.nan, np.nan]
        print("  [-] 提示: LNDb 样本量较小，在 70% 样本下多数票累计质量仍不足 19.0，符合理论推导的空认证预测。")

    # ------------------------------------------------------------------
    # 审计 3: 实性结节 (Texture >= 4) 敏感性分析
    # ------------------------------------------------------------------
    print("\n[审计 3: LNDb 实性结节 (Texture >= 4.0) 指南切点敏感性分析]")
    solid_mask = (df_nodules["texture"] >= 4.0).values
    n_solid = solid_mask.sum()
    if n_solid > 20:
        diam_solid = diam[solid_mask]
        w_any_solid = w_any[solid_mask]
        risk_solid_60 = calculate_risk(w_any_solid, diam_solid < 6.0)
        sign_solid_60 = (diam_solid < 6.0).mean()
        print(f"  -> 纯实性结节数: {n_solid} 颗 (占 {n_solid/n_nodules:.1%})")
        print(f"  -> 实施 <6.0 mm Any 规则在纯实性结节上的漏诊风险: {risk_solid_60:.4f} | 直签率: {sign_solid_60:.1%}")
    else:
        risk_solid_60 = np.nan

    # 4. 汇总结果落盘
    summary_df = pd.DataFrame([{
        "Dataset": "LNDb (CHUSJ, Portugal)",
        "N_Nodules": n_nodules,
        "N_Patients": n_patients,
        "Any_6mm_Evaluated_Risk": risk_any_60,
        "Any_6mm_Signoff_Rate": sign_any_60,
        "Maj_7mm_Evaluated_Risk_under_Any": risk_maj_70_under_any,
        "Maj_7mm_Evaluated_Risk_under_Maj": risk_maj_70_under_maj,
        "Valid_Recal_Runs": valid_runs,
        "Median_Leakage_Ratio": med_leak,
        "Leakage_CI_2.5": ci_leak[0],
        "Leakage_CI_97.5": ci_leak[1]
    }])
    out_csv = os.path.join(RESULT_DIR, "10_external_validation_lndb_summary.csv")
    summary_df.to_csv(out_csv, index=False)
    print(f"\n[+] 外部验证结果已真实落盘 -> {out_csv}")
    print("=" * 80)


if __name__ == "__main__":
    run_lndb_external_audit()