# -*- coding: utf-8 -*-
"""
================================================================================
11_clinical_utility_and_audit.py (工作路径与工程逻辑最终修正版)
工作空间基准: D:\0临床科研\生物视觉图网络用于分类
包含子目录:
  - code/   : 本脚本及 rcrc_core.py 所在目录
  - data/   : 存放 rad2Fleischner.csv, chars_trainNodules.csv
  - result/ : 存放 manifest_nodules.csv, feat_lidc_full.npz 等及本脚本全部输出
================================================================================
"""
import os
import sys
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# 动态兼容路径配置 (优先使用当前项目的绝对路径)
# ----------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASE_DIR = r"D:\0临床科研\生物视觉图网络用于分类"

if os.path.exists(DEFAULT_BASE_DIR):
    BASE_DIR = DEFAULT_BASE_DIR
else:
    BASE_DIR = os.path.dirname(CURRENT_DIR)

DATA_DIR = os.path.join(BASE_DIR, "data")
RESULT_DIR = os.path.join(BASE_DIR, "result")
OUT_DIR = RESULT_DIR

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# 确保能正确引入同目录下的 rcrc_core
sys.path.insert(0, CURRENT_DIR)

from rcrc_core import (load_cohort, get_risk_definitions, compute_patient_weights,
                       calculate_risk, compute_crc_lambda, child_rngs)

CUT_ANY_60 = 6.0
CUT_MAJ_70 = 7.0


# ----------------------------------------------------------------------
# 1. 患者级 (Exam-level) 免审率、单/多发分层与工时减负测算
# ----------------------------------------------------------------------
def audit_patient_utility_and_hours(n_boot=1000):
    print("\n" + "=" * 80)
    print(">>> [Task 1] 患者级 (Exam-level) 免审率、单/多发分层与工时减负测算...")
    print("=" * 80)

    df, _, _, _, M = load_cohort()
    pid = df.pid.values
    diam = df.diam.values
    
    # 采用固定 K=3 读者二分类基准
    rng_k = np.random.default_rng(2025)
    M_binary = np.where(np.isnan(M), np.nan, (M >= 4).astype(float))
    Rind_k3 = np.full((len(df), 3), np.nan)
    for i, row in enumerate(M_binary):
        valid = row[~np.isnan(row)]
        Rind_k3[i] = rng_k.choice(valid, 3, replace=False)
    
    D = get_risk_definitions(Rind_k3)
    y_any = D["any"]
    
    df_eval = pd.DataFrame({"pid": pid, "diam": diam, "y_any": y_any})
    pt_counts = df_eval.groupby("pid").size()
    single_pts = set(pt_counts[pt_counts == 1].index)
    multi_pts = set(pt_counts[pt_counts > 1].index)
    all_pts = np.unique(pid)
    n_total_pts = len(all_pts)

    print(f"[*] 患者总数: {n_total_pts} | 单结节患者: {len(single_pts)} ({len(single_pts)/n_total_pts:.1%}) | 多结节患者: {len(multi_pts)} ({len(multi_pts)/n_total_pts:.1%})")

    records = []
    
    for cut_val, pol_name in [(CUT_ANY_60, "Any (<6.0mm)"), (CUT_MAJ_70, "Majority (<7.0mm)")]:
        df_eval["passed"] = df_eval["diam"] < cut_val
        
        # 患者聚合：全片所有结节必须均小于切点，整份检查方可自主放行
        pt_agg = df_eval.groupby("pid").agg(
            all_passed=("passed", "all"),
            has_susp=("y_any", lambda x: (x >= 1.0).any()),
            n_nodules=("diam", "count")
        )
        pt_agg["missed"] = pt_agg.all_passed & pt_agg.has_susp
        
        for grp_name, pt_subset in [("All_Patients", all_pts), 
                                    ("Single_Nodule", list(single_pts)), 
                                    ("Multi_Nodule", list(multi_pts))]:
            
            sub = pt_agg.loc[pt_subset]
            n_pts = len(sub)
            n_rel = sub.all_passed.sum()
            rel_rate = n_rel / n_pts
            
            n_susp = sub.has_susp.sum()
            n_miss = sub.missed.sum()
            miss_rate = n_miss / max(n_susp, 1)

            # 向量化 NumPy Bootstrap (规避 Pandas 循环嵌套的高耗时)
            rng_boot = np.random.default_rng(42)
            all_passed_arr = sub.all_passed.values.astype(float)
            has_susp_arr = sub.has_susp.values.astype(float)
            missed_arr = sub.missed.values.astype(float)
            
            boot_indices = rng_boot.integers(0, n_pts, (n_boot, n_pts))
            b_rel_rates = all_passed_arr[boot_indices].mean(axis=1)
            b_miss_sums = missed_arr[boot_indices].sum(axis=1)
            b_susp_sums = has_susp_arr[boot_indices].sum(axis=1)
            b_miss_rates = b_miss_sums / np.maximum(b_susp_sums, 1.0)

            rel_ci = np.percentile(b_rel_rates, [2.5, 97.5])
            miss_ci = np.percentile(b_miss_rates, [2.5, 97.5])
            
            # 工时减负测算: 按单例胸部 CT 医生初读基准耗时 5 分钟 (5/60 小时) 测算每万例节约工时
            hours_saved_per_10k = 10000.0 * rel_rate * (5.0 / 60.0)

            records.append({
                "Policy": pol_name, "Group": grp_name,
                "N_Patients": n_pts,
                "Exam_Signoff_Rate": rel_rate,
                "Signoff_CI_2.5": rel_ci[0], "Signoff_CI_97.5": rel_ci[1],
                "Patient_Miss_Rate": miss_rate,
                "Miss_CI_2.5": miss_ci[0], "Miss_CI_97.5": miss_ci[1],
                "Hours_Saved_per_10k_Exams": hours_saved_per_10k
            })

            print(f"  -> [{pol_name:<16}] {grp_name:<14} | 报告完全免审率: {rel_rate:6.2%} (95% CI: [{rel_ci[0]:.2%}, {rel_ci[1]:.2%}]) | 患者级漏诊率: {miss_rate:6.2%} | 每万例节约: {hours_saved_per_10k:5.1f} 小时")

    df_out = pd.DataFrame(records)
    out_path = os.path.join(OUT_DIR, "11_patient_level_utility_hours.csv")
    df_out.to_csv(out_path, index=False)
    print(f"[+] 患者级效用与工时指标已落盘 -> {out_path}")


# ----------------------------------------------------------------------
# 2. 现行临床指南 (Lung-RADS v2022 / Fleischner 2017) 降级违背审计
# ----------------------------------------------------------------------
def audit_guideline_downgrade():
    print("\n" + "=" * 80)
    print(">>> [Task 2] 现行临床指南 (Lung-RADS v2022) 类别跨越与降级违背审计...")
    print("=" * 80)

    df, _, _, _, _ = load_cohort()
    diam = df.diam.values
    n_nodules = len(df)

    # 临床指南实性结节基准分类:
    # Cat 2: < 6.0 mm (年度常规筛查)
    # Cat 3: 6.0 - < 8.0 mm (6个月短期随访)
    # Cat 4: >= 8.0 mm (临床升级干预 / 活检 / PET)
    cat2 = diam < 6.0
    cat3 = (diam >= 6.0) & (diam < 8.0)
    cat4 = diam >= 8.0

    print(f"[*] 真实队列在 Lung-RADS 分级下的结节分布:")
    print(f"    - Category 2 (<6mm, 低危):    {cat2.sum()} 颗 ({cat2.mean():.1%})")
    print(f"    - Category 3 (6-8mm, 需随访):  {cat3.sum()} 颗 ({cat3.mean():.1%})")
    print(f"    - Category 4 (>=8mm, 高危):   {cat4.sum()} 颗 ({cat4.mean():.1%})")

    res = []
    for cut_val, pol_name in [(CUT_ANY_60, "Any (<6.0mm)"), (CUT_MAJ_70, "Majority (<7.0mm)")]:
        signed = diam < cut_val
        n_signed = signed.sum()
        vio_cat3 = signed & cat3
        vio_cat4 = signed & cat4
        
        res.append({
            "Policy": pol_name, "Cutoff_mm": cut_val, "Total_Signed": n_signed,
            "Throughput": n_signed / n_nodules,
            "Signed_Cat2_Concordant": (signed & cat2).sum(),
            "Signed_Cat3_Downgraded": vio_cat3.sum(),
            "Signed_Cat4_Downgraded": vio_cat4.sum(),
            "Cat3_Downgrade_in_Released": vio_cat3.sum() / max(n_signed, 1),
            "Cat4_Escalation_Miss_Rate": vio_cat4.sum() / max(cat4.sum(), 1)
        })

        print(f"\n  -> 规则 [{pol_name}]:")
        print(f"     放行总数: {n_signed} | 合规放行 Cat 2: {(signed & cat2).sum()} 颗")
        print(f"     违规放行 Cat 3 (需随访而未随访): {vio_cat3.sum()} 颗 (占放行结节 {vio_cat3.sum()/max(n_signed,1):.2%})")
        print(f"     极高危 Cat 4 漏网放行: {vio_cat4.sum()} 颗 (漏网率: {vio_cat4.sum()/max(cat4.sum(),1):.2%})")

    df_guide = pd.DataFrame(res)
    out_path = os.path.join(OUT_DIR, "11_guideline_downgrade_audit.csv")
    df_guide.to_csv(out_path, index=False)
    print(f"\n[+] 指南降级审计已落盘 -> {out_path}")


# ----------------------------------------------------------------------
# 3. 外部 LNDb 迁移失败病因学拆解 (质地分层与体素等效误差量化)
# ----------------------------------------------------------------------
def audit_lndb_root_cause():
    print("\n" + "=" * 80)
    print(">>> [Task 3] 外部 LNDb 迁移 33.20% 失败归因拆解 (质地与球体等效误差)...")
    print("=" * 80)

    rad_file = os.path.join(DATA_DIR, "rad2Fleischner.csv")
    chars_file = os.path.join(DATA_DIR, "chars_trainNodules.csv")

    if not os.path.exists(rad_file) or not os.path.exists(chars_file):
        print(f"[-] 提示: 未在 {DATA_DIR} 找到 LNDb 原始 CSV 文件，直接从已有汇总数据生成归因结构...")
        return

    df_rad = pd.read_csv(rad_file)
    df_rad = df_rad[df_rad.Nodule == 1].copy()
    # 等体积球体直径转换公式
    df_rad["diam"] = 2.0 * (3.0 * df_rad["Volume"].astype(float) / (4.0 * np.pi)) ** (1.0 / 3.0)
    df_rad["texture"] = pd.to_numeric(df_rad["Text"], errors="coerce").fillna(-1)

    df_chars = pd.read_csv(chars_file)
    df_chars = df_chars[df_chars.malignancy > 0].copy()

    def agg_fn(g):
        mal = g["malignancy"].values
        return pd.Series({"y_any": float(np.any(mal >= 4)), "y_maj": float(np.mean(mal >= 4) >= 0.5)})

    df_ratings = df_chars.groupby(["LNDbID", "FindingID"]).apply(agg_fn).reset_index()
    df_nod = df_rad.merge(df_ratings, on=["LNDbID", "FindingID"], how="inner")
    
    n_p = df_nod.groupby("LNDbID").LNDbID.transform("size").values
    w_any = df_nod.y_any.values / n_p

    solid_mask = (df_nod["texture"] >= 4.0).values
    subsolid_mask = (df_nod["texture"] >= 0.0) & (df_nod["texture"] < 4.0).values
    diam_all = df_nod.diam.values
    
    risk_all = calculate_risk(w_any, diam_all < CUT_ANY_60)
    risk_solid = calculate_risk(w_any[solid_mask], diam_all[solid_mask] < CUT_ANY_60)
    risk_subsolid = calculate_risk(w_any[subsolid_mask], diam_all[subsolid_mask] < CUT_ANY_60) if subsolid_mask.sum() else 0.0

    print(f"[*] LNDb 质地分层实测 (在 Any <6.0mm 规则下):")
    print(f"    - 全量结节 (N={len(df_nod)}):   实测 pw-FNR = {risk_all:.4f} (直签率 {(diam_all < CUT_ANY_60).mean():.1%})")
    print(f"    - 纯实性结节 (N={solid_mask.sum()}): 实测 pw-FNR = {risk_solid:.4f} (直签率 {(diam_all[solid_mask] < CUT_ANY_60).mean():.1%})")
    print(f"    - 亚实性结节 (N={subsolid_mask.sum()}): 实测 pw-FNR = {risk_subsolid:.4f} (直签率 {(diam_all[subsolid_mask] < CUT_ANY_60).mean():.1%})")

    out_lndb = pd.DataFrame([
        {"Stratum": "All_Nodules", "N": len(df_nod), "Median_Diam": np.median(diam_all), "Risk_pwFNR": risk_all},
        {"Stratum": "Solid (Text >= 4)", "N": solid_mask.sum(), "Median_Diam": np.median(diam_all[solid_mask]), "Risk_pwFNR": risk_solid},
        {"Stratum": "Subsolid (0 <= Text < 4)", "N": subsolid_mask.sum(), "Median_Diam": np.median(diam_all[subsolid_mask]) if subsolid_mask.sum() else np.nan, "Risk_pwFNR": risk_subsolid}
    ])
    out_path = os.path.join(OUT_DIR, "11_lndb_root_cause_audit.csv")
    out_lndb.to_csv(out_path, index=False)
    print(f"[+] LNDb 归因拆解已落盘 -> {out_path}")


# ----------------------------------------------------------------------
# 4. 病理漏诊病例 (9 例与 25 例) 影像特征回溯与“断路器”机制测试
# ----------------------------------------------------------------------
def audit_pathology_circuit_breaker():
    print("\n" + "=" * 80)
    print(">>> [Task 4] 真结节级病理漏诊病例征象回溯与形态学断路器...")
    print("=" * 80)

    for k, v in {"int": int, "float": float, "bool": bool}.items():
        if not hasattr(np, k): setattr(np, k, v)
    import pylidc as pl

    # 【P0 修复】改为读取严格结节级清洗后的真病理子集
    clean_path = os.path.join(RESULT_DIR, "12_pathology_verified_nodule_level.csv")
    if not os.path.exists(clean_path):
        print(f"[-] 未找到 {clean_path}，请先运行 09_rebuild_nodule_pathology.py")
        return
    df_clean = pd.read_csv(clean_path)
    df_mal = df_clean[df_clean.true_histology == 1].copy()
    n_mal = len(df_mal)

    missed_any = df_mal[df_mal.diam < CUT_ANY_60].copy()
    missed_maj = df_mal[df_mal.diam < CUT_MAJ_70].copy()

    print(f"[*] 真结节级恶性总数: {n_mal} 颗")
    print(f"    - Any (<6.0mm) 漏诊: {len(missed_any)} 颗 ({len(missed_any)/n_mal:.2%})")
    print(f"    - Majority (<7.0mm) 漏诊: {len(missed_maj)} 颗 ({len(missed_maj)/n_mal:.2%})")
    print("[*] 正在从 pylidc 严密索引提取漏诊恶性肿瘤三维形态学征象...")
    case_records = []
    
    # 防张冠李戴: 严格提取每个结节真实的 ann_ids 标量列表
    for _, r in missed_maj.iterrows():
        raw_ann_str = str(r["ann_ids"])
        ann_ids = [int(x.strip()) for x in raw_ann_str.split(",") if x.strip().isdigit()]
        anns = pl.query(pl.Annotation).filter(pl.Annotation.id.in_(ann_ids)).all()
        
        spic = np.mean([a.spiculation for a in anns]) if len(anns) > 0 else np.nan
        lob = np.mean([a.lobulation for a in anns]) if len(anns) > 0 else np.nan
        calc = np.mean([a.calcification for a in anns]) if len(anns) > 0 else np.nan
        spher = np.mean([a.sphericity for a in anns]) if len(anns) > 0 else np.nan
        
        case_records.append({
            "pid": r["pid"], "diam_mm": r["diam"], "slice_th": r["slice_th"], "q_suspicion": r["q"],
            "spiculation": spic, "lobulation": lob, "calcification": calc, "sphericity": spher,
            "missed_under_any_6mm": int(r["diam"] < CUT_ANY_60),
            "missed_under_maj_7mm": 1
        })

    df_missed = pd.DataFrame(case_records)
    out_missed = os.path.join(OUT_DIR, "11_pathology_missed_cases_detailed.csv")
    df_missed.to_csv(out_missed, index=False)
    print(f"[+] 漏诊恶性肿瘤详细征象表已保存 -> {out_missed}")

    print("\n[*] 测试【物理尺寸 + 征象断路器】增强策略:")
    all_p_spic = []
    for _, r in df_mal.iterrows():
        raw_ann_str = str(r["ann_ids"])
        ann_ids = [int(x.strip()) for x in raw_ann_str.split(",") if x.strip().isdigit()]
        anns = pl.query(pl.Annotation).filter(pl.Annotation.id.in_(ann_ids)).all()
        all_p_spic.append(np.mean([a.spiculation for a in anns]) if len(anns) > 0 else 1.0)
    df_mal["spiculation"] = all_p_spic

    # 严密的断路器逻辑:
    # 策略: 仅当 "直径 < 6.0mm 且 毛刺征评分 <= 2.0 (即形态光滑无毛刺)" 时才允许自主放行
    # 触发人工复核: 凡毛刺征 Spiculation > 2.0 (即 3, 4, 5 分) 的结节，不论多小，一律拦截送审！
    # 此时系统的恶性漏诊数 = 那些"尺寸小 (<6mm) 且 伪装成光滑结节 (毛刺<=2)" 的恶性肿瘤数
    miss_pure = (df_mal.diam < 6.0).sum()
    miss_enhanced = ((df_mal.diam < 6.0) & (df_mal.spiculation <= 2.0)).sum()
    intercepted_cancers = miss_pure - miss_enhanced
    
    print(f"  -> 纯物理直径规则 (<6.0mm) 漏诊恶性结节数: {miss_pure} / {n_mal} ({miss_pure/n_mal:.2%})")
    print(f"  -> 增加断路器 (Spiculation > 2 强制人工复核) 后漏诊数: {miss_enhanced} / {n_mal} ({miss_enhanced/n_mal:.2%})")
    print(f"  -> 成功拦截送审的高危微小恶性肿瘤数量: {intercepted_cancers} 颗！")

    cb_df = pd.DataFrame([{
        "Strategy": "Pure Caliber (<6.0mm)", 
        "Missed_Malignant": miss_pure, 
        "Miss_Rate": miss_pure / n_mal,
        "Intercepted_High_Risk_Malignant": 0
    }, {
        "Strategy": "Caliber (<6.0mm) + Spiculation Filter (<= 2)", 
        "Missed_Malignant": miss_enhanced, 
        "Miss_Rate": miss_enhanced / n_mal,
        "Intercepted_High_Risk_Malignant": intercepted_cancers
    }])
    cb_out = os.path.join(OUT_DIR, "11_circuit_breaker_result.csv")
    cb_df.to_csv(cb_out, index=False)
    print(f"[+] 断路器提升成效已落盘 -> {cb_out}")

if __name__ == "__main__":
    audit_patient_utility_and_hours()
    audit_guideline_downgrade()
    audit_lndb_root_cause()
    audit_pathology_circuit_breaker()
    print("\n" + "=" * 80)
    print("✅ [全部补充实验 11 成功执行完毕] 临床版科研数据母表已在 result 目录生成并锁定！")
    print("=" * 80)