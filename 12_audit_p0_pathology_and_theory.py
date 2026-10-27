# -*- coding: utf-8 -*-
"""
================================================================================
audit_p0_pathology_and_theory.py:
P0 级核心证据审计 —— 病理标签逐结节溯源与 LTT 数学有效性单元测试
工作路径: D:\0临床科研\生物视觉图网络用于分类
================================================================================
"""
import os
import sys
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASE_DIR = r"D:\0临床科研\生物视觉图网络用于分类"
BASE_DIR = DEFAULT_BASE_DIR if os.path.exists(DEFAULT_BASE_DIR) else os.path.dirname(CURRENT_DIR)

RESULT_DIR = os.path.join(BASE_DIR, "result")
DIAG_XLS = os.path.join(RESULT_DIR, "tcia-diagnosis-data-2012-04-20.xls")
MANIFEST_CSV = os.path.join(RESULT_DIR, "manifest_nodules.csv")

sys.path.insert(0, CURRENT_DIR)
from rcrc_core import load_cohort, evalue_martingale, compute_crc_lambda, compute_ltt_lambda


# ======================================================================
# 任务 1: TCIA 病理元数据与结节归属法医级穿透
# ======================================================================
def audit_tcia_pathology_source():
    print("\n" + "=" * 80)
    print("🔍 [P0 任务 1] 穿透审计: TCIA 病理诊断表的粒度与标签污染检查")
    print("=" * 80)

    if not os.path.exists(DIAG_XLS):
        print(f"[-] 错误: 未在 {DIAG_XLS} 找到 TCIA 诊断表！")
        return

    import xlrd
    wb = xlrd.open_workbook(DIAG_XLS)
    print(f"[*] TCIA 诊断 Excel 包含的 Sheet 列表: {wb.sheet_names()}")

    sh = wb.sheet_by_index(0)
    print(f"[*] Sheet 0 名称: '{sh.name}', 总行数: {sh.nrows}, 总列数: {sh.ncols}")
    
    headers = [str(sh.cell_value(0, c)).strip() for c in range(sh.ncols)]
    print(f"[*] 表头字段: {headers}")

    # 打印前 5 行样本，确认是否包含结节空间坐标/解剖位置
    print("\n[*] 前 3 行原始数据样例:")
    for r in range(1, min(4, sh.nrows)):
        print(f"    行 {r}: {sh.row_values(r)}")

    # 读取 manifest_nodules.csv
    df_man = pd.read_csv(MANIFEST_CSV)
    df_p = df_man[df_man["pathology"].isin([0, 1])].copy()
    
    print("\n" + "-" * 50)
    print(f"[*] manifest_nodules.csv 中被标为病理确诊的病灶总数: {len(df_p)}")
    print(f"[*] 涵盖的唯一患者数: {df_p.pid.nunique()}")
    
    # 统计每个患者体内的结节数分布
    nod_per_pt = df_p.groupby("pid").size()
    single_nod_pts = nod_per_pt[nod_per_pt == 1].index
    multi_nod_pts = nod_per_pt[nod_per_pt > 1].index

    print(f"    - 单结节确诊患者数: {len(single_nod_pts)} (对应结节: {len(single_nod_pts)} 颗)")
    print(f"    - 多结节确诊患者数: {len(multi_nod_pts)} (对应结节: {nod_per_pt[multi_nod_pts].sum()} 颗)")
    
    # 重点检查那 9 例被放行的恶性结节
    df_mal = df_p[df_p.pathology == 1].copy()
    missed_9 = df_mal[df_mal.diam < 6.0].copy()
    
    print("\n" + "-" * 50)
    print("🚨 [深度解剖] 那 9 例在 <6.0mm 下被漏诊的“恶性结节”其宿主患者情况:")
    for _, r in missed_9.iterrows():
        pt_id = r["pid"]
        total_nods_in_pt = (df_man.pid == pt_id).sum()
        all_diams = df_man[df_man.pid == pt_id].diam.values
        print(f"    患者 {pt_id}: 本结节直径 = {r['diam']:.2f}mm | 该患者在 LIDC 中共有 {total_nods_in_pt} 颗结节 | 所有结节尺寸: {np.round(all_diams, 1)} mm")

    print("\n[*] 审计提示: 如果患者体内有大结节（如 >15mm）同时有这颗小结节，那么小结节继承患者恶性诊断属于【生态学谬误】！")


# ======================================================================
# 任务 2: LTT 投注鞅数学有效性模拟测试 (Null Hypothesis Calibration)
# ======================================================================
def audit_martingale_theory():
    print("\n" + "=" * 80)
    print("🧪 [P0 任务 2] 数学审计: Kelly 投注鞅与置换 e-value 在原假设下的有效性检验")
    print("=" * 80)

    # 设定原假设环境: E[U] = alpha = 0.05 (临界原假设 Boundary H0)
    # 理论保证: 在原假设成立时，e-value 拒绝原假设 (K >= 1/delta) 的概率必须 <= delta
    alpha = 0.05
    delta = 0.10
    threshold = 1.0 / delta
    n_pts = 300
    n_sims = 1000
    
    print(f"[*] 正在模拟 H0 边界流 (E[U] = α = {alpha})，重复试验 {n_sims} 次...")
    rng = np.random.default_rng(2026)
    
    false_rejections = 0
    e_values = []
    
    for i in range(n_sims):
        # 产生服从边界原假设的 U (均值为 alpha 的伯努利或贝塔分布)
        U_null = rng.binomial(1, alpha, size=n_pts).astype(float)
        
        # 运行 rcrc_core 中的置换平均 e 值算法
        e_val = evalue_martingale(U_null, alpha, perms=100, rng=rng)
        e_values.append(e_val)
        if e_val >= threshold:
            false_rejections += 1

    emp_type1_error = false_rejections / n_sims
    mean_eval = np.mean(e_values)
    
    print(f"[*] 理论上界要求: 假阳性率 (Type-I Error) <= {delta:.2%}")
    print(f"[*] 仿真计算结果: 实测假阳性率 = {emp_type1_error:.2%} | 平均 e-value = {mean_eval:.4f}")
    
    if emp_type1_error <= delta:
        print(f"✅ 【通过】鞅检验在原假设下严格受控 (Type-I Error: {emp_type1_error:.2%} <= {delta:.2%})，数学性质稳固！")
    else:
        print(f"❌ 【警告】实测假阳性率超标！置换平均存在自由度泄漏，需收紧 Kelly 投注比例！")


if __name__ == "__main__":
    audit_tcia_pathology_source()
    audit_martingale_theory()
    print("\n" + "=" * 80)
    print("✅ P0 级诊断执行完毕，请将终端输出完整复制返回。")
    print("=" * 80)