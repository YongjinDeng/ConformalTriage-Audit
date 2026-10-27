# -*- coding: utf-8 -*-
"""
09_pathology_audit.py (清洗后真结节级病理对齐版)
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rcrc_core import RESULT_DIR

CLEAN_PATH_CSV = os.path.join(RESULT_DIR, "12_pathology_verified_nodule_level.csv")


def run_pathology_audit():
    print("\n" + "=" * 80)
    print(">>> 正在运行 09: TCIA 组织病理学确诊子集穿透审计 (真结节级清洗版)")
    print("=" * 80)

    if not os.path.exists(CLEAN_PATH_CSV):
        raise FileNotFoundError(f"[-] 未找到清洗文件: {CLEAN_PATH_CSV}，请先运行 12 脚本！")

    df_p = pd.read_csv(CLEAN_PATH_CSV)
    n_total = len(df_p)
    n_mal = (df_p.true_histology == 1).sum()
    n_ben = (df_p.true_histology == 0).sum()

    print(f"[*] 载入真结节级病理子集: {n_total} 颗病灶 (恶性: {n_mal} 颗, 良性: {n_ben} 颗)")

    # 1. Any (<6.0mm)
    miss_any = df_p[(df_p.true_histology == 1) & (df_p.diam < 6.0)]
    sign_any = df_p[(df_p.true_histology == 0) & (df_p.diam < 6.0)]
    r_any = len(miss_any) / max(n_mal, 1)
    b_any = len(sign_any) / max(n_ben, 1)
    print(f"\n[1. Any-reader 认证规则审计 (< 6.0 mm)]")
    print(f"  -> 确诊恶性肿瘤漏诊数: {len(miss_any)} / {n_mal} (恶性漏诊率: {r_any:.2%})")
    print(f"  -> 确诊良性病灶放行数: {len(sign_any)} / {n_ben} (良性放行率: {b_any:.2%})")

    # 2. Majority (<7.0mm)
    miss_maj = df_p[(df_p.true_histology == 1) & (df_p.diam < 7.0)]
    sign_maj = df_p[(df_p.true_histology == 0) & (df_p.diam < 7.0)]
    r_maj = len(miss_maj) / max(n_mal, 1)
    b_maj = len(sign_maj) / max(n_ben, 1)
    print(f"\n[2. Majority 认证规则审计 (< 7.0 mm)]")
    print(f"  -> 确诊恶性肿瘤漏诊数: {len(miss_maj)} / {n_mal} (恶性漏诊率: {r_maj:.2%})")
    print(f"  -> 确诊良性病灶放行数: {len(sign_maj)} / {n_ben} (良性放行率: {b_maj:.2%})")

    # 落盘
    res = pd.DataFrame([
        {"Policy": "Any (<6.0mm)", "Cutoff_mm": 6.0, "Total_Malignant": n_mal, "Missed_Malignant": len(miss_any),
         "True_Cancer_Miss_Rate": r_any, "Total_Benign": n_ben, "Signed_Benign": len(sign_any), "Benign_Release_Rate": b_any},
        {"Policy": "Majority (<7.0mm)", "Cutoff_mm": 7.0, "Total_Malignant": n_mal, "Missed_Malignant": len(miss_maj),
         "True_Cancer_Miss_Rate": r_maj, "Total_Benign": n_ben, "Signed_Benign": len(sign_maj), "Benign_Release_Rate": b_maj}
    ])
    out_csv = os.path.join(RESULT_DIR, "09_pathology_audit_summary.csv")
    res.to_csv(out_csv, index=False)
    print(f"\n[+] 真实组织病理审计汇总已更新落盘 -> {out_csv}")
    print("=" * 80)


if __name__ == "__main__":
    run_pathology_audit()