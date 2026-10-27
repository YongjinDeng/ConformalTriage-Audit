# -*- coding: utf-8 -*-
"""
================================================================================
12_rebuild_nodule_level_pathology.py:
TCIA 结节级真金标准重构与生态学谬误剔除脚本
工作路径: D:\0临床科研\生物视觉图网络用于分类
================================================================================
"""
import os
import sys
import shutil
import requests
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASE_DIR = r"D:\0临床科研\生物视觉图网络用于分类"
BASE_DIR = DEFAULT_BASE_DIR if os.path.exists(DEFAULT_BASE_DIR) else os.path.dirname(CURRENT_DIR)

RESULT_DIR = os.path.join(BASE_DIR, "result")
DATA_DIR = os.path.join(BASE_DIR, "data")
DIAG_XLS = os.path.join(RESULT_DIR, "tcia-diagnosis-data-2012-04-20.xls")
MANIFEST_CSV = os.path.join(RESULT_DIR, "manifest_nodules.csv")
OUT_CSV = os.path.join(RESULT_DIR, "12_pathology_verified_nodule_level.csv")

sys.path.insert(0, CURRENT_DIR)


def get_tcia_xls():
    if os.path.exists(DIAG_XLS):
        return True
    cands = [
        os.path.join(r"D:\0临床科研\生物视觉图网络用于分类\result", "tcia-diagnosis-data-2012-04-20.xls"),
        os.path.join(DATA_DIR, "tcia-diagnosis-data-2012-04-20.xls")
    ]
    for c in cands:
        if os.path.exists(c):
            shutil.copyfile(c, DIAG_XLS)
            return True
    try:
        url = "https://www.cancerimagingarchive.net/wp-content/uploads/tcia-diagnosis-data-2012-04-20.xls"
        r = requests.get(url, timeout=30)
        with open(DIAG_XLS, "wb") as f:
            f.write(r.content)
        return True
    except Exception as e:
        print(f"[-] 无法获取诊断表: {e}")
        return False


def run_pathology_rebuild():
    print("\n" + "=" * 80)
    print("🔬 [P0 攻坚] 启动 TCIA 结节级病理真实性重构与生态学谬误清洗")
    print("=" * 80)

    if not get_tcia_xls():
        print(f"[-] 错误: 未能在 {DIAG_XLS} 找到诊断表！")
        return

    import xlrd
    wb = xlrd.open_workbook(DIAG_XLS)
    sh = wb.sheet_by_index(0)

    headers = [str(sh.cell_value(0, c)).strip() for c in range(sh.ncols)]
    print(f"[*] 成功解析 Excel 表头，包含 {len(headers)} 列")

    # 1. 解析患者级与结节级诊断数据
    # 查找所有 Nodule X Diagnosis 列
    nodule_diag_cols = {}
    for c_idx, h in enumerate(headers):
        h_lower = h.lower()
        if "nodule" in h_lower and "diagnosis" in h_lower:
            # 提取序号，例如 Nodule 1 -> 1
            parts = h.split()
            for p in parts:
                if p.isdigit():
                    nodule_diag_cols[int(p)] = c_idx
                    break

    print(f"[*] 识别出包含结节级诊断的序号: {sorted(list(nodule_diag_cols.keys()))}")

    patient_records = {}
    for r in range(1, sh.nrows):
        vals = sh.row_values(r)
        p_raw = str(vals[0]).strip()
        pid = f"LIDC-IDRI-{int(p_raw):04d}" if p_raw.isdigit() else p_raw

        # 患者级诊断 (Col B, 通常是 vals[1])
        pt_diag = -1
        try:
            v_pt = vals[1]
            if isinstance(v_pt, (int, float)) and v_pt in (1, 2, 3):
                pt_diag = 1 if v_pt in (2, 3) else 0 # 1=恶性, 0=良性
        except Exception:
            pass

        # 逐个结节诊断
        nod_diags = {}
        for nod_num, c_idx in nodule_diag_cols.items():
            try:
                v_nod = vals[c_idx]
                if isinstance(v_nod, (int, float)) and v_nod in (1, 2, 3):
                    nod_diags[nod_num] = 1 if v_nod in (2, 3) else 0
                elif isinstance(v_nod, (int, float)) and v_nod == 0:
                    nod_diags[nod_num] = -1 # 未知/无病理
            except Exception:
                pass

        patient_records[pid] = {
            "patient_level_diag": pt_diag,
            "nodule_diags": nod_diags
        }

    # 2. 匹配本地 manifest_nodules.csv
    df_man = pd.read_csv(MANIFEST_CSV)
    print(f"[*] 载入本地结节清单: {len(df_man)} 颗结节，涵盖 {df_man.pid.nunique()} 例患者")

    verified_records = []
    
    for pid, group in df_man.groupby("pid"):
        if pid not in patient_records:
            continue
        
        info = patient_records[pid]
        pt_diag = info["patient_level_diag"]
        nod_diags = info["nodule_diags"]
        
        n_nods_in_ct = len(group)
        # 按直径降序排列，通常临床编号 Nodule 1 为最大主病灶
        sorted_group = group.sort_values("diam", ascending=False).reset_index(drop=True)

        for nod_rank, (_, row) in enumerate(sorted_group.iterrows(), start=1):
            ann_id = row["ann_ids"]
            diam = row["diam"]
            slice_th = row["slice_th"]
            q = row["q"]

            final_label = -1
            label_level = "None"

            # 规则 1: 该结节在 Excel 中有显式结节级诊断
            if nod_rank in nod_diags and nod_diags[nod_rank] != -1:
                final_label = nod_diags[nod_rank]
                label_level = f"Strict_Nodule_Level_{nod_rank}"
            
            # 规则 2: 患者级有明确诊断，且该患者体内有且仅有 1 颗结节 (无混杂孤立结节)
            elif n_nods_in_ct == 1 and pt_diag != -1:
                final_label = pt_diag
                label_level = "Single_Nodule_Patient_Inherited"

            # 规则 3: 多结节患者，无显式结节病理 -> 拒绝污染，标记为 -1
            else:
                final_label = -1
                label_level = "Unconfirmed_Companion_Nodule"

            if final_label in (0, 1):
                verified_records.append({
                    "pid": pid, "diam": diam, "slice_th": slice_th, "q": q,
                    "ann_ids": ann_id,
                    "true_histology": final_label, # 1=恶性, 0=良性
                    "evidence_level": label_level,
                    "is_single_nodule_case": int(n_nods_in_ct == 1)
                })

    df_clean = pd.DataFrame(verified_records)
    df_clean.to_csv(OUT_CSV, index=False)

    print("\n" + "=" * 80)
    print("📊 [重构结果对比与证据分级]")
    print("=" * 80)
    print(f"[*] 清洗后真正具备病理确诊依据的结节总数: {len(df_clean)} 颗 (此前广播膨胀值为 220 颗)")
    print(f"    - 确诊真实恶性肿瘤 (True Malignant): {df_clean.true_histology.sum()} 颗")
    print(f"    - 确诊真实良性病灶 (True Benign):    {(df_clean.true_histology == 0).sum()} 颗")
    print(f"[*] 恶性肿瘤占比: {df_clean.true_histology.mean():.1%} (此前虚高恶性率为 80.0%)")
    
    print("\n[*] 证据层级构成:")
    for lvl, count in df_clean.evidence_level.value_counts().items():
        print(f"    - {lvl:<35}: {count} 颗")

    # 3. 重新审计放行切点在真病理子集上的表现
    print("\n" + "=" * 80)
    print("🎯 [终极验真] 干净真病理子集下的真实漏诊表现:")
    print("=" * 80)
    
    n_true_mal = df_clean.true_histology.sum()
    n_true_ben = (df_clean.true_histology == 0).sum()

    for cut, name in [(6.0, "Any (<6.0mm)"), (7.0, "Majority (<7.0mm)")]:
        missed_mal = df_clean[(df_clean.true_histology == 1) & (df_clean.diam < cut)]
        signed_ben = df_clean[(df_clean.true_histology == 0) & (df_clean.diam < cut)]
        
        miss_rate = len(missed_mal) / max(n_true_mal, 1)
        ben_release = len(signed_ben) / max(n_true_ben, 1)
        
        print(f"  -> [{name:<18}]:")
        print(f"     真恶性漏诊数: {len(missed_mal)} / {n_true_mal} ({miss_rate:.2%})")
        print(f"     真良性放行数: {len(signed_ben)} / {n_true_ben} ({ben_release:.2%})")
        
        if len(missed_mal) > 0:
            print(f"     [漏诊恶性结节清单]:")
            for _, r in missed_mal.iterrows():
                print(f"       患者: {r.pid:<14} | 直径: {r.diam:.2f}mm | 证据级别: {r.evidence_level} | 专家可疑度: {r.q:.2f}")

    print("=" * 80)


if __name__ == "__main__":
    run_pathology_rebuild()