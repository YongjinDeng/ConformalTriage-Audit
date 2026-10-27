# -*- coding: utf-8 -*-
"""
================================================================================
00_data_pipeline.py: LIDC-IDRI 全量数据获取、切片与特征提取总工程
整合原：
  Step 1: 01_prep_manifest.py (TCIA 病理清洗与 pylidc 元数据清单生成)
  Step 2: 02_stream_download.py (TCIA API 流式 DICOM 下载与 3D 各向同性切片)
  Step 3: 03_extract_fmcib.py (FMCIB 4096 维 foc 与 ctx 深度特征提取)
用法：
  python 00_data_pipeline.py --step 1    # 仅生成清洗清单 manifest_nodules.csv
  python 00_data_pipeline.py --step 2    # 流式下载与切片 (带断点续传，即裁即删)
  python 00_data_pipeline.py --step 3    # 提取 FMCIB 深度特征并落盘 npz
  python 00_data_pipeline.py --all       # 端到端从零全自动运行三部曲
================================================================================
"""

import os
import sys
import time
import glob
import shutil
import zipfile
import argparse
import requests
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------- 全局目录与路径 ----------------
BASE_DIR = Path(r"D:\0临床科研\生物视觉图网络用于分类")
RESULT_DIR = BASE_DIR / "result"
TMP_DIR = Path(r"D:\lidc_tmp")
PATCH_DIR = Path(r"D:\lidc_patches")

RESULT_DIR.mkdir(parents=True, exist_ok=True)
PATCH_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_CSV = RESULT_DIR / "manifest_nodules.csv"
DIAG_XLS = RESULT_DIR / "tcia-diagnosis-data-2012-04-20.xls"
FEAT_NPZ = RESULT_DIR / "feat_lidc_full.npz"
DONE_FILE = PATCH_DIR / "done.txt"
META_FILE = PATCH_DIR / "meta.csv"

TCIA_IMG_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage"
TCIA_DIAG_URL = "https://www.cancerimagingarchive.net/wp-content/uploads/tcia-diagnosis-data-2012-04-20.xls"


# ======================================================================
# Step 1: 解析 TCIA 病理与 pylidc 清洗清单生成
# ======================================================================
def run_step1_manifest():
    print("\n" + "=" * 80)
    print("🚀 [Step 1] 解析 TCIA 病理诊断表并清洗 LIDC 元数据...")
    print("=" * 80)

    # 1. 下载并解析 TCIA 病理表
    if not DIAG_XLS.exists():
        print(f"[*] 正在下载 TCIA 病理诊断表 -> {DIAG_XLS}")
        r = requests.get(TCIA_DIAG_URL, timeout=30)
        with open(DIAG_XLS, "wb") as f:
            f.write(r.content)

    import xlrd
    wb = xlrd.open_workbook(str(DIAG_XLS))
    sh = wb.sheet_by_index(0)
    diag_map = {}
    for r in range(1, sh.nrows):
        vals = sh.row_values(r)
        p = str(vals[0]).strip()
        pid = f"LIDC-IDRI-{int(p):04d}" if p.isdigit() else p
        for v in vals[1:5]:
            if isinstance(v, (int, float)) and v in (1, 2, 3):
                diag_map[pid] = 1 if v in (2, 3) else 0  # 1=良性, 2/3=恶性
                break

    # 2. 离线清洗 pylidc 结节 (排除合并簇 > 4 和 < 3 的非共识结节)
    for k, v in {"int": int, "float": float, "bool": bool}.items():
        if not hasattr(np, k): setattr(np, k, v)
    import pylidc as pl

    rows, n_merged = [], 0
    print("[*] 正在通过 pylidc 遍历结节聚类标注...")
    for scan in pl.query(pl.Scan):
        try:
            clusters = scan.cluster_annotations(verbose=False)
        except TypeError:
            clusters = scan.cluster_annotations()
            
        for cl in clusters:
            if len(cl) > 4:
                n_merged += 1
                continue
            if len(cl) < 3:
                continue
            r = [a.malignancy for a in cl]
            rows.append(dict(
                pid=scan.patient_id, uid=scan.series_instance_uid,
                slice_th=scan.slice_thickness, ann_ids=",".join(str(a.id) for a in cl),
                n_read=len(cl), ratings=",".join(map(str, r)),
                diam=float(np.mean([a.diameter for a in cl])),
                disagree=int(any(x >= 4 for x in r) and any(x <= 2 for x in r)),
                q=float(np.mean([x >= 4 for x in r])),
                pathology=diag_map.get(scan.patient_id, -1)
            ))

    df = pd.DataFrame(rows)
    df.to_csv(MANIFEST_CSV, index=False)
    print(f"[+] Step 1 完成！清洗合规结节总数: {len(df)} 颗 (涵盖 {df.uid.nunique()} 个序列) -> {MANIFEST_CSV}")


# ======================================================================
# Step 2: 流式从 TCIA 下载 DICOM 并切出 3D 物理切片 (断点续传)
# ======================================================================
def fetch_series(uid, dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir.parent / f"{uid}.zip"
    for attempt in range(1, 4):
        try:
            r = requests.get(TCIA_IMG_URL, params={"SeriesInstanceUID": uid}, stream=True, timeout=600)
            if r.status_code == 200:
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk: f.write(chunk)
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(dest_dir)
                if zip_path.exists(): zip_path.unlink()
                return True
        except Exception:
            time.sleep(10)
    if zip_path.exists(): zip_path.unlink()
    return False


def load_dicom_volume(dest_dir):
    import pydicom
    files = list(dest_dir.rglob("*.dcm"))
    if not files:
        files = [f for f in dest_dir.rglob("*") if f.is_file() and not f.name.endswith(".zip")]
    slices = [pydicom.dcmread(str(f)) for f in files if hasattr(pydicom.dcmread(str(f), stop_before_pixels=True), "ImagePositionPatient")]
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    slice_spacing = abs(float(slices[1].ImagePositionPatient[2]) - float(slices[0].ImagePositionPatient[2])) if len(slices) > 1 else float(getattr(slices[0], "SliceThickness", 1.0))
    sp = (float(slices[0].PixelSpacing[0]), float(slices[0].PixelSpacing[1]), float(slice_spacing))
    imgs = [(s.pixel_array.astype(np.float32) * float(getattr(s, "RescaleSlope", 1.0)) + float(getattr(s, "RescaleIntercept", -1024.0))) for s in slices]
    return np.stack(imgs, axis=-1), sp, str(getattr(slices[0], "Manufacturer", "Unknown")).strip(), str(getattr(slices[0], "ConvolutionKernel", "Unknown")).strip()


def crop_iso(vol, sp, ctr, size_mm, out_mm):
    from scipy.ndimage import zoom
    half = [int(np.ceil(size_mm / 2.0 / s)) + 2 for s in sp]
    c = np.round(ctr).astype(int)
    sl, pad = [], []
    for d in range(3):
        lo, hi = c[d] - half[d], c[d] + half[d]
        pad.append((max(0, -lo), max(0, hi - vol.shape[d])))
        sl.append(slice(max(0, lo), min(vol.shape[d], hi)))
    sub = np.pad(vol[tuple(sl)], pad, constant_values=-1024).astype(np.float32)
    sub = zoom(sub, [s / out_mm for s in sp], order=1)
    n = int(round(size_mm / out_mm))
    o = [(s - n) // 2 for s in sub.shape]
    return sub[o[0]:o[0]+n, o[1]:o[1]+n, o[2]:o[2]+n]


def run_step2_download():
    print("\n" + "=" * 80)
    print("🚀 [Step 2] 启动 LIDC 全量流式下载与 3D 结节物理切片...")
    print("=" * 80)

    if not MANIFEST_CSV.exists():
        raise FileNotFoundError(f"未找到清单文件，请先执行 --step 1: {MANIFEST_CSV}")

    df_man = pd.read_csv(MANIFEST_CSV)
    done_uids = set(DONE_FILE.read_text().split()) if DONE_FILE.exists() else set()
    grouped = list(df_man.groupby("uid"))
    remaining_scans = [item for item in grouped if item[0] not in done_uids]

    print(f"  - 全量合规序列总数: {len(grouped)}")
    print(f"  - 已完成断点数:     {len(done_uids)}")
    print(f"  - 本次待处理序列数: {len(remaining_scans)}")
    print("=" * 80)

    for k, v in {"int": int, "float": float, "bool": bool}.items():
        if not hasattr(np, k): setattr(np, k, v)
    import pylidc as pl

    for idx, (uid, g) in enumerate(remaining_scans, 1):
        pid = g.pid.iloc[0]
        dest_dir = TMP_DIR / pid / uid
        t0 = time.time()
        print(f"[{idx}/{len(remaining_scans)}] 下载 {pid} (UID: {uid[:12]}...)...", end="", flush=True)

        if not fetch_series(uid, dest_dir):
            print(" ❌ 下载失败跳过")
            shutil.rmtree(TMP_DIR / pid, ignore_errors=True)
            continue

        try:
            vol, sp, mfg, kernel = load_dicom_volume(dest_dir)
            rec = []
            for _, r in g.iterrows():
                ann_ids = [int(x) for x in str(r.ann_ids).split(",")]
                anns = pl.query(pl.Annotation).filter(pl.Annotation.id.in_(ann_ids)).all()
                ctr = np.mean([a.centroid for a in anns], axis=0)
                foc = crop_iso(vol, sp, ctr, 50, 1.0).astype(np.int16)   # 50mm @ 1mm
                ctx = crop_iso(vol, sp, ctr, 100, 2.0).astype(np.int16) # 100mm @ 2mm
                nid = f"{pid}_{ann_ids[0]}"
                np.savez_compressed(PATCH_DIR / f"{nid}.npz", foc=foc, ctx=ctx)
                rec.append(dict(nid=nid, uid=uid, pid=pid, manufacturer=mfg, kernel=kernel,
                                slice_th=r.slice_th, n_read=r.n_read, ratings=r.ratings,
                                diam=r.diam, disagree=r.disagree, q=r.q))

            pd.DataFrame(rec).to_csv(META_FILE, mode="a", header=not META_FILE.exists(), index=False)
            with open(DONE_FILE, "a") as f: f.write(uid + "\n")
            print(f" ✅ 提取 {len(g)} 个切片 [{time.time()-t0:.1f}s]")
        except Exception as e:
            print(f" ❌ 处理异常: {e}")
        finally:
            shutil.rmtree(TMP_DIR / pid, ignore_errors=True)

    print("\n[+] Step 2 完成！切片落盘目录: " + str(PATCH_DIR))


# ======================================================================
# Step 3: 提取 FMCIB 4096 维 foc 与 ctx 深度特征
# ======================================================================
def norm_hu(x):
    return np.clip((x.astype(np.float32) + 1024.0) / 3072.0, 0.0, 1.0)


def run_step3_extract():
    print("\n" + "=" * 80)
    print("🧠 [Step 3] 启动全量结节 FMCIB 4096 维深度特征提取 (CPU 并行)...")
    print("=" * 80)

    import torch
    from fmcib.models import fmcib_model

    torch.set_num_threads(8)
    device = torch.device("cpu")
    model = fmcib_model().to(device).eval()

    df_man = pd.read_csv(MANIFEST_CSV)
    df_man["nid"] = df_man["pid"].astype(str) + "_" + df_man["ann_ids"].astype(str).str.split(",").str[0]
    df_man = df_man.drop_duplicates("nid").reset_index(drop=True)

    existing_nids = set(os.path.basename(f).replace(".npz", "") for f in glob.glob(str(PATCH_DIR / "*.npz")))
    valid_df = df_man[df_man.nid.isin(existing_nids)].reset_index(drop=True)
    n_total = len(valid_df)
    print(f"[*] 检测到已切片结节: {n_total} 颗 (目标: 1,378)")

    feats_foc, feats_ctx, valid_nids = [], [], []
    t0 = time.time()

    with torch.no_grad():
        for i, r in valid_df.iterrows():
            nid = r["nid"]
            data = np.load(PATCH_DIR / f"{nid}.npz")
            x_foc = torch.from_numpy(norm_hu(data["foc"])).unsqueeze(0).unsqueeze(0).to(device)
            x_ctx = torch.from_numpy(norm_hu(data["ctx"])).unsqueeze(0).unsqueeze(0).to(device)
            feats_foc.append(model(x_foc).flatten().cpu().numpy())
            feats_ctx.append(model(x_ctx).flatten().cpu().numpy())
            valid_nids.append(nid)

            if len(valid_nids) % 100 == 0 or len(valid_nids) == n_total:
                elapsed = time.time() - t0
                print(f"  -> 进度: {len(valid_nids):4d}/{n_total} ({len(valid_nids)/n_total*100:.1f}%) | 耗时: {elapsed:.0f}s")

    np.savez_compressed(
        FEAT_NPZ, 
        foc=np.stack(feats_foc), 
        ctx=np.stack(feats_ctx), 
        nids=np.array(valid_nids)
    )
    print(f"\n[+] Step 3 完成！全量特征落盘: {FEAT_NPZ}")
    print(f"    - foc 矩阵形状: {np.stack(feats_foc).shape}")
    print(f"    - ctx 矩阵形状: {np.stack(feats_ctx).shape}")


# ======================================================================
# CLI 命令行调度
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="LIDC 全量原始数据生产流水线")
    parser.add_argument("--step", type=int, choices=[1, 2, 3], help="执行特定阶段 (1:清单, 2:下载切片, 3:特征提取)")
    parser.add_argument("--all", action="store_true", help="端到端顺序执行所有三阶段")

    args = parser.parse_args()
    if not args.step and not args.all:
        parser.print_help()
        sys.exit(0)

    if args.all or args.step == 1:
        run_step1_manifest()
    if args.all or args.step == 2:
        run_step2_download()
    if args.all or args.step == 3:
        run_step3_extract()


if __name__ == "__main__":
    main()