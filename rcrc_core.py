# -*- coding: utf-8 -*-
"""
rcrc_core.py: 全局统一核心数学与数据底座 (v2)
新增: betting_path / sample_k_readers / binarize_ratings / expand_patients / CANON_K3_SEED
原有函数 (load_cohort, compute_crc_lambda, evalue_martingale, compute_ltt_lambda ...) 逻辑保持不变。

【CRC 严格性备忘 (写论文用)】
令患者 p 的 L_p(λ)=Σ_i w_i 1[s_i<λ], m_p=Σ_i w_i ∈[0,1]。g_p=L_p-α m_p 关于 λ 单调、上界 (1-α)。
对 g 做标准 CRC: 选最大 λ 使 Σ_cal g_p + (1-α) <= 0, 即 ΣL <= αΣm-(1-α)，与 compute_crc_lambda 完全一致。
在患者可交换下 => E[L_test(λ̂)] <= α E[m_test]，即"比值的期望"受控 (期望保证, 非 PAC)。
"""
import os
import numpy as np
import pandas as pd

BASE_DIR = os.environ.get("RCRC_BASE_DIR", r"D:\0临床科研\生物视觉图网络用于分类")
RESULT_DIR = os.path.join(BASE_DIR, "result")
ALPHAS = (0.05, 0.10, 0.20)
CANON_K3_SEED = 2025      # 所有脚本共用的 K=3 读者子集种子


def child_rngs(master_seed, n):
    """基于 SeedSequence 派生 n 个绝对独立的无偏随机数发生器"""
    ss = np.random.SeedSequence(master_seed).spawn(n)
    return [np.random.default_rng(s) for s in ss]


def load_cohort():
    """全量队列唯一加载器：严格验证 N=1,378 与 3-4 读者结构"""
    mani_path = os.path.join(RESULT_DIR, "manifest_nodules.csv")
    feat_path = os.path.join(RESULT_DIR, "feat_lidc_full.npz")

    if not os.path.exists(mani_path) or not os.path.exists(feat_path):
        raise FileNotFoundError(f"未找到核心数据文件，请检查目录: {RESULT_DIR}")

    m = pd.read_csv(mani_path)
    m["nid"] = m.pid.astype(str) + "_" + m.ann_ids.astype(str).str.split(",").str[0]

    npz = np.load(feat_path)
    nids = npz["nids"]
    feats = npz["foc"]  # 统一使用 foc 特征

    df = m.drop_duplicates("nid").set_index("nid").loc[nids].reset_index()
    assert len(df) == 1378, f"结节数量非 1,378 (实测 {len(df)})"
    assert len(feats) == 1378, "特征数组长度与元数据不一致"

    M = np.full((len(df), 6), np.nan)
    reader_counts = []
    for i, s in enumerate(df.ratings):
        r = [int(x) for x in str(s).split(",") if str(x).strip().isdigit()]
        M[i, :len(r)] = r
        reader_counts.append(len(r))
    M = M[:, ~np.all(np.isnan(M), axis=0)]
    df["n_readers"] = reader_counts

    assert set(df.n_readers.unique()) == {3, 4}, "读者数包含 3 与 4 之外的值"

    def ind(t):
        return np.where(np.isnan(M), np.nan, (M >= t).astype(float))

    errs = {t: np.abs(np.nanmean(ind(t), axis=1) - df.q.values).mean() for t in range(1, 6)}
    tau = min(errs, key=errs.get)
    if errs[tau] > 0.05:
        raise ValueError(f"q 与 ratings >= {tau} 的 MAE = {errs[tau]:.4f} 异常！")

    return df, ind(tau), tau, feats, M


# ----------------------------------------------------------------------
# 读者抽样 / 定义 (新增向量化 K 读者抽样, 全脚本统一)
# ----------------------------------------------------------------------
def binarize_ratings(M, tau=4):
    return np.where(np.isnan(M), np.nan, (M >= tau).astype(float))


def sample_k_readers(Mb, rng, k=3):
    """对每个结节从有效读者中均匀无放回抽 k 个 (向量化, 返回 n×k, 无 NaN)"""
    valid = ~np.isnan(Mb)
    assert (valid.sum(axis=1) >= k).all(), "存在读者数 < k 的结节"
    keys = rng.random(Mb.shape)
    keys[~valid] = np.inf
    idx = np.argsort(keys, axis=1)[:, :k]
    return np.take_along_axis(Mb, idx, axis=1)


def get_risk_definitions(Rind):
    mean_val = np.nanmean(Rind, axis=1)
    return {
        "any": np.nanmax(Rind, axis=1),
        "mean": mean_val,
        "majority": (mean_val >= 0.5).astype(float),
        "unanimous": np.nanmin(Rind, axis=1)
    }


def compute_patient_weights(df, y):
    n_p = df.groupby("pid").pid.transform("size").values
    return y / n_p


def calculate_risk(w, a):
    return (w * a).sum() / max(w.sum(), 1e-12)


# ----------------------------------------------------------------------
# CRC (逻辑不变)
# ----------------------------------------------------------------------
def compute_crc_lambda(s, w, alpha, B=1.0, v=None, clip_max=10.0):
    if v is not None:
        v_clipped = np.clip(v, 1.0 / clip_max, clip_max)
        w = w * v_clipped
        B = B * v_clipped.max()
    order = np.argsort(s, kind="stable")
    cum_w = np.r_[0.0, np.cumsum(w[order])]
    threshold = alpha * w.sum() - (1.0 - alpha) * B
    if threshold < 0:
        return -np.inf
    K = np.searchsorted(cum_w, threshold, side="right") - 1
    return s[order][K] if K < len(s) else np.inf


# ----------------------------------------------------------------------
# 投注鞅 (认证 / 报警)
# ----------------------------------------------------------------------
def evalue_martingale(U, m0, perms, rng, floor=1e-2):
    """认证方向: H0: E[U] >= m0 (风险>α)。置换平均 => 固定样本量 e 值 (非在线)。逻辑不变。"""
    P = np.stack([rng.permutation(U) for _ in range(perms)])
    K = np.ones(perms)
    s1, s2 = np.zeros(perms), np.zeros(perms)
    for t in range(P.shape[1]):
        mu = (m0 + s1) / (t + 1)
        var = np.maximum((0.25 + m0 ** 2 + s2) / (t + 1) - mu ** 2, floor)
        lam = np.clip((m0 - mu) / (var + (m0 - mu) ** 2), 0.0, 0.9 / (1.0 - m0))
        u = P[:, t]
        K *= 1.0 + lam * (m0 - u)
        s1 += u
        s2 += u * u
    return K.mean()


def betting_path(u, m0, floor=1e-3, cap=0.9, direction="up"):
    """
    严格可预测 (predictable) 投注财富路径, 按到达顺序逐个更新, 无置换 => Ville 不等式给出 anytime-valid。
    direction="up"  : H0: E[u] <= m0 (风险<=α), 财富乘 1+λ(u-m0), 超过 1/δ 即报警 (监控用)
    direction="down": H0: E[u] >= m0 (风险>α),  财富乘 1+λ(m0-u), 超过 1/δ 即认证
    λ_t 仅依赖 u_1..u_{t-1} (plug-in Kelly, 带先验收缩)。
    """
    sign = 1.0 if direction == "down" else -1.0
    lam_max = cap / (1.0 - m0) if direction == "down" else cap / m0
    K, s1, s2 = 1.0, 0.0, 0.0
    path = np.empty(len(u))
    for t, ut in enumerate(u):
        mu = (m0 + s1) / (t + 1)
        var = max((0.25 + m0 ** 2 + s2) / (t + 1) - mu ** 2, floor)
        lam = float(np.clip(sign * (m0 - mu) / (var + (m0 - mu) ** 2), 0.0, lam_max))
        K *= 1.0 + lam * sign * (m0 - ut)
        path[t] = K
        s1 += ut
        s2 += ut * ut
    return path


def compute_ltt_lambda(s, w, pid, alpha, delta, grid, floor=1e-2, perms=100, seed=0):
    unique_p, inv_p = np.unique(pid, return_inverse=True)
    m = np.bincount(inv_p, weights=w, minlength=len(unique_p))
    rng = np.random.default_rng(seed)
    best = -np.inf
    for lam in grid:
        l = np.bincount(inv_p, weights=w * (s < lam), minlength=len(unique_p))
        U = np.clip(l - alpha * m + alpha, 0.0, 1.0)
        if evalue_martingale(U, alpha, perms, rng, floor) >= (1.0 / delta):
            best = lam
        else:
            break
    return best


# ----------------------------------------------------------------------
# Bootstrap 工具
# ----------------------------------------------------------------------
def expand_patients(ids_by_pt, pts, cnts=None):
    """按患者(含重复次数)展开为结节下标; 重复抽中的患者 = 多个独立簇, 权重不变"""
    if cnts is None:
        cnts = np.ones(len(pts), dtype=int)
    parts = [np.tile(ids_by_pt[int(p)], int(c)) for p, c in zip(pts, cnts)]
    return np.concatenate(parts) if parts else np.array([], dtype=int)


def patient_bootstrap_ci(metric_fn, pid, n_boot=1000, seed=42):
    unique_p, inv_p = np.unique(pid, return_inverse=True)
    n_pts = len(unique_p)
    rng = np.random.default_rng(seed)
    pt_to_indices = [np.where(inv_p == i)[0] for i in range(n_pts)]

    boot_vals = []
    for _ in range(n_boot):
        boot_pts = rng.integers(0, n_pts, n_pts)
        boot_idx = np.concatenate([pt_to_indices[p] for p in boot_pts])
        boot_vals.append(metric_fn(boot_idx))

    return np.percentile(boot_vals, [2.5, 97.5])