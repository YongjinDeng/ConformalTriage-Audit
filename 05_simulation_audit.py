# -*- coding: utf-8 -*-
"""
05_simulation_audit.py: 大样本渐近紧致性仿真与双侧 z 检验 (严格落盘版)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from rcrc_core import (calculate_risk, compute_crc_lambda, compute_ltt_lambda, 
                       child_rngs, RESULT_DIR, ALPHAS)


def run_simulation_audit(rep_small=100, rep_large=200):
    print("\n" + "=" * 80)
    print(">>> 正在运行 05: 大样本仿真 (N=200,000) 渐近紧致性与严谨双侧 z 检验 (落盘版)")
    print("=" * 80)

    def simulate_pop(n_pop, rng):
        k = rng.integers(1, 4, n_pop)
        p = np.repeat(np.arange(n_pop), k)
        z = rng.normal(0.0, 0.7, n_pop)
        diam = np.exp(rng.normal(2.3, 0.5, len(p)) + 0.3 * z[p])
        prob = 1.0 / (1.0 + np.exp(-(5.0 * (np.log(diam) - 2.7) + z[p])))
        return p, diam, rng.binomial(4, prob) / 4.0 / k[p]

    rng_master = np.random.default_rng(2025)
    print("[*] 正在构建 N=200,000 超大无偏真实总体...")
    _, big_d, big_w = simulate_pop(200000, rng_master)
    grid = np.arange(3.0, 40.0, 0.5)

    rows = []
    for al in ALPHAS:
        # n = 300 临床规模
        for _ in range(rep_small):
            p300, d300, w300 = simulate_pop(300, rng_master)
            lam_crc = compute_crc_lambda(d300, w300, al)
            lam_ltt = compute_ltt_lambda(d300, w300, p300, al, 0.10, grid, perms=100)
            rows.append(("CRC (n=300)", al, calculate_risk(big_w, big_d < lam_crc), (big_d < lam_crc).mean()))
            rows.append(("LTT (n=300)", al, calculate_risk(big_w, big_d < lam_ltt), (big_d < lam_ltt).mean()))

        # n = 3000 大样本紧致性
        for _ in range(rep_large):
            p3k, d3k, w3k = simulate_pop(3000, rng_master)
            lam_crc3k = compute_crc_lambda(d3k, w3k, al)
            rows.append(("CRC (n=3000)", al, calculate_risk(big_w, big_d < lam_crc3k), (big_d < lam_crc3k).mean()))

    T = pd.DataFrame(rows, columns=["method", "alpha", "risk", "sign"])
    T["violation"] = T.risk > T.alpha

    grp = T.groupby(["alpha", "method"])
    summary = grp.agg(
        总体风险均值=("risk", "mean"),
        MC标准误=("risk", lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
        真实违约概率=("violation", "mean"),
        直签吞吐率=("sign", "mean")
    )
    
    z_scores = (summary["总体风险均值"] - summary.index.get_level_values("alpha")) / summary["MC标准误"]
    summary["z统计量"] = z_scores.round(3)
    summary["未检出违约 (z < 1.645)"] = z_scores < 1.645
    summary["证实显著紧致/保守 (z < -1.645)"] = z_scores < -1.645

    # 严格落盘 CSV 供 Fig 6 纯粹读取
    sim_csv = os.path.join(RESULT_DIR, "05_simulation_summary.csv")
    summary.to_csv(sim_csv)
    print(f"[+] 仿真统计汇总已落盘 -> {sim_csv}")
    print("\n[大样本仿真紧致性统计检验大表]")
    print(summary)


if __name__ == "__main__":
    run_simulation_audit()