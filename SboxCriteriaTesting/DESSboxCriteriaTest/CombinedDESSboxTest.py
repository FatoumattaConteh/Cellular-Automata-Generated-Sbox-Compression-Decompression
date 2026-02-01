#!/usr/bin/env python3
"""
des_sbox_metrics.py

Compute comprehensive cryptographic strength metrics for the 8 DES S-boxes:
- Bijection check
- Per-output nonlinearity (Walsh/FWHT)
- Min nonlinearity across output masks (mask search)
- Strict Avalanche Criterion (SAC)
- Bit Independence Criterion (BIC)
- Balanced output frequencies & deviations
- Fixed points (count + list)
- Differential Distribution Table (DDT) and summary stats (max, mean)

Outputs:
 - prints per-SBox summary
 - writes JSON file "des_sbox_metrics.json" with detailed results
"""

import json
import itertools
from typing import List, Dict, Any
import numpy as np

# -----------------------------
# DES S-Boxes (standard 4x16 tables)
# -----------------------------
DES_SBOXES = [
    [
        [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
        [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
        [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
        [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]
    ],
    [
        [15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
        [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
        [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
        [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]
    ],
    [
        [10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
        [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
        [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
        [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12]
    ],
    [
        [7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
        [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
        [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
        [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14]
    ],
    [
        [2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
        [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
        [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
        [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3]
    ],
    [
        [12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
        [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
        [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
        [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13]
    ],
    [
        [4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
        [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
        [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
        [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12]
    ],
    [
        [13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
        [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
        [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
        [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11]
    ]
]

# -----------------------------
# Utility helpers
# -----------------------------
def flatten_des_sbox(sbox_4x16: List[List[int]]) -> List[int]:
    """Flatten 4x16 DES table into 64-element list indexed by input 0..63."""
    flat = []
    for row in range(4):
        for col in range(16):
            flat.append(sbox_4x16[row][col])
    return flat

def sbox_output(sbox_4x16: List[List[int]], input6: int) -> int:
    """Compute 6->4 mapping used by DES S-box: row = b6 b1, col = b2..b5"""
    # row: bit6 (msb) and bit1 (lsb)
    row = ((input6 & 0b100000) >> 4) | (input6 & 0b1)
    col = (input6 >> 1) & 0b1111
    return sbox_4x16[row][col]

# -----------------------------
# Fast Walsh-Hadamard (FWHT)
# -----------------------------
def fwht(a: List[int]) -> List[int]:
    """In-place FWHT returning transformed list (copy)."""
    res = a[:]
    n = len(res)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x = res[j]; y = res[j + h]
                res[j] = x + y
                res[j + h] = x - y
        h *= 2
    return res

# -----------------------------
# Nonlinearity calculations
# -----------------------------
def compute_nonlinearity_for_mask(flat_sbox: List[int], mask: int) -> int:
    """
    For a given mask on output bits (1..15), compute boolean function nonlinearity
    using Walsh transform. flat_sbox maps input 0..63 -> output 0..15.
    """
    n = 6
    N = 1 << n  # 64
    tt = [0] * N
    for x in range(N):
        out = flat_sbox[x]
        # parity of masked bits: 1 if odd number of 1's in (out & mask)
        bit = bin(out & mask).count("1") & 1
        tt[x] = 1 if bit else 0
    # map 0->1, 1->-1 for Walsh
    F = [1 if b == 0 else -1 for b in tt]
    W = fwht(F)
    W_abs_max = max(abs(v) for v in W)
    NL = (1 << (n - 1)) - (W_abs_max // 2)
    return int(NL)

def per_output_nonlinearity(flat_sbox: List[int]) -> List[int]:
    """Compute nonlinearity per output bit (4 outputs)."""
    n, m = 6, 4
    NL = []
    for out_bit in range(m):
        # create truth table for this output bit
        F = []
        for x in range(1 << n):
            bit = (flat_sbox[x] >> out_bit) & 1
            F.append(1 if bit == 0 else -1)
        W = fwht(F)
        W_abs_max = max(abs(v) for v in W)
        nl = (1 << (n - 1)) - (W_abs_max // 2)
        NL.append(int(nl))
    return NL

# -----------------------------
# SAC, BIC, Balanced, Fixed points
# -----------------------------
def strict_avalanche_flat(flat_sbox: List[int]) -> List[float]:
    """SAC: for each output bit, average fraction of flips when flipping each input bit."""
    n, m = 6, 4
    sac = np.zeros((n, m), dtype=float)
    for x in range(64):
        y = np.array([ (flat_sbox[x] >> b) & 1 for b in range(m) ])
        for i in range(n):
            x_flipped = x ^ (1 << i)
            y_flipped = np.array([ (flat_sbox[x_flipped] >> b) & 1 for b in range(m) ])
            sac[i] += (y != y_flipped)
    # normalize by number of inputs (64)
    sac = sac / 64.0
    # return average across input-bit flips for each output bit
    return list(sac.mean(axis=0))

def bit_independence_flat(flat_sbox: List[int]) -> Dict[str, Any]:
    """
    Compute BIC using correlation between output-bit avalanche vectors.
    Returns:
      - bic_matrix: |corr| matrix (m x m)
      - avg_bic: average off-diagonal correlation
      - bic_score: 1 - avg_bic (higher is better)
    """
    n, m = 6, 4
    avalanche = {i: [] for i in range(m)}

    for x in range(64):
        y = [(flat_sbox[x] >> k) & 1 for k in range(m)]
        for b in range(n):
            xf = x ^ (1 << b)
            yf = [(flat_sbox[xf] >> k) & 1 for k in range(m)]
            for i in range(m):
                avalanche[i].append(y[i] ^ yf[i])

    bic_matrix = np.zeros((m, m), dtype=float)
    for i in range(m):
        for j in range(m):
            if i == j:
                bic_matrix[i, j] = 0.0
            else:
                a = np.array(avalanche[i])
                b = np.array(avalanche[j])
                if np.std(a) == 0 or np.std(b) == 0:
                    bic_matrix[i, j] = 0.0
                else:
                    bic_matrix[i, j] = abs(np.corrcoef(a, b)[0, 1])

    off_diag = [bic_matrix[i, j] for i in range(m) for j in range(m) if i != j]
    avg_bic = float(np.mean(off_diag))
    bic_score = float(1.0 - avg_bic)

    return {
        "bic_matrix": np.round(bic_matrix, 6).tolist(),
        "avg_bic": round(avg_bic, 6),
        "bic_score": round(bic_score, 6)
    }



def balanced_output(flat_sbox: List[int]) -> Dict[str, Any]:
    """Return freq of 1s per output bit and deviation from 0.5."""
    counts = np.zeros(4, dtype=int)
    for x in flat_sbox:
        for j in range(4):
            counts[j] += (x >> j) & 1
    freqs = (counts / 64.0).tolist()
    deviations = (np.abs(np.array(freqs) - 0.5)).tolist()
    return {"freqs": freqs, "deviations": deviations}

def fixed_points(flat_sbox: List[int]) -> Dict[str, Any]:
    """Count fixed points where S(x) == x & 0xF"""
    fixed = [x for x in range(64) if flat_sbox[x] == (x & 0xF)]
    return {"count": len(fixed), "points": fixed}

# -----------------------------
# DDT
# -----------------------------
def compute_ddt(flat_sbox: List[int]) -> Dict[str, Any]:
    """Compute full DDT (64 x 16). Return summary stats."""
    N = 64
    M = 16
    table = np.zeros((N, M), dtype=int)
    for dx in range(N):
        for x in range(N):
            dy = flat_sbox[x] ^ flat_sbox[x ^ dx]
            table[dx, dy] += 1
    # exclude dx=0 row for stats
    table_nonzero = table[1:, :]
    max_count = int(table_nonzero.max())
    mean_count = float(table_nonzero.mean())
    # also compute distribution of max per dx
    max_per_dx = table_nonzero.max(axis=1).tolist()
    return {
        "ddt_table": table.tolist(),
        "max_count": max_count,
        "mean_count": mean_count,
        "max_per_dx": max_per_dx
    }

# -----------------------------
# Runner per S-box
# -----------------------------
def evaluate_des_sbox(sbox_4x16: List[List[int]]) -> Dict[str, Any]:
    flat = flatten_des_sbox(sbox_4x16)
    # basic
    bijective = (len(set(flat)) == 64)
    # per-output nonlinearity
    per_out_nl = per_output_nonlinearity(flat)
    # mask-based search: find minimum nonlinearity across non-zero masks (1..15)
    mask_nls = {mask: compute_nonlinearity_for_mask(flat, mask) for mask in range(1, 1<<4)}
    min_mask = min(mask_nls, key=lambda k: mask_nls[k])
    min_nl = mask_nls[min_mask]
    # SAC
    sac = strict_avalanche_flat(flat)
    # BIC
    bic_res = bit_independence_flat(flat)
    # balanced and fixed points
    balanced = balanced_output(flat)
    fixed = fixed_points(flat)
    # DDT
    ddt_res = compute_ddt(flat)
    # compose result
    return {
        "bijective": bool(bijective),
        "per_output_nonlinearity": [int(x) for x in per_out_nl],
        "mask_nonlinearity_min": int(min_nl),
        "mask_nonlinearity_min_mask": int(min_mask),
        "mask_nonlinearity_all": {str(k): int(v) for k, v in mask_nls.items()},
        "sac": [float(round(x, 6)) for x in sac],
        "bic": bic_res,
        "balanced": balanced,
        "fixed_points": fixed,
        "ddt_stats": {
            "max_count": int(ddt_res["max_count"]),
            "mean_count": float(ddt_res["mean_count"])
        },
        # keep full DDT if desired (could be large); included for completeness
        "ddt_table": ddt_res["ddt_table"]
    }

# -----------------------------
# Main
# -----------------------------
def main():
    all_results: List[Dict[str, Any]] = []
    for idx, sbox in enumerate(DES_SBOXES, start=1):
        print(f"Evaluating DES S-box {idx}...")
        res = evaluate_des_sbox(sbox)
        res["sbox_index"] = idx
        all_results.append(res)

    # compute compact averages & summary
    per_output_nl_arr = np.array([r["per_output_nonlinearity"] for r in all_results], dtype=float)
    sac_arr = np.array([r["sac"] for r in all_results], dtype=float)
    mask_min_arr = np.array([r["mask_nonlinearity_min"] for r in all_results], dtype=float)
    ddt_max_arr = np.array([r["ddt_stats"]["max_count"] for r in all_results], dtype=float)
    ddt_mean_arr = np.array([r["ddt_stats"]["mean_count"] for r in all_results], dtype=float)
    fixed_counts = np.array([r["fixed_points"]["count"] for r in all_results], dtype=int)
    bic_avg_arr = np.array([r["bic"]["avg_bic"] for r in all_results], dtype=float)
    bic_score_arr = np.array([r["bic"]["bic_score"] for r in all_results], dtype=float)

    summary = {
        "avg_per_output_nonlinearity": list(np.round(per_output_nl_arr.mean(axis=0), 3)),
        "avg_sac": list(np.round(sac_arr.mean(axis=0), 6)),
        "avg_mask_min_nonlinearity": float(round(mask_min_arr.mean(), 3)),
        "avg_ddt_max": float(round(ddt_max_arr.mean(), 3)),
        "avg_ddt_mean": float(round(ddt_mean_arr.mean(), 3)),
        "avg_fixed_points": float(round(fixed_counts.mean(), 3)),
        "sboxes_tested": len(all_results),
        "avg_bic_correlation": float(round(bic_avg_arr.mean(), 6)),
        "avg_bic_score": float(round(bic_score_arr.mean(), 6))
    }

    out = {
        "sboxes": all_results,
        "summary": summary
    }

    # write JSON
    out_path = "des_sbox_metrics.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # print compact table-style summary
    print("\n=== Compact Summary per S-box ===")
    header = ("SBox", "Bij", "MinNL(mask)", "PerOutNL(4)", "AvgSAC(4)", "DDT_max", "FixedPts")
    print("{:<6} {:<3} {:<12} {:<20} {:<20} {:<8} {:<8}".format(*header))
    for r in all_results:
        print("{:<6} {:<3} {:<12} {:<20} {:<20} {:<8} {:<8}".format(
            r["sbox_index"],
            "Y" if r["bijective"] else "N",
            r["mask_nonlinearity_min"],
            str(r["per_output_nonlinearity"]),
            str([round(x,6) for x in r["sac"]]),
            r["ddt_stats"]["max_count"],
            r["fixed_points"]["count"]
        ))
    print("\nSaved detailed JSON ->", out_path)
    print("\nAverages:", json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
