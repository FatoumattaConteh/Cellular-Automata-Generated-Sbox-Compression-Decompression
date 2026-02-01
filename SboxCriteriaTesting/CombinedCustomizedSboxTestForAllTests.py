# combined_sbox_tests_with_nl.py
# Merges CA-shuffle + improved nonlinearity code with full S-box test suite,
# runs tests on every candidate, and reports per-candidate stats + averages.

import os
import json
import math
import hashlib
import secrets
import random
from typing import Dict, List, Tuple, Any

import numpy as np
import qrcode
from PIL import Image
import base64

# ---------------- Device Secret (simple file fallback) ----------------
def get_or_create_device_secret(path="device_secret.bin") -> bytes:
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    s = secrets.token_bytes(32)
    with open(path, "wb") as f:
        f.write(s)
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass
    return s

# ---------------- QR + VC helpers (minimal) ----------------
def generate_qr(data: str, filename="original_qr.png") -> np.ndarray:
    qr = qrcode.QRCode(version=None, error_correction=qrcode.constants.ERROR_CORRECT_H)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill="black", back_color="white")
    img.save(filename)
    return np.array(img, dtype=np.uint8)

def create_shares(qr_array: np.ndarray, seed=None) -> Tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    rows, cols = qr_array.shape
    s1 = np.zeros((rows, cols), dtype=np.uint8)
    s2 = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            if qr_array[i, j] == 0:
                pat = rng.choice([[0, 0], [1, 1]])
            else:
                pat = rng.choice([[0, 1], [1, 0]])
            s1[i, j] = pat[0]
            s2[i, j] = pat[1]
    return s1, s2

# ---------------- Base S-box (grouped) ----------------
BASE_SBOX = {
    "00": [0,3,5,6,24,27,29,30,40,43,45,46,48,51,53,54],
    "01": [9,10,12,15,17,18,20,23,33,34,36,39,57,58,60,63],
    "10": [1,2,4,7,25,26,28,31,41,42,44,47,49,50,52,55],
    "11": [8,11,13,14,16,19,21,22,32,35,37,38,56,59,61,62],
}

# ---------------- CA (Rule 30) RNG & shuffle ----------------
def ca30_bits_from_seed(seed_hex: str, steps: int = 120, cells: int = 128) -> List[int]:
    digest = hashlib.sha256(seed_hex.encode()).digest()
    bits = ''.join(f"{b:08b}" for b in digest)
    state = [int(b) for b in bits]
    if len(state) == 0:
        state = [1] * cells
    if len(state) < cells:
        state = (state * ((cells // len(state)) + 1))[:cells]
    out = []
    for _ in range(steps):
        new = []
        for i in range(cells):
            left = state[(i - 1) % cells]
            c = state[i]
            right = state[(i + 1) % cells]
            new.append(left ^ (c | right))
        state = new
        out.extend(state)
    return out

def shuffle_sbox_with_ca(base_sbox: Dict[str, List[int]], seed_hex: str,
                         key_bits_per_value=8, steps=120, cells=128) -> Dict[str, List[int]]:
    bits = ca30_bits_from_seed(seed_hex, steps=steps, cells=cells)
    shuffled = {}
    idx = 0
    for rule, vals in base_sbox.items():
        keys = []
        for _ in vals:
            chunk = bits[idx:idx+key_bits_per_value]
            idx += key_bits_per_value
            if len(chunk) < key_bits_per_value:
                # wrap-around (simple)
                chunk = (chunk + bits)[:key_bits_per_value]
            num = int(''.join(str(b) for b in chunk), 2)
            keys.append(num)
        paired = list(zip(keys, vals.copy()))
        paired.sort(key=lambda x: x[0])
        shuffled[rule] = [v for _, v in paired]
    return shuffled

def grouped_sbox_to_flat(grouped_sbox: Dict[str, List[int]]) -> List[int]:
    flat = [None] * 64
    for rule, vals in grouped_sbox.items():
        for idx, val in enumerate(vals):
            flat[val] = idx
    if None in flat:
        raise ValueError("Missing inputs in grouped_sbox -> flat")
    return flat

# ---------------- Compression / Decompression helpers (kept for completeness) ----------------
def build_reverse_sbox(sbox: Dict[str, List[int]]) -> Dict[int, Tuple[str,int]]:
    rev = {}
    for rule, vals in sbox.items():
        for idx, v in enumerate(vals):
            rev[v] = (rule, idx)
    return rev

def compress_bits(bits: str, sbox: Dict[str, List[int]]) -> Tuple[str, List[str]]:
    rev = build_reverse_sbox(sbox)
    out_bits = []
    rules = []
    for i in range(0, len(bits), 6):
        chunk = bits[i:i+6]
        if len(chunk) < 6:
            continue
        val = int(chunk, 2)
        if val not in rev:
            raise ValueError(f"Value {val} not found in S-box")
        rule, idx = rev[val]
        out_bits.append(format(idx, '04b'))
        rules.append(rule)
    return ''.join(out_bits), rules

def decompress_bits(compressed_bits: str, rules: List[str], sbox: Dict[str, List[int]]) -> str:
    out = []
    for i, rule in enumerate(rules):
        start = i * 4
        idx = int(compressed_bits[start:start+4], 2)
        val = sbox[rule][idx]
        out.append(format(val, '06b'))
    return ''.join(out)

# ------------------ Walsh / Nonlinearity utilities (from your improved code) ------------------
def fwht(a: List[int]) -> List[int]:
    res = a[:]
    n = len(res)
    h = 1
    while h < n:
        for i in range(0, n, h*2):
            for j in range(i, i+h):
                x = res[j]; y = res[j+h]
                res[j] = x + y
                res[j+h] = x - y
        h *= 2
    return res

def compute_nonlinearity_for_mask(flat_sbox: List[int], mask: int) -> int:
    n = 6
    N = 1 << n
    tt = [0] * N
    for x in range(N):
        out = flat_sbox[x]
        bit = bin(out & mask).count("1") & 1
        tt[x] = 1 if bit else 0
    # transform 0/1 -> +1/-1 as in Walsh transform
    F = [1 if b == 0 else -1 for b in tt]
    W = fwht(F)
    W_abs_max = max(abs(v) for v in W)
    NL = (1 << (n - 1)) - (W_abs_max // 2)
    return NL

def test_sbox_nonlinearity(grouped_sbox: Dict[str, List[int]]) -> Tuple[int, int]:
    flat = grouped_sbox_to_flat(grouped_sbox)
    min_nl = None; min_mask = None
    for mask in range(1, 1 << 4):
        nl = compute_nonlinearity_for_mask(flat, mask)
        if min_nl is None or nl < min_nl:
            min_nl = nl; min_mask = mask
    return min_nl, min_mask

def walsh_hadamard_nonlinearity_per_output(flat_sbox: List[int]) -> List[int]:
    # Compute per-output-bit nonlinearity using FWHT on each output bit stream (output bits are 4)
    n, m = 6, 4
    NL = []
    for out_bit in range(m):
        f = [((x >> (m - 1 - out_bit)) & 1) for x in flat_sbox]
        # convert to +1 / -1
        F = [1 if b == 0 else -1 for b in f]
        W = fwht(F)
        W_abs_max = max(abs(v) for v in W)
        nl = (1 << (n - 1)) - (W_abs_max // 2)
        NL.append(int(nl))
    return NL

# ---------------- S-box test suite (SAC, BIC, Balanced, NoFixed, DDT) ----------------
def strict_avalanche_test(flat_sbox: List[int]) -> List[float]:
    n, m = 6, 4
    sac = np.zeros((n, m), dtype=float)
    for x in range(64):
        y = np.array([int(b) for b in format(flat_sbox[x], '04b')])
        for i in range(n):
            x_flipped = x ^ (1 << i)
            y_flipped = np.array([int(b) for b in format(flat_sbox[x_flipped], '04b')])
            sac[i] += (y != y_flipped)
    # return average change per output bit (normalized to 0..1)
    sac = sac / 64.0
    # average across input-bit flips for each output bit
    return list(sac.mean(axis=0))

def bic_test(flat_sbox):
    """
    Bit Independence Criterion (BIC)
    Measures correlation between output bit changes
    under single input-bit flips.
    """
    n_inputs = 64
    n_inbits = 6
    n_outbits = 4

    bic_data = [[] for _ in range(n_inbits)]

    for x in range(n_inputs):
        y = flat_sbox[x]
        for i in range(n_inbits):
            xf = x ^ (1 << i)
            yf = flat_sbox[xf]

            change_vec = []
            for j in range(n_outbits):
                change_vec.append(
                    ((y >> j) & 1) ^ ((yf >> j) & 1)
                )

            bic_data[i].append(change_vec)

    correlations = []
    max_corr = 0.0

    for i in range(n_inbits):
        changes = np.array(bic_data[i])  # 64 × 4
        for a in range(n_outbits):
            for b in range(a + 1, n_outbits):
                c = np.corrcoef(changes[:, a], changes[:, b])[0, 1]
                if np.isnan(c):
                    c = 0.0
                c = abs(c)
                correlations.append(c)
                max_corr = max(max_corr, c)

    avg_corr = float(np.mean(correlations)) if correlations else 0.0
    bic_score = 1.0 - avg_corr

    return {
        "avg_correlation": avg_corr,
        "max_correlation": max_corr,
        "bic_score": bic_score
    }



def balanced_output(flat_sbox: List[int]) -> Tuple[List[float], List[float]]:
    counts = np.zeros(4, dtype=int)
    for x in flat_sbox:
        for j in range(4):
            counts[j] += (x >> j) & 1
    freqs = (counts / 64.0).tolist()
    deviations = (np.abs(np.array(freqs) - 0.5)).tolist()
    return freqs, deviations

def no_fixed_points(flat_sbox: List[int]) -> Tuple[int, List[int]]:
    fixed = [x for x in range(64) if flat_sbox[x] == (x & 0xF)]
    return len(fixed), fixed

def ddt(flat_sbox: List[int]) -> Tuple[np.ndarray, Dict[str, Any]]:
    N, M = 64, 16
    table = np.zeros((N, M), dtype=int)
    for dx in range(N):
        for x in range(N):
            dy = flat_sbox[x] ^ flat_sbox[x ^ dx]
            table[dx, dy] += 1
    max_count = int(table[1:, :].max())
    mean_count = float(table[1:, :].mean())
    return table, {"max_count": max_count, "mean_count": mean_count}

# ---------------- JSON-safe converter ----------------
def make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return make_json_safe(obj.tolist())
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    return obj

# ---------------- Main: run tests on many CA-shuffled candidates ----------------
def main():
    # Parameters (tune for speed)
    MAX_TRIES = 50     # number of CA-shuffled candidates to test
    CA_STEPS = 120
    CA_CELLS = 128
    KEY_BITS_PER_VALUE = 8

    device_secret = get_or_create_device_secret()
    device_secret_hex = device_secret.hex()

    # Prepare share bits (we won't focus on compression here)
    data = "Transaction ID:12345, Amount: $90"
    qr_array = generate_qr(data, "original_qr.png")
    share1, share2 = create_shares(qr_array, seed=12345)
    share_bits = ''.join(str(b) for b in share1.ravel())
    while len(share_bits) % 6 != 0:
        share_bits += '0'
    share_hash = hashlib.sha256(share_bits.encode()).hexdigest()

    all_candidates_results = []

    print(f"Testing {MAX_TRIES} CA-shuffled S-box candidates (CA steps={CA_STEPS}, cells={CA_CELLS})...")

    for attempt in range(1, MAX_TRIES + 1):
        nonce = secrets.token_hex(12)
        seed_input = share_hash + nonce + device_secret_hex
        seed_hash = hashlib.sha256(seed_input.encode()).hexdigest()

        # Build candidate
        candidate = shuffle_sbox_with_ca(BASE_SBOX, seed_hash,
                                        key_bits_per_value=KEY_BITS_PER_VALUE,
                                        steps=CA_STEPS, cells=CA_CELLS)

        # Convert to flat mapping for many tests
        try:
            flat = grouped_sbox_to_flat(candidate)
        except ValueError as e:
            # If candidate mapping missing (shouldn't happen), mark failure
            print(f"Attempt {attempt}: grouped->flat failed: {e}")
            continue

        # Tests
        bijective = (len(set([v for vals in candidate.values() for v in vals])) == 64)
        min_nl, min_mask = test_sbox_nonlinearity(candidate)
        per_output_nl = walsh_hadamard_nonlinearity_per_output(flat)
        sac = strict_avalanche_test(flat)             # list of 4 floats (0..1)
        bic_stats = bic_test(flat)      # 4x4 matrix list
        balanced_freqs, balanced_devs = balanced_output(flat)
        no_fixed_count, fixed_list = no_fixed_points(flat)
        ddt_table, ddt_stats = ddt(flat)

        candidate_result = {
            "attempt": attempt,
            "nonce_prefix": nonce[:8],
            "bijective": bool(bijective),
            "min_nonlinearity": int(min_nl),
            "min_mask": int(min_mask),
            "per_output_nl": per_output_nl,
            "sac": sac,
            "bic_avg_corr": bic_stats["avg_correlation"],
            "bic_max_corr": bic_stats["max_correlation"],
            "bic_score": bic_stats["bic_score"],
            "balanced_freqs": balanced_freqs,
            "balanced_devs": balanced_devs,
            "no_fixed_count": int(no_fixed_count),
            "fixed_points": fixed_list,
            "ddt_stats": ddt_stats
        }

        all_candidates_results.append(candidate_result)

        # small progress print every 25 attempts
        if attempt % 25 == 0 or attempt == 1 or attempt == MAX_TRIES:
            print(f"  attempt {attempt:4d}: min_nl={min_nl}, bijective={bijective}, no_fixed={no_fixed_count}")

    # ---- Compute averages over all candidates ----
    # Convert numeric lists to numpy arrays for averaging
    per_output_nl_arr = np.array([r["per_output_nl"] for r in all_candidates_results], dtype=float)  # shape (N,4)
    sac_arr = np.array([r["sac"] for r in all_candidates_results], dtype=float)                     # shape (N,4)
    balanced_arr = np.array([r["balanced_freqs"] for r in all_candidates_results], dtype=float)     # shape (N,4)
    no_fixed_arr = np.array([r["no_fixed_count"] for r in all_candidates_results], dtype=float)
    min_nl_arr = np.array([r["min_nonlinearity"] for r in all_candidates_results], dtype=float)
    ddt_max_arr = np.array([r["ddt_stats"]["max_count"] for r in all_candidates_results], dtype=float)
    ddt_mean_arr = np.array([r["ddt_stats"]["mean_count"] for r in all_candidates_results], dtype=float)
    bic_avg_corr_arr = np.array(
    [r["bic_avg_corr"] for r in all_candidates_results], dtype=float
   )
    bic_score_arr = np.array(
    [r["bic_score"] for r in all_candidates_results], dtype=float
)




    averages = {
        "avg_per_output_nl": list(per_output_nl_arr.mean(axis=0)) if len(per_output_nl_arr) else [],
        "avg_sac": list(sac_arr.mean(axis=0)) if len(sac_arr) else [],
        "avg_balanced_freqs": list(balanced_arr.mean(axis=0)) if len(balanced_arr) else [],
        "avg_no_fixed_count": float(no_fixed_arr.mean()) if len(no_fixed_arr) else 0.0,
        "avg_min_nonlinearity": float(min_nl_arr.mean()) if len(min_nl_arr) else 0.0,
        "avg_ddt_max": float(ddt_max_arr.mean()) if len(ddt_max_arr) else 0.0,
        "avg_ddt_mean": float(ddt_mean_arr.mean()) if len(ddt_mean_arr) else 0.0,
        "candidates_tested": len(all_candidates_results),
        "avg_bic_avg_corr": float(bic_avg_corr_arr.mean()),
        "avg_bic_score": float(bic_score_arr.mean())
        
    }
    averages.update({
    "avg_bic_avg_corr": float(bic_avg_corr_arr.mean()),
    "avg_bic_score": float(bic_score_arr.mean())
})

    # Save per-candidate results and averages
    out = {
        "candidates": all_candidates_results,
        "averages": averages
    }

    out_path = "all_candidates_sbox_tests.json"
    with open(out_path, "w") as f:
        json.dump(make_json_safe(out), f, indent=2)

    # Print a compact average summary
    print("\n=== AVERAGE STATISTICS OVER %d CANDIDATES ===" % averages["candidates_tested"])
    print("Avg per-output NL (4 bits):", averages["avg_per_output_nl"])
    print("Avg SAC (4 bits):", averages["avg_sac"])
    print("Avg BIC Correlation:", averages["avg_bic_avg_corr"])
    print("Avg BIC Score:", averages["avg_bic_score"])
    print("Avg Balanced freqs (4 bits):", averages["avg_balanced_freqs"])
    print("Avg no. fixed points:", averages["avg_no_fixed_count"])
    print("Avg min NL (mask search):", averages["avg_min_nonlinearity"])
    print("Avg DDT max / mean:", averages["avg_ddt_max"], "/", averages["avg_ddt_mean"])
    print(f"\nSaved detailed results to: {out_path}")

if __name__ == "__main__":
    main()
