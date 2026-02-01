#!/usr/bin/env python3
# batch_qr_vc_pipeline_optimized.py
"""
OPTIMIZED: Batch QR -> Visual Cryptography -> CA-driven S-box compression pipeline
Key optimizations:
1. Pre-build reverse S-box once
2. Use bytearray for bit operations
3. Batch string operations
4. Optional tracemalloc (disable for speed)
5. Cached entropy calculations
"""

import os
import json
import math
import base64
import hashlib
import secrets
import random
import time
import tracemalloc
import re
from collections import defaultdict
from statistics import mean
from PIL import Image
import numpy as np
import qrcode
import pandas as pd

# ----------------------------
# Configuration
# ----------------------------
DATASET_DIR = "./SboxScriptForTestingOnDataset/qr_images_dataset"
OUTPUT_DIR = "./SboxScriptForTestingOnDataset/optimized_results_with_iteration_pct"
ITERATIONS = 20
CA_STEPS = 50
CA_CELLS = 64
MAX_FILES = None
THRESH = 128
ENABLE_MEMORY_PROFILING = True  # Set to True for detailed memory analysis
SAVE_ITERATION_FILES = False  # Set to True to save per-iteration JSON files (slower)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Optimized Utilities
# ----------------------------
def shannon_entropy(data):
    """Optimized entropy calculation"""
    if isinstance(data, np.ndarray):
        data = data.ravel()
    if len(data) == 0:
        return 0.0
    
    # Use numpy for faster counting
    unique, counts = np.unique(data, return_counts=True)
    length = len(data)
    probs = counts / length
    return -np.sum(probs * np.log2(probs))

def shannon_entropy_from_bits(bit_string):
    """Fast entropy from bit string"""
    if not bit_string:
        return 0.0
    ones = bit_string.count('1')
    zeros = len(bit_string) - ones
    if ones == 0 or zeros == 0:
        return 0.0
    p1 = ones / len(bit_string)
    p0 = zeros / len(bit_string)
    return -(p1 * math.log2(p1) + p0 * math.log2(p0))

def get_device_secret(secret_file="device_secret.bin"):
    if os.path.exists(secret_file):
        with open(secret_file, "rb") as f:
            return f.read()
    secret = secrets.token_bytes(32)
    with open(secret_file, "wb") as f:
        f.write(secret)
    return secret

# ----------------------------
# Base S-box
# ----------------------------
BASE_SBOX = {
    "00": [0,3,5,6,24,27,29,30,40,43,45,46,48,51,53,54],
    "01": [9,10,12,15,17,18,20,23,33,34,36,39,57,58,60,63],
    "10": [1,2,4,7,25,26,28,31,41,42,44,47,49,50,52,55],
    "11": [8,11,13,14,16,19,21,22,32,35,37,38,56,59,61,62],
}

# ----------------------------
# CA generator + S-box shuffle
# ----------------------------
def generate_ca_bits(seed: str, steps: int = CA_STEPS, cells: int = CA_CELLS):
    digest = hashlib.sha256(seed.encode()).digest()
    bits = ''.join(f'{byte:08b}' for byte in digest)
    state = [int(b) for b in bits[:cells]]
    if len(state) < cells:
        state = (state * ((cells // len(state)) + 1))[:cells]
    
    output_bits = []
    for _ in range(steps):
        new_state = []
        for i in range(cells):
            left = state[(i - 1) % cells]
            center = state[i]
            right = state[(i + 1) % cells]
            new_val = left ^ (center | right)
            new_state.append(new_val)
        state = new_state
        output_bits.extend(state)
    return output_bits

def shuffle_sbox_with_ca(original_sbox, seed: str):
    bits = generate_ca_bits(seed)
    shuffled = {}
    idx = 0
    for rule, values in original_sbox.items():
        rand_nums = []
        for _ in range(len(values)):
            chunk = bits[idx: idx+4]
            idx += 4
            if len(chunk) < 4:
                chunk = bits[:4]
                idx = 4
            rand_nums.append(int(''.join(map(str, chunk)), 2))
        paired = list(zip(rand_nums, values.copy()))
        paired.sort(key=lambda x: x[0])
        shuffled[rule] = [val for _, val in paired]
    return shuffled

def build_reverse_sbox(sbox):
    """Build reverse lookup - now called ONCE per file"""
    rev = {}
    for rule, vals in sbox.items():
        for idx, v in enumerate(vals):
            rev[v] = (rule, idx)
    return rev

# ----------------------------
# OPTIMIZED Compression / Decompression
# ----------------------------
def compress_bits_optimized(bits, sbox, rev_sbox):
    """
    Optimized compression using pre-built reverse S-box
    and efficient string building
    """
    out_bits = []
    rules = []
    
    # Process in chunks of 6 bits
    for i in range(0, len(bits) - 5, 6):
        chunk = bits[i:i+6]
        val = int(chunk, 2)
        
        if val not in rev_sbox:
            raise ValueError(f"Value {val} not found in S-box")
        
        rule, idx = rev_sbox[val]
        out_bits.append(format(idx, '04b'))
        rules.append(rule)
    
    return ''.join(out_bits), rules

def decompress_bits_optimized(compressed_bits, rules, sbox):
    """
    Optimized decompression using list comprehension
    """
    if len(compressed_bits) < len(rules) * 4:
        raise ValueError("corrupt compressed_bits length")
    
    out = []
    for i, rule in enumerate(rules):
        start = i * 4
        idx_bits = compressed_bits[start:start+4]
        idx = int(idx_bits, 2)
        val = sbox[rule][idx]
        out.append(format(val, '06b'))
    
    return ''.join(out)

# ----------------------------
# Base64 helpers (unchanged)
# ----------------------------
def bits_to_b64_json(bits):
    if bits == "":
        return {"bit_length": 0, "data": ""}
    b = int(bits, 2).to_bytes((len(bits)+7)//8, byteorder='big')
    return {"bit_length": len(bits), "data": base64.b64encode(b).decode()}

def b64_json_to_bits(b64meta):
    bitlen = int(b64meta.get("bit_length", 0))
    data_b64 = b64meta.get("data", "")
    if bitlen == 0 or data_b64 == "":
        return ""
    raw = base64.b64decode(data_b64)
    bits_full = ''.join(f"{byte:08b}" for byte in raw)
    return bits_full[-bitlen:]

# ----------------------------
# Image / VC helpers
# ----------------------------
def try_embed_qr(data_str):
    try:
        qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H)
        qr.add_data(data_str)
        qr.make(fit=True)
        return True, qr.version
    except Exception:
        return False, None

def create_shares(qr_array):
    rows, cols = qr_array.shape
    share1 = np.zeros((rows, cols), dtype=np.uint8)
    share2 = np.zeros((rows, cols), dtype=np.uint8)
    
    for i in range(rows):
        for j in range(cols):
            if qr_array[i,j] == 0:
                pattern = random.choice([[0,0],[1,1]])
            else:
                pattern = random.choice([[0,1],[1,0]])
            share1[i,j] = pattern[0]
            share2[i,j] = pattern[1]
    
    return share1, share2

def load_image_as_binary(path, thresh=THRESH):
    im = Image.open(path).convert('L')
    arr = np.array(im)
    return (arr >= thresh).astype(np.uint8)

# ----------------------------
# Filename / version parsing
# ----------------------------
VERSION_RE = re.compile(r'v(\d+)_')

def infer_version_from_path(path):
    fname = os.path.basename(path)
    m = VERSION_RE.search(fname)
    if m:
        return int(m.group(1))
    parent = os.path.basename(os.path.dirname(path))
    m = VERSION_RE.search(parent)
    if m:
        return int(m.group(1))
    m2 = re.search(r'version[_\-]?(\d+)', parent, re.IGNORECASE)
    if m2:
        return int(m2.group(1))
    return None

# ----------------------------
# OPTIMIZED process_file
# ----------------------------
def process_file(path, out_dir, iterations=ITERATIONS):
    fname = os.path.splitext(os.path.basename(path))[0]
    
    # Only create directory if we're saving files
    if SAVE_ITERATION_FILES:
        file_out_dir = os.path.join(out_dir, fname)
        os.makedirs(file_out_dir, exist_ok=True)

    try:
        qr_bin = load_image_as_binary(path)
    except Exception as e:
        return {"filename": fname, "error": f"load_error: {e}"}

    flat_bits = ''.join(str(int(v)) for v in qr_bin.ravel().tolist())
    qr_size = len(flat_bits)
    qr_entropy = shannon_entropy(qr_bin)

    share1, share2 = create_shares(qr_bin)
    share1_bits = ''.join(str(b) for b in share1.ravel().tolist())
    share1_entropy = shannon_entropy(share1)
    original_length = len(share1_bits)

    padded = share1_bits + '0' * ((6 - len(share1_bits) % 6) % 6)
    padding_added = len(padded) - original_length
    
    # Optional: save padding info
    if SAVE_ITERATION_FILES:
        with open(os.path.join(file_out_dir, "padding_info.json"), "w") as f:
            json.dump({"padding": padding_added, "original_length": original_length}, f)

    nonce = secrets.token_hex(16)
    share_hash = hashlib.sha256(share1_bits.encode()).hexdigest()
    device_secret = get_device_secret().hex()
    seed = hashlib.sha256((share_hash + nonce + device_secret).encode()).hexdigest()

    shuffled_sbox = shuffle_sbox_with_ca(BASE_SBOX, seed)
    
    # Optional: save shuffled sbox
    if SAVE_ITERATION_FILES:
        with open(os.path.join(file_out_dir, "shuffled_sbox.json"), "w") as f:
            json.dump(shuffled_sbox, f)

    # KEY OPTIMIZATION: Build reverse S-box ONCE
    rev_sbox = build_reverse_sbox(shuffled_sbox)

    compressed_bits = padded
    iteration_stats = []
    peak_memory_bytes = 0
    total_compress_time_ms = 0.0
    total_decompress_time_ms = 0.0

    comp_time_ms_per_iteration = []
    decomp_time_ms_per_iteration = []

    if ENABLE_MEMORY_PROFILING:
        tracemalloc.start()

    # Store compressed data and rules in memory (not files)
    compressed_data_history = []
    rules_history = []

    # ---------------- Compression loop (OPTIMIZED) ----------------
    for i in range(iterations):
        iter_no = i + 1
        t0 = time.perf_counter()
        
        if ENABLE_MEMORY_PROFILING:
            snap_before = tracemalloc.take_snapshot()

        try:
            # Use optimized compression with pre-built reverse S-box
            compressed_bits, rules = compress_bits_optimized(compressed_bits, shuffled_sbox, rev_sbox)
        except Exception as e:
            iteration_stats.append({"iter": iter_no, "error": str(e)})
            break

        t1 = time.perf_counter()
        
        mem_diff = 0
        if ENABLE_MEMORY_PROFILING:
            snap_after = tracemalloc.take_snapshot()
            mem_diff = sum(stat.size_diff for stat in snap_after.compare_to(snap_before, 'filename'))
            peak_memory_bytes = max(peak_memory_bytes, mem_diff)
        
        comp_time_ms = (t1 - t0) * 1000
        total_compress_time_ms += comp_time_ms
        comp_time_ms_per_iteration.append(round(comp_time_ms, 3))

        compression_pct = ((original_length - len(compressed_bits)) / len(compressed_bits)) * 100
        iteration_bits = len(compressed_bits)

        # Store in memory
        compressed_data_history.append(compressed_bits)
        rules_history.append(rules)

        # Optional: save iteration files
        if SAVE_ITERATION_FILES:
            b64meta = bits_to_b64_json(compressed_bits)
            with open(os.path.join(file_out_dir, f"compressed_iter_{iter_no}.json"), "w") as f:
                json.dump(b64meta, f)
            with open(os.path.join(file_out_dir, f"rules_iter_{iter_no}.json"), "w") as f:
                json.dump(rules, f)

        fit_raw, ver_raw = try_embed_qr(compressed_bits)
        fit_b64, ver_b64 = try_embed_qr(json.dumps(bits_to_b64_json(compressed_bits)))

        iteration_stats.append({
            "iter": iter_no,
            "compressed_bits_len": iteration_bits,
            "compression_pct": round(compression_pct, 6),
            "comp_time_ms": round(comp_time_ms, 3),
            "decomp_time_ms": 0.0,
            "comp_memory_bytes": mem_diff,
            "qr_fit_raw": fit_raw,
            "qr_fit_base64": fit_b64,
            "fit_raw_qr_version": ver_raw,
            "fit_base64_qr_version": ver_b64
        })

    # ---------------- Decompression loop (OPTIMIZED) ----------------
    decompressed_bits = None
    for it in reversed([it for it in iteration_stats if "error" not in it]):
        iter_no = it["iter"]
        
        # Get from memory instead of files
        comp_bits_i = compressed_data_history[iter_no - 1]
        rules = rules_history[iter_no - 1]
        
        t0 = time.perf_counter()
        
        if ENABLE_MEMORY_PROFILING:
            snap_before = tracemalloc.take_snapshot()
        
        try:
            # Use optimized decompression
            decompressed_bits = decompress_bits_optimized(comp_bits_i, rules, shuffled_sbox)
        except Exception as e:
            it["decompress_error"] = str(e)
            decompressed_bits = None
            break
        
        t1 = time.perf_counter()
        
        mem_diff = 0
        if ENABLE_MEMORY_PROFILING:
            snap_after = tracemalloc.take_snapshot()
            mem_diff = sum(stat.size_diff for stat in snap_after.compare_to(snap_before, 'filename'))
            peak_memory_bytes = max(peak_memory_bytes, mem_diff)
        
        decomp_time_ms = (t1 - t0) * 1000
        total_decompress_time_ms += decomp_time_ms
        decomp_time_ms_per_iteration.insert(0, round(decomp_time_ms, 3))

        # Update iteration stats
        for s in iteration_stats:
            if s.get("iter") == iter_no:
                s["decomp_time_ms"] = round(decomp_time_ms, 3)
                break

    # Use optimized entropy calculation
    final_entropy = shannon_entropy_from_bits(compressed_bits)
    avg_compression_pct = round(mean([it["compression_pct"] for it in iteration_stats if "compression_pct" in it]), 6) if iteration_stats else 0

    if ENABLE_MEMORY_PROFILING:
        tracemalloc.stop()

    return {
        "filename": fname,
        "qr_size_bits": qr_size,
        "qr_entropy": qr_entropy,
        "share1_size_bits": len(share1_bits),
        "share1_entropy": share1_entropy,
        "padding_added": padding_added,
        "num_iterations_successful": len(iteration_stats),
        "iteration_stats": iteration_stats,
        "bits_per_iteration": [it["compressed_bits_len"] for it in iteration_stats],
        "compression_pct_per_iteration": [it["compression_pct"] for it in iteration_stats],
        "comp_time_ms_per_iteration": comp_time_ms_per_iteration,
        "decomp_time_ms_per_iteration": decomp_time_ms_per_iteration,
        "compression_pct_avg": avg_compression_pct,
        "final_entropy": final_entropy,
        "total_compress_time_ms": round(total_compress_time_ms, 3),
        "total_decompress_time_ms": round(total_decompress_time_ms, 3),
        "peak_memory_bytes": int(peak_memory_bytes)
    }

def batch_process(dataset_dir=DATASET_DIR, output_dir=OUTPUT_DIR, max_files=MAX_FILES):
    files = sorted([os.path.join(r,f) for r,_,fs in os.walk(dataset_dir) for f in fs if f.lower().endswith(('.png','.jpg','.jpeg'))])
    if max_files: 
        files = files[:max_files]

    results = []
    start_time = time.perf_counter()
    
    for idx, f in enumerate(files, 1):
        fname = os.path.basename(f)
        print(f"\n[{idx}/{len(files)}] Processing '{fname}' ...")
        t0_file = time.perf_counter()
        try:
            res = process_file(f, output_dir)
        except Exception as e:
            res = {"filename": os.path.splitext(fname)[0], "error": f"unhandled_exception: {e}"}
        results.append(res)
        t1_file = time.perf_counter()
        elapsed = t1_file - start_time
        avg_time_per_file = elapsed / idx
        remaining_files = len(files) - idx
        est_remaining = remaining_files * avg_time_per_file
        print(f"    Completed file {idx}/{len(files)}. Time for this file: {t1_file-t0_file:.2f}s | "
              f"Elapsed: {elapsed/60:.2f}min | Estimated remaining: {est_remaining/60:.2f}min")

    # ---------------- Batch report ----------------
    rows = []
    for r in results:
        if "error" in r:
            rows.append({"filename": r["filename"], "error": r["error"]})
            continue
        row = {
            **{k:r[k] for k in ["filename","qr_size_bits","qr_entropy","share1_size_bits","share1_entropy",
                                 "padding_added","num_iterations_successful","iteration_stats","final_entropy",
                                 "compression_pct_avg","peak_memory_bytes"]},
            "bits_per_iteration": r["bits_per_iteration"],
            "compression_pct_per_iteration": r["compression_pct_per_iteration"],
            "comp_time_ms_per_iteration": r["comp_time_ms_per_iteration"],
            "decomp_time_ms_per_iteration": r["decomp_time_ms_per_iteration"],
            "total_compress_time_ms": r["total_compress_time_ms"],
            "total_decompress_time_ms": r["total_decompress_time_ms"]
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir,"batch_report.csv"),index=False)
    print(f"[+] Batch report saved to: {os.path.join(output_dir,'batch_report.csv')}")

    # ---------------- Version summary ----------------
    version_groups = defaultdict(list)
    for r in results:
        ver = infer_version_from_path(r["filename"]) or -1
        version_groups[ver].append(r)

    version_rows = []
    for ver, items in sorted(version_groups.items(), key=lambda x: x[0]):
        version_rows.append({
            "version": ver if ver!=-1 else "unknown",
            "count": len(items),
            "bits_per_iteration": [it["bits_per_iteration"] for it in items],
            "compression_pct_per_iteration": [it["compression_pct_per_iteration"] for it in items],
            "comp_time_ms_per_iteration": [it["comp_time_ms_per_iteration"] for it in items],
            "decomp_time_ms_per_iteration": [it["decomp_time_ms_per_iteration"] for it in items],
            "compression_pct_avg": round(mean([it["compression_pct_avg"] for it in items]),6),
            "final_entropy": [it["final_entropy"] for it in items],
            "qr_size_bits": [it["qr_size_bits"] for it in items],
            "qr_entropy": [it["qr_entropy"] for it in items],
            "share1_size_bits": [it["share1_size_bits"] for it in items],
            "share1_entropy": [it["share1_entropy"] for it in items],
            "padding_added": [it["padding_added"] for it in items],
            "num_iterations_successful": [it["num_iterations_successful"] for it in items],
            "total_compress_time_ms": [it["total_compress_time_ms"] for it in items],
            "total_decompress_time_ms": [it["total_decompress_time_ms"] for it in items],
            "peak_memory_bytes": [it["peak_memory_bytes"] for it in items]
        })
    pd.DataFrame(version_rows).to_csv(os.path.join(output_dir,"version_summary.csv"),index=False)
    print(f"[+] Version summary saved to: {os.path.join(output_dir,'version_summary.csv')}")

    return df, results

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    print(f"Memory profiling: {'ENABLED' if ENABLE_MEMORY_PROFILING else 'DISABLED (for speed)'}")
    df, results = batch_process()
    print(df.head(20).to_string(index=False))