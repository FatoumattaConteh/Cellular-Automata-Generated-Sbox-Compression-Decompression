"""
S-box multi-CA nonlinearity tester.

Behavior:
- Device-secret management (keyring preferred; fallback local file).
- Generate QR + VC share (same pipeline as before) to produce share_bits.
- For up to MAX_TRIES:
    nonce <- random
    seed = SHA256(share_hash || nonce || device_secret_hex)
    candidate_sbox <- shuffled by multi-rule CA (rules: 30,45,90,150)
    compute min nonlinearity across masks -> min_nl
    record stats; accept if min_nl >= NL_THRESHOLD
- Print summary and save best candidate & metadata if found.
"""

import qrcode
import numpy as np
import random
import json
from PIL import Image
import math
import hashlib
import secrets
import base64
import os
import stat
import sys
import statistics
import time

# Try to import keyring (optional, but recommended)
try:
    import keyring
    KEYRING_AVAILABLE = True
except Exception:
    KEYRING_AVAILABLE = False

# ----------------------------
# Utilities
# ----------------------------
def shannon_entropy(data):
    if isinstance(data, np.ndarray):
        data = data.ravel().tolist()
    if not data:
        return 0.0
    freq = {}
    for symbol in data:
        freq[symbol] = freq.get(symbol, 0) + 1
    entropy = 0.0
    length = len(data)
    for count in freq.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy

# ----------------------------
# Device-secret management
# ----------------------------
SERVICE_NAME = "sbox_ca_pipeline"
USERNAME = os.path.expanduser("~")  # use home dir as "username" key in keyring
LOCAL_SECRET_PATH = "device_secret.bin"

def generate_device_secret_bytes(nbytes=32):
    return secrets.token_bytes(nbytes)

def store_secret_in_file(secret_bytes, path=LOCAL_SECRET_PATH):
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(secret_bytes)
    os.replace(tmp_path, path)
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass

def load_secret_from_file(path=LOCAL_SECRET_PATH):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            data = f.read()
        return data
    except Exception:
        return None

def get_or_create_device_secret():
    # 1) try keyring
    if KEYRING_AVAILABLE:
        try:
            v = keyring.get_password(SERVICE_NAME, USERNAME)
            if v is not None:
                return bytes.fromhex(v)
        except Exception:
            pass
    # 2) try local file
    secret = load_secret_from_file(LOCAL_SECRET_PATH)
    if secret is not None:
        return secret
    # 3) create and store
    secret = generate_device_secret_bytes(32)
    if KEYRING_AVAILABLE:
        try:
            keyring.set_password(SERVICE_NAME, USERNAME, secret.hex())
            print("[*] Device secret stored in OS keyring.")
            return secret
        except Exception:
            pass
    try:
        store_secret_in_file(secret, LOCAL_SECRET_PATH)
        print(f"[*] Device secret saved to local file '{LOCAL_SECRET_PATH}'.")
    except Exception as e:
        print("[!] Warning: failed to save device secret to local file:", e)
    return secret

# ----------------------------
# QR + VC helpers
# ----------------------------
def generate_qr(data, filename="original_qr.png"):
    qr = qrcode.QRCode(version=None, error_correction=qrcode.constants.ERROR_CORRECT_H)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill="black", back_color="white")
    img.save(filename)
    return np.array(img, dtype=np.uint8)

def create_shares(qr_array):
    rows, cols = qr_array.shape
    share1 = np.zeros((rows, cols), dtype=np.uint8)
    share2 = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            if qr_array[i, j] == 0:  # black pixel
                pattern = random.choice([[0, 0], [1, 1]])
            else:
                pattern = random.choice([[0, 1], [1, 0]])
            share1[i, j] = pattern[0]
            share2[i, j] = pattern[1]
    return share1, share2

# ----------------------------
# Base S-box (complete coverage 0..63 across 4 rules)
# ----------------------------
BASE_SBOX = {
    "00": [0,3,5,6,24,27,29,30,40,43,45,46,48,51,53,54],
    "01": [9,10,12,15,17,18,20,23,33,34,36,39,57,58,60,63],
    "10": [1,2,4,7,25,26,28,31,41,42,44,47,49,50,52,55],
    "11": [8,11,13,14,16,19,21,22,32,35,37,38,56,59,61,62],
}

# ----------------------------
# Multi-rule CA helpers (Wolfram rules)
# ----------------------------
def evolve_rule_once(state, rule_number):
    """Evolve one step of a 1D binary cellular automaton (periodic boundary)
       using Wolfram rule_number (0..255)."""
    rule_bin = f"{rule_number:08b}"[::-1]  # map pattern (111..000) to index 0..7
    n = len(state)
    new = [0]*n
    for i in range(n):
        left = state[(i-1) % n]
        center = state[i]
        right = state[(i+1) % n]
        pattern = (left << 2) | (center << 1) | right
        new[i] = int(rule_bin[pattern])
    return new

def generate_multi_ca_bits(seed: str, steps: int = 120, cells: int = 128, rule_sequence=(30,45,90,150)):
    """Generate a long bit stream by cascading multiple CA rules each step."""
    digest = hashlib.sha256(seed.encode()).digest()
    bits = ''.join(f'{byte:08b}' for byte in digest)
    # initialize state by repeating/truncating bits
    state = [int(b) for b in bits]
    if len(state) < cells:
        # repeat to reach cells
        state = (state * ((cells // len(state)) + 1))[:cells]
    else:
        state = state[:cells]
    output = []
    for _ in range(steps):
        for r in rule_sequence:
            state = evolve_rule_once(state, r)
        output.extend(state)
    return output

def shuffle_sbox_with_multi_ca(original_sbox, seed: str, steps: int = 120, cells: int = 128, rule_sequence=(30,45,90,150)):
    bits = generate_multi_ca_bits(seed, steps=steps, cells=cells, rule_sequence=rule_sequence)
    shuffled = {}
    idx = 0
    for rule, values in original_sbox.items():
        rand_nums = []
        for _ in range(len(values)):
            chunk = bits[idx: idx+4]
            idx += 4
            if len(chunk) < 4:
                # wrap-around if needed
                chunk = bits[:4]
            rand_nums.append(int(''.join(map(str, chunk)), 2))
        paired = list(zip(rand_nums, values.copy()))
        paired.sort(key=lambda x: x[0])
        shuffled[rule] = [val for _, val in paired]
    return shuffled

# ----------------------------
# Walsh / Nonlinearity utilities (unchanged)
# ----------------------------
def fwht(a):
    n = len(a)
    res = a[:]
    h = 1
    while h < n:
        for i in range(0, n, h*2):
            for j in range(i, i+h):
                x = res[j]; y = res[j+h]
                res[j] = x + y
                res[j+h] = x - y
        h *= 2
    return res

def grouped_sbox_to_flat(grouped_sbox):
    flat = [None]*64
    for rule, vals in grouped_sbox.items():
        for idx, original_val in enumerate(vals):
            flat[original_val] = idx
    missing = [i for i,v in enumerate(flat) if v is None]
    if missing:
        raise ValueError(f"Grouped S-box missing inputs: {missing}")
    return flat

def compute_nonlinearity_for_mask(flat_sbox, mask):
    n = 6; N = 1<<n
    tt = [0]*N
    for x in range(N):
        out = flat_sbox[x]
        bit = bin(out & mask).count("1") & 1
        tt[x] = 1 if bit else 0
    F = [1 if b==0 else -1 for b in tt]
    W = fwht(F)
    W_abs_max = max(abs(v) for v in W)
    NL = (1 << (n-1)) - (W_abs_max // 2)
    return NL

def test_sbox_nonlinearity(grouped_sbox):
    flat = grouped_sbox_to_flat(grouped_sbox)
    min_nl = None; min_mask = None
    for mask in range(1, 1<<4):
        nl = compute_nonlinearity_for_mask(flat, mask)
        if min_nl is None or nl < min_nl:
            min_nl = nl; min_mask = mask
    return min_nl, min_mask

# ----------------------------
# Main test driver
# ----------------------------
def main():
    NL_THRESHOLD = 22        # desired min nonlinearity (tuneable)
    MAX_TRIES = 200          # how many nonces to try
    CA_STEPS = 120
    CA_CELLS = 128
    RULE_SEQUENCE = (30,45,90,150)

    # obtain device secret
    device_secret = get_or_create_device_secret()
    if device_secret is None:
        print("[!] Couldn't get device-secret. Exiting.")
        sys.exit(1)
    device_secret_hex = device_secret.hex()
    print("[*] Device-secret ready (kept on device).")

    # make QR + VC shares (we only need share_bits for seed mixing)
    data = "Transaction ID: 12345, Amount: $90"
    qr_array = generate_qr(data, "original_qr.png")
    print("[*] QR generated.")
    share1, share2 = create_shares(qr_array)
    Image.fromarray((share1*255).astype(np.uint8)).save("share1.png")
    Image.fromarray((share2*255).astype(np.uint8)).save("share2.png")
    print("[*] Shares saved.")

    share_bits = ''.join(str(b) for b in share1.ravel().tolist())
    original_length = len(share_bits)
    while len(share_bits) % 6 != 0:
        share_bits += '0'
    padding_added = len(share_bits) - original_length
    with open("padding_info.json", "w") as f:
        json.dump({"padding": padding_added, "original_length": original_length}, f)

    share_hash = hashlib.sha256(share_bits.encode()).hexdigest()

    # try many nonces and collect NL stats
    nl_list = []
    attempts_info = []
    best_candidate = None
    best_nl = -1
    chosen_nonce = None
    start_time = time.time()
    print(f"[*] Searching for shuffled S-box with min nonlinearity >= {NL_THRESHOLD} using multi-rule CA {RULE_SEQUENCE}")
    for attempt in range(1, MAX_TRIES+1):
        nonce = secrets.token_hex(16)
        seed_input = share_hash + nonce + device_secret_hex
        seed_hash = hashlib.sha256(seed_input.encode()).hexdigest()
        candidate = shuffle_sbox_with_multi_ca(BASE_SBOX, seed=seed_hash, steps=CA_STEPS, cells=CA_CELLS, rule_sequence=RULE_SEQUENCE)
        min_nl, min_mask = test_sbox_nonlinearity(candidate)
        nl_list.append(min_nl)
        attempts_info.append((attempt, nonce, min_nl, min_mask))
        if min_nl > best_nl:
            best_nl = min_nl
            best_candidate = candidate
            best_nonce = nonce
        # print progress for problematic or every 10th attempt
        if attempt % 10 == 0 or min_nl < NL_THRESHOLD:
            print(f" attempt {attempt}: min_nl={min_nl} mask={format(min_mask,'04b')} nonce={nonce[:8]}...")
        if min_nl >= NL_THRESHOLD:
            chosen_nonce = nonce
            chosen_sbox = candidate
            chosen_min_nl = min_nl
            print(f"[*] Acceptable S-box found on attempt {attempt}: min_nl={min_nl}")
            break

    elapsed = time.time() - start_time

    # summary stats
    if nl_list:
        s_min = min(nl_list); s_max = max(nl_list); s_mean = statistics.mean(nl_list)
        s_med = statistics.median(nl_list); s_std = statistics.pstdev(nl_list)
    else:
        s_min = s_max = s_mean = s_med = s_std = None

    print("\n=== Nonlinearity search summary ===")
    print(f"Attempts: {len(nl_list)} | Time: {elapsed:.1f}s")
    print(f"min: {s_min}, max: {s_max}, mean: {s_mean:.2f}, median: {s_med}, stddev: {s_std:.2f}")
    weak = sum(1 for v in nl_list if v < NL_THRESHOLD)
    print(f"Weak (<{NL_THRESHOLD}): {weak} / {len(nl_list)}")

    # Save best candidate and attempt log
    if best_candidate is not None:
        with open("best_sbox_candidate.json", "w") as f:
            json.dump(best_candidate, f, indent=2)
        with open("attempts_nl_log.json", "w") as f:
            json.dump([{"attempt":a,"nonce":n,"min_nl":nl,"mask":m} for (a,n,nl,m) in attempts_info], f, indent=2)
        print(f"Saved best candidate (min_nl={best_nl}) to best_sbox_candidate.json")

    if chosen_nonce is not None:
        # Save chosen sbox + metadata
        meta = {
            "nonce": chosen_nonce,
            "ca_steps": CA_STEPS,
            "ca_cells": CA_CELLS,
            "rule_sequence": RULE_SEQUENCE,
            "nl_observed": chosen_min_nl
        }
        with open("chosen_sbox.json", "w") as f:
            json.dump(chosen_sbox, f, indent=2)
        with open("sbox_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        print("[*] Chosen S-box + metadata saved (for later decompression).")
    else:
        print("[!] No acceptable S-box found in attempt budget. Best candidate saved for inspection.")

if __name__ == "__main__":
    main()
