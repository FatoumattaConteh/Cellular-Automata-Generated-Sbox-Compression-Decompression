"""
VC + QR + CA30 + Fisher–Yates S-box pipeline (nonlinearity testing stage).

- Device-secret management (keyring preferred; file fallback).
- Generates QR + VC shares to produce share_bits.
- Uses SHA256(share_hash || nonce || device_secret_hex) as seed.
- Uses CA Rule 30 to produce bitstream, then Fisher–Yates (per-rule) to shuffle S-box.
- Tests nonlinearity (Walsh transform) and accepts candidate if min_nl >= NL_THRESHOLD.
- If accepted: saves chosen sbox and metadata. Otherwise saves best candidate for inspection.
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
import sys
import time
import statistics

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
# CA (Rule 30) RNG & Fisher–Yates shuffle helpers
# ----------------------------
def generate_ca_bits(seed: str, steps: int = 120, cells: int = 128):
    """
    Produce a list of bits using CA Rule 30 seeded by SHA256(seed).
    Output length = steps * cells
    """
    digest = hashlib.sha256(seed.encode()).digest()
    bits = ''.join(f'{byte:08b}' for byte in digest)
    state = [int(b) for b in bits[:cells]]
    if len(state) < cells and len(state) > 0:
        state = (state * ((cells // len(state)) + 1))[:cells]
    output_bits = []
    for _ in range(steps):
        new_state = []
        for i in range(cells):
            left = state[(i - 1) % cells]
            center = state[i]
            right = state[(i + 1) % cells]
            new_val = left ^ (center | right)   # Rule 30
            new_state.append(new_val)
        state = new_state
        output_bits.extend(state)
    return output_bits

def get_bits_as_int(bits, ptr, k):
    """Read k bits from bits at pointer ptr (wraps around); returns (value, new_ptr)."""
    n = len(bits)
    if n == 0:
        return 0, ptr
    # gather k bits, wrap as needed
    out = 0
    for i in range(k):
        b = bits[(ptr + i) % n]
        out = (out << 1) | b
    return out, (ptr + k) % n

def fisher_yates_with_ca(values, bits, start_ptr=0):
    """
    Fisher–Yates shuffle values[] using CA-bitstream 'bits' as RNG.
    Uses rejection sampling to get uniform indices in [0..i].
    Returns shuffled list and updated pointer.
    """
    arr = values[:]  # copy
    n = len(arr)
    ptr = start_ptr
    for i in range(n-1, 0, -1):
        # need an integer in range [0, i]
        k = (i+1).bit_length()  # number of bits needed
        attempts = 0
        while True:
            val, ptr = get_bits_as_int(bits, ptr, k)
            attempts += 1
            if val <= i:
                j = val
                break
            # rejection: try again (ptr advanced already)
            if attempts > 1000:
                # fallback: use Python RNG seeded from bits chunk
                seed_chunk = 0
                for _ in range(8):
                    v, ptr = get_bits_as_int(bits, ptr, 8)
                    seed_chunk = (seed_chunk << 8) ^ v
                local_rng = random.Random(seed_chunk)
                j = local_rng.randint(0, i)
                break
        arr[i], arr[j] = arr[j], arr[i]
    return arr, ptr

def shuffle_sbox_with_ca(original_sbox, seed: str, steps: int = 120, cells: int = 128):
    """
    Use CA30 bitstream (generate_ca_bits) and Fisher–Yates to shuffle each rule's list.
    Returns shuffled grouped S-box (dict).
    """
    bits = generate_ca_bits(seed, steps=steps, cells=cells)
    shuffled = {}
    ptr = 0
    for rule, values in original_sbox.items():
        shuffled_vals, ptr = fisher_yates_with_ca(values.copy(), bits, start_ptr=ptr)
        shuffled[rule] = shuffled_vals
    return shuffled

# ----------------------------
# Compression / Decompression helpers (bit-string) - unchanged
# ----------------------------
def build_reverse_sbox(sbox):
    rev = {}
    for rule, vals in sbox.items():
        for idx, v in enumerate(vals):
            rev[v] = (rule, idx)
    return rev

def compress_bits(bits, sbox):
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

def decompress_bits(compressed_bits, rules, sbox):
    out = []
    for i, rule in enumerate(rules):
        start = i*4
        idx = int(compressed_bits[start:start+4], 2)
        val = sbox[rule][idx]
        out.append(format(val, '06b'))
    return ''.join(out)

# ----------------------------
# Base64 bit-preserving wrappers - unchanged
# ----------------------------
def bits_to_b64_json(bits):
    if bits == "":
        return {"b64": "", "bitlen": 0}
    b = int(bits, 2).to_bytes((len(bits) + 7)//8, byteorder='big')
    return {"b64": base64.b64encode(b).decode(), "bitlen": len(bits)}

def b64_json_to_bits(b64json):
    b64 = b64json["b64"]
    bitlen = b64json["bitlen"]
    if b64 == "" or bitlen == 0:
        return ""
    raw = base64.b64decode(b64)
    bits_full = ''.join(f"{byte:08b}" for byte in raw)
    return bits_full[-bitlen:]

# ----------------------------
# Walsh / Nonlinearity utilities - unchanged
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
# QR fit helper - unchanged
# ----------------------------
def try_embed_qr(data_str):
    try:
        qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H)
        qr.add_data(data_str)
        qr.make(fit=True)
        return True, qr.version
    except Exception:
        return False, None

# ----------------------------
# Main pipeline
# ----------------------------
def main():
    # Parameters
    NL_THRESHOLD = 22
    MAX_TRIES = 100
    CA_STEPS = 500      # you can raise this to increase mixing
    CA_CELLS = 300      # increase cell count for more entropy
    ITERATIONS = 10

    # Get device-secret (no password prompt)
    device_secret = get_or_create_device_secret()
    if device_secret is None:
        print("[!] Failed to obtain or create device-secret. Exiting.")
        sys.exit(1)
    device_secret_hex = device_secret.hex()
    print("[*] Device-secret ready (kept on device).")

    # Generate QR and VC shares
    data = "Transaction ID: 12345, Amount: $90"
    qr_array = generate_qr(data, "original_qr.png")
    print("[*] QR generated")
    share1, share2 = create_shares(qr_array)
    Image.fromarray((share1*255).astype(np.uint8)).save("share1.png")
    Image.fromarray((share2*255).astype(np.uint8)).save("share2.png")
    print("[*] Shares saved")

    share_bits = ''.join(str(b) for b in share1.ravel().tolist())
    original_length = len(share_bits)
    while len(share_bits) % 6 != 0:
        share_bits += '0'
    padding_added = len(share_bits) - original_length
    with open("padding_info.json", "w") as f:
        json.dump({"padding": padding_added, "original_length": original_length}, f)

    share_hash = hashlib.sha256(share_bits.encode()).hexdigest()

    # Rejection sampling to find strong shuffled sbox
    print(f"[*] Searching for shuffled S-box with min nonlinearity >= {NL_THRESHOLD} (CA30 + Fisher–Yates)")
    chosen_nonce = None; chosen_sbox = None; chosen_min_nl = None
    best_candidate = None; best_nl = -1
    nl_values = []

    start_time = time.time()
    for attempt in range(1, MAX_TRIES+1):
        nonce = secrets.token_hex(16)
        seed_input = share_hash + nonce + device_secret_hex
        seed_hash = hashlib.sha256(seed_input.encode()).hexdigest()
        candidate = shuffle_sbox_with_ca(BASE_SBOX, seed=seed_hash, steps=CA_STEPS, cells=CA_CELLS)
        min_nl, min_mask = test_sbox_nonlinearity(candidate)
        nl_values.append(min_nl)
        if min_nl > best_nl:
            best_nl = min_nl
            best_candidate = candidate
            best_nonce = nonce
        if attempt % 10 == 0 or min_nl < NL_THRESHOLD:
            print(f" attempt {attempt}: min_nl={min_nl} mask={format(min_mask,'04b')} nonce={nonce[:8]}...")
        if min_nl >= NL_THRESHOLD:
            chosen_nonce = nonce
            chosen_sbox = candidate
            chosen_min_nl = min_nl
            print(f"[*] Acceptable S-box found at attempt {attempt} (min_nl={min_nl})")
            break

    elapsed = time.time() - start_time
    if nl_values:
        print("\n=== Summary ===")
        print(f"Attempts: {len(nl_values)}, elapsed: {elapsed:.1f}s")
        print(f"min: {min(nl_values)}, max: {max(nl_values)}, mean: {statistics.mean(nl_values):.2f}, median: {statistics.median(nl_values)}, std: {statistics.pstdev(nl_values):.2f}")
    else:
        print("[!] No attempts recorded (unexpected)")

    if chosen_sbox is None:
        print(f"[!] No acceptable S-box found after {MAX_TRIES} tries — saved best candidate with min_nl={best_nl}")
        if best_candidate is not None:
            with open("best_sbox_candidate.json", "w") as f:
                json.dump(best_candidate, f, indent=2)
            with open("best_nonce.txt", "w") as f:
                f.write(best_nonce)
    else:
        meta = {
            "nonce": chosen_nonce,
            "ca_steps": CA_STEPS,
            "ca_cells": CA_CELLS,
            "nl_observed": chosen_min_nl
        }
        with open("chosen_sbox.json", "w") as f:
            json.dump(chosen_sbox, f, indent=2)
        with open("sbox_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        print("[*] Chosen S-box + metadata saved (for later decompression).")

    # Print chosen or best S-box for inspection
    print("\n--- Best / Chosen S-BOX (for inspection) ---")
    box_to_print = chosen_sbox if chosen_sbox is not None else best_candidate
    if box_to_print is not None:
        for r,v in box_to_print.items():
            print(f"Rule {r}: {v}")
    print("--- end S-BOX ---\n")

    # (Optional) proceed with compression using chosen_sbox if desired
    if chosen_sbox is not None:
        print("[*] You can continue with compression/decompression phases using chosen_sbox.")
    else:
        print("[*] No chosen S-box; inspect best_sbox_candidate.json for next steps.")

if __name__ == "__main__":
    main()
