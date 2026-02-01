"""
VC + QR + CA-shuffled S-box pipeline using a device-stored secret (no password prompt).

Behavior:
- Tries to store/retrieve a device-secret using the OS keyring (recommended).
- If keyring isn't available, falls back to storing device_secret.bin with restrictive permissions (0600).
- Uses device-secret (32 bytes) combined with share_hash and a random nonce to form CA seed:
    seed = sha256( share_hash || nonce || device_secret_hex )
- Performs CA shuffle, nonlinearity rejection, iterative compression (10 rounds), base64/QR checks, decompression, QR reconstruction.
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
    # Write secret to file and set permissions to 0o600
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(secret_bytes)
    # atomic rename
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
    """
    Returns bytes (device-secret). Tries keyring first, then local file fallback.
    """
    # 1) try keyring
    if KEYRING_AVAILABLE:
        try:
            v = keyring.get_password(SERVICE_NAME, USERNAME)
            if v is not None:
                # stored as hex
                return bytes.fromhex(v)
        except Exception:
            # fall through to file fallback
            pass

    # 2) try local file
    secret = load_secret_from_file(LOCAL_SECRET_PATH)
    if secret is not None:
        return secret

    # 3) create new secret and store
    secret = generate_device_secret_bytes(32)
    # try to store in keyring
    if KEYRING_AVAILABLE:
        try:
            keyring.set_password(SERVICE_NAME, USERNAME, secret.hex())
            print("[*] Device secret stored in OS keyring.")
            return secret
        except Exception:
            pass

    # fallback to local file
    try:
        store_secret_in_file(secret, LOCAL_SECRET_PATH)
        print(f"[*] Device secret saved to local file '{LOCAL_SECRET_PATH}' (permissions set to 600 if supported).")
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
# CA (Rule 30) RNG & shuffle
# ----------------------------
def generate_ca_bits(seed: str, steps: int = 120, cells: int = 128):
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

def shuffle_sbox_with_ca(original_sbox, seed: str, steps: int = 120, cells: int = 128):
    bits = generate_ca_bits(seed, steps=steps, cells=cells)
    shuffled = {}
    idx = 0
    for rule, values in original_sbox.items():
        rand_nums = []
        for _ in range(len(values)):
            chunk = bits[idx: idx+4]
            idx += 4
            if len(chunk) < 4:
                chunk = bits[:4]
            rand_nums.append(int(''.join(map(str, chunk)), 2))
        paired = list(zip(rand_nums, values.copy()))
        paired.sort(key=lambda x: x[0])
        shuffled[rule] = [val for _, val in paired]
    return shuffled

# ----------------------------
# Compression / Decompression helpers (bit-string)
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
# Base64 bit-preserving wrappers
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
# Walsh / Nonlinearity utilities
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
# QR fit helper
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
    MAX_TRIES = 200
    CA_STEPS = 120
    CA_CELLS = 128
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
    print(f"[*] Searching for shuffled S-box with min nonlinearity >= {NL_THRESHOLD}")
    chosen_nonce = None; chosen_sbox = None; chosen_min_nl = None
    for attempt in range(1, MAX_TRIES+1):
        nonce = secrets.token_hex(16)
        seed_input = share_hash + nonce + device_secret_hex
        seed_hash = hashlib.sha256(seed_input.encode()).hexdigest()
        candidate = shuffle_sbox_with_ca(BASE_SBOX, seed=seed_hash, steps=CA_STEPS, cells=CA_CELLS)
        min_nl, min_mask = test_sbox_nonlinearity(candidate)
        if attempt % 10 == 0 or min_nl < NL_THRESHOLD:
            print(f" attempt {attempt}: min_nl={min_nl} mask={format(min_mask,'04b')} nonce={nonce[:8]}...")
        if min_nl >= NL_THRESHOLD:
            chosen_nonce = nonce
            chosen_sbox = candidate
            chosen_min_nl = min_nl
            print(f"[*] Acceptable S-box found at attempt {attempt} (min_nl={min_nl})")
            break

    if chosen_sbox is None:
        print(f"[!] No acceptable S-box found after {MAX_TRIES} tries — consider raising CA strength or lowering threshold.")
        sys.exit(1)

    # Save metadata (nonce, CA params). device_secret NOT saved here.
    meta = {
        "nonce": chosen_nonce,
        "ca_steps": CA_STEPS,
        "ca_cells": CA_CELLS,
        "nl_observed": chosen_min_nl
    }
    with open("sbox_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Print chosen shuffled sbox
    print("\n--- Chosen shuffled S-BOX ---")
    for r,v in chosen_sbox.items():
        print(f"Rule {r}: {v}")
    print("--- end S-BOX ---\n")

    # Compression iterations using chosen_sbox
    compressed_bits = share_bits
    all_rules = []
    for i in range(ITERATIONS):
        compressed_bits, rules = compress_bits(compressed_bits, chosen_sbox)
        all_rules.append(rules)
        b64meta = bits_to_b64_json(compressed_bits)
        with open(f"compressed_iter_{i+1}.json", "w") as f:
            json.dump(b64meta, f)
        with open(f"rules_iter_{i+1}.json", "w") as f:
            json.dump(rules, f)
        print(f"Iteration {i+1}: compressed bits={len(compressed_bits)}, rules len={len(json.dumps(rules))}, base64 len={len(b64meta['b64'])}")
        fit_b64, ver_b64 = try_embed_qr(b64meta["b64"])
        print(f"Iteration {i+1}: QR fit (base64) = {fit_b64}" + (f", version {ver_b64}" if fit_b64 else ""))
        test_decompressed_bits = decompress_bits(compressed_bits, rules, chosen_sbox)
        with open(f"decompressed_iter_{i+1}.txt", "w") as f:
            f.write(test_decompressed_bits)
        print(f"Iteration {i+1}: decompressed bits length = {len(test_decompressed_bits)}\n")

    # Full decompression (reverse)
    decompressed_bits = None
    for i in range(ITERATIONS-1, -1, -1):
        with open(f"compressed_iter_{i+1}.json") as f:
            b64meta = json.load(f)
        with open(f"rules_iter_{i+1}.json") as f:
            rules = json.load(f)
        compressed_bits_i = b64_json_to_bits(b64meta)
        decompressed_bits = decompress_bits(compressed_bits_i, rules, chosen_sbox)

    # Remove padding
    with open("padding_info.json") as f:
        pinfo = json.load(f)
    final_decompressed = decompressed_bits[:pinfo["original_length"]]

    # Save restored share and reconstructed QR
    restored_array = np.array([int(b) for b in final_decompressed]).reshape(share1.shape)
    Image.fromarray((restored_array*255).astype(np.uint8)).save("restored_share1.png")
    with open("decompressed_share1.txt", "w") as f:
        f.write(final_decompressed)
    reconstructed_qr = np.logical_xor(restored_array, share2).astype(np.uint8)
    Image.fromarray((reconstructed_qr*255).astype(np.uint8)).save("reconstructed_qr.png")
    print("[*] Pipeline complete. Final decompressed share1 length:", len(final_decompressed))
    print("Metadata saved to sbox_metadata.json. Device-secret remains on device (keyring or local file).")

if __name__ == "__main__":
    main()
