"""
VC + QR + DRBG-shuffled S-box pipeline using a device-stored secret (no password prompt).

Changes:
- Replaced CA-based shuffle with HMAC-DRBG + Fisher-Yates shuffle.
- DRBG is seeded with: sha256(share_hash || nonce || device_secret).
- Everything else remains the same (compression, QR fit, nonlinearity test).
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
import hmac
import struct

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
SERVICE_NAME = "sbox_drbg_pipeline"
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
    if KEYRING_AVAILABLE:
        try:
            v = keyring.get_password(SERVICE_NAME, USERNAME)
            if v is not None:
                return bytes.fromhex(v)
        except Exception:
            pass
    secret = load_secret_from_file(LOCAL_SECRET_PATH)
    if secret is not None:
        return secret
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
# DRBG (HMAC-DRBG with SHA256)
# ----------------------------
class HMAC_DRBG:
    def __init__(self, seed: bytes):
        self.K = b"\x00" * 32
        self.V = b"\x01" * 32
        self.reseed(seed)

    def reseed(self, seed: bytes):
        self.K = hmac.new(self.K, self.V + b"\x00" + seed, hashlib.sha256).digest()
        self.V = hmac.new(self.K, self.V, hashlib.sha256).digest()
        if seed and len(seed) > 0:
            self.K = hmac.new(self.K, self.V + b"\x01" + seed, hashlib.sha256).digest()
            self.V = hmac.new(self.K, self.V, hashlib.sha256).digest()

    def generate(self, nbytes: int) -> bytes:
        output = b""
        while len(output) < nbytes:
            self.V = hmac.new(self.K, self.V, hashlib.sha256).digest()
            output += self.V
        return output[:nbytes]

def fisher_yates_shuffle(values, drbg: HMAC_DRBG):
    arr = values.copy()
    n = len(arr)
    for i in range(n - 1, 0, -1):
        r_bytes = drbg.generate(4)
        r = struct.unpack(">I", r_bytes)[0] % (i + 1)
        arr[i], arr[r] = arr[r], arr[i]
    return arr

def shuffle_sbox_with_drbg(original_sbox, seed: str):
    seed_bytes = hashlib.sha256(seed.encode()).digest()
    drbg = HMAC_DRBG(seed_bytes)
    shuffled = {}
    for rule, values in original_sbox.items():
        shuffled[rule] = fisher_yates_shuffle(values, drbg)
    return shuffled

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
# Main pipeline (nonlinearity test only for now)
# ----------------------------
def main():
    NL_THRESHOLD = 22
    MAX_TRIES = 100

    device_secret = get_or_create_device_secret()
    if device_secret is None:
        print("[!] Failed to obtain device-secret.")
        sys.exit(1)
    device_secret_hex = device_secret.hex()
    print("[*] Device-secret ready.")

    # Simulate share hash
    fake_bits = "101010" * 1000
    share_hash = hashlib.sha256(fake_bits.encode()).hexdigest()

    chosen_nonce = None; chosen_sbox = None; chosen_min_nl = None
    best_candidate = None; best_min_nl = -1

    for attempt in range(1, MAX_TRIES+1):
        nonce = secrets.token_hex(16)
        seed_input = share_hash + nonce + device_secret_hex
        seed_hash = hashlib.sha256(seed_input.encode()).hexdigest()
        candidate = shuffle_sbox_with_drbg(BASE_SBOX, seed=seed_hash)
        min_nl, min_mask = test_sbox_nonlinearity(candidate)
        if min_nl > best_min_nl:
            best_min_nl = min_nl
            best_candidate = candidate
        if attempt % 10 == 0 or min_nl >= NL_THRESHOLD:
            print(f" attempt {attempt}: min_nl={min_nl}, mask={format(min_mask,'04b')} nonce={nonce[:8]}...")
        if min_nl >= NL_THRESHOLD:
            chosen_nonce = nonce
            chosen_sbox = candidate
            chosen_min_nl = min_nl
            print(f"[*] Acceptable S-box found at attempt {attempt} (min_nl={min_nl})")
            break

    if chosen_sbox is None:
        print(f"[!] No acceptable S-box found after {MAX_TRIES} tries — best candidate has min_nl={best_min_nl}")
        with open("best_sbox_candidate.json", "w") as f:
            json.dump(best_candidate, f, indent=2)
    else:
        print("\n--- Chosen shuffled S-BOX ---")
        for r,v in chosen_sbox.items():
            print(f"Rule {r}: {v}")
        print("--- end S-BOX ---\n")

if __name__ == "__main__":
    main()
