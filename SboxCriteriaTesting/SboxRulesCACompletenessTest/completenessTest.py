"""
QR -> VC -> CA30-shuffled S-box -> Completeness testing

Behavior:
 - Same device-secret & QR/VC scaffolding as before.
 - For each CA-shuffled candidate S-box (grouped 6->4 mapping), compute completeness:
     For each output bit j (0..3), check if flipping each input bit i (0..5)
     causes at least one change.
 - Completeness score = #influences / 24.
 - Save JSON summaries and best candidate by highest completeness.
"""

import qrcode
import numpy as np
import random
import json
from PIL import Image
import hashlib
import secrets
import os
import sys

# ---------- Device secret helpers ----------
LOCAL_SECRET_PATH = "device_secret.bin"
SERVICE_NAME = "sbox_ca_completeness"
USERNAME = os.path.expanduser("~")

def generate_device_secret_bytes(nbytes=32):
    return secrets.token_bytes(nbytes)

def store_secret_in_file(secret_bytes, path=LOCAL_SECRET_PATH):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(secret_bytes)
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass

def load_secret_from_file(path=LOCAL_SECRET_PATH):
    try:
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return None

def get_or_create_device_secret():
    s = load_secret_from_file(LOCAL_SECRET_PATH)
    if s is not None:
        return s
    s = generate_device_secret_bytes(32)
    try:
        store_secret_in_file(s, LOCAL_SECRET_PATH)
        print(f"[*] Device secret saved to {LOCAL_SECRET_PATH}")
    except Exception as e:
        print("[!] Could not store device secret to file:", e)
    return s

# ---------- QR + VC helpers ----------
def generate_qr(data, filename):
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill="black", back_color="white").convert("L")
    img.save(filename)
    img_array = np.array(img)
    return (img_array > 128).astype(np.uint8)  # 1 for black, 0 for white

def create_shares(qr_array, seed_none=None):
    rows, cols = qr_array.shape
    share1 = np.zeros((rows, cols), dtype=np.uint8)
    share2 = np.zeros((rows, cols), dtype=np.uint8)
    rng = random.Random()
    if seed_none is not None:
        rng.seed(seed_none)
    for i in range(rows):
        for j in range(cols):
            if qr_array[i, j] == 1:  # black pixel
                pat = rng.choice([[1,1],[0,0]])
            else:
                pat = rng.choice([[1,0],[0,1]])
            share1[i,j] = pat[0]
            share2[i,j] = pat[1]
    return share1, share2

# ---------- Base grouped S-box ----------
BASE_SBOX = {
    "00": [0,3,5,6,24,27,29,30,40,43,45,46,48,51,53,54],
    "01": [9,10,12,15,17,18,20,23,33,34,36,39,57,58,60,63],
    "10": [1,2,4,7,25,26,28,31,41,42,44,47,49,50,52,55],
    "11": [8,11,13,14,16,19,21,22,32,35,37,38,56,59,61,62],
}

# ---------- CA30 bit generator ----------
def ca30_bits_from_seed(seed_hex, steps=120, cells=128):
    digest = hashlib.sha256(seed_hex.encode()).digest()
    bits = ''.join(f"{b:08b}" for b in digest)
    state = [int(b) for b in bits]
    if len(state) < cells:
        times = (cells // len(state)) + 1
        state = (state * times)[:cells]
    else:
        state = state[:cells]
    out_bits = []
    for _ in range(steps):
        new = []
        for i in range(cells):
            left = state[(i-1) % cells]
            c = state[i]
            right = state[(i+1) % cells]
            new_val = left ^ (c | right)   # Rule 30
            new.append(new_val)
        state = new
        out_bits.extend(state)
    return out_bits

def shuffle_sbox_with_ca(base_sbox, seed_hex, key_bits_per_value=8, steps=120, cells=128):
    bits = ca30_bits_from_seed(seed_hex, steps=steps, cells=cells)
    shuffled = {}
    idx = 0
    for rule, vals in base_sbox.items():
        keys = []
        for _ in range(len(vals)):
            chunk = bits[idx: idx + key_bits_per_value]
            idx += key_bits_per_value
            if len(chunk) < key_bits_per_value:
                chunk = (chunk + bits)[:key_bits_per_value]
            num = int(''.join(str(b) for b in chunk), 2)
            keys.append(num)
        paired = list(zip(keys, vals.copy()))
        paired.sort(key=lambda x: x[0])
        shuffled[rule] = [v for _, v in paired]
    return shuffled

# ---------- Helpers ----------
def grouped_sbox_to_flat(grouped_sbox):
    flat = [None]*64
    for rule, vals in grouped_sbox.items():
        for idx, val in enumerate(vals):
            flat[val] = idx  # store 4-bit index as output
    missing = [i for i,v in enumerate(flat) if v is None]
    if missing:
        raise ValueError("Missing inputs in grouped S-box: " + str(missing))
    return flat

# ---------- Completeness Test ----------
def compute_completeness(grouped_sbox):
    flat = grouped_sbox_to_flat(grouped_sbox)  # maps 0..63 -> 0..15
    influences = [[False]*6 for _ in range(4)]  # [out_bit][in_bit]

    for x in range(64):
        y = flat[x]
        for i in range(6):  # input bit to flip
            x2 = x ^ (1 << i)
            y2 = flat[x2]
            dy = y ^ y2
            for j in range(4):
                if (dy >> j) & 1:
                    influences[j][i] = True

    # Count influences
    total = 0
    for j in range(4):
        for i in range(6):
            if influences[j][i]:
                total += 1
    return total, influences  # score, matrix

# ---------- Main loop ----------
def main():
    MAX_TRIES = 200
    KEY_BITS = 8
    CA_STEPS = 120
    CA_CELLS = 256
    report_every = 10

    device_secret = get_or_create_device_secret()
    if device_secret is None:
        print("[!] Cannot obtain device-secret. Exiting.")
        sys.exit(1)
    device_secret_hex = device_secret.hex()

    # Sample QR data and VC shares
    data = "Transaction ID: 12345, Amount: $90"
    qr_array = generate_qr(data, "original_qr.png")
    share1, share2 = create_shares(qr_array, seed_none=12345)
    Image.fromarray((share1*255).astype("uint8")).save("vc_share1.png")
    Image.fromarray((share2*255).astype("uint8")).save("vc_share2.png")
    share_bits = ''.join(str(b) for b in share1.ravel().tolist())
    while len(share_bits) % 6 != 0:
        share_bits += "0"
    share_hash = hashlib.sha256(share_bits.encode()).hexdigest()

    best_candidate = None
    best_score = -1
    results = []

    print(f"[*] Starting {MAX_TRIES} CA-shuffle attempts (CA30), computing Completeness...")

    for attempt in range(1, MAX_TRIES+1):
        nonce = secrets.token_hex(12)
        seed_input = share_hash + nonce + device_secret_hex
        seed_hash = hashlib.sha256(seed_input.encode()).hexdigest()
        candidate = shuffle_sbox_with_ca(BASE_SBOX, seed_hash,
                                         key_bits_per_value=KEY_BITS,
                                         steps=CA_STEPS, cells=CA_CELLS)

        score, influences = compute_completeness(candidate)

        results.append({
            "attempt": attempt,
            "nonce": nonce,
            "score": score
        })

        if score > best_score:
            best_candidate = candidate
            best_score = score
            best_meta = {"attempt": attempt, "nonce": nonce, "score": score}

        if attempt % report_every == 0 or attempt == 1:
            print(f" attempt {attempt:3d}: completeness={score}/24 (nonce={nonce[:8]}...)")

    # Print chosen candidate summary
    print("\n--- Best CA-shuffled S-box (grouped) by Completeness ---")
    for rule, vals in best_candidate.items():
        print(f" Rule {rule}: {vals}")
    print("--- end S-box ---\n")

    print("Best candidate Completeness score:", best_score, "/ 24")

    # Save results
    with open("sbox_completeness_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("best_sbox_by_completeness.json", "w") as f:
        json.dump({"meta": best_meta, "sbox": best_candidate, "score": best_score}, f, indent=2)

    print("\n[*] Done. Best candidate saved to best_sbox_by_completeness.json")
    print("Best attempt:", best_meta)

if __name__ == "__main__":
    main()
