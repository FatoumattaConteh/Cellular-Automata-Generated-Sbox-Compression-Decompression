"""
QR -> VC -> CA30-shuffled S-box -> Differential Distribution Table (DDT) testing

Behavior:
 - Same device-secret & QR/VC scaffolding as before.
 - For each CA-shuffled candidate S-box (grouped 6->4 mapping), compute the DDT:
     rows Δx ∈ [0..63], cols Δy ∈ [0..15]
 - Compute statistics: max nonzero DDT entry (differential uniformity), mean, worst Δx/Δy pairs.
 - Save JSON summaries and best candidate by (min max_entry, tie-break mean).
"""

import qrcode
import numpy as np
import random
import json
from PIL import Image
import math
import hashlib
import secrets
import os
import sys

# Optional keyring
try:
    import keyring
    KEYRING_AVAILABLE = True
except Exception:
    KEYRING_AVAILABLE = False

LOCAL_SECRET_PATH = "device_secret.bin"
SERVICE_NAME = "sbox_ca_ddt"
USERNAME = os.path.expanduser("~")

# ---------- Device secret helpers ----------
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
    # try keyring
    if KEYRING_AVAILABLE:
        try:
            v = keyring.get_password(SERVICE_NAME, USERNAME)
            if v is not None:
                return bytes.fromhex(v)
        except Exception:
            pass
    # try file
    s = load_secret_from_file(LOCAL_SECRET_PATH)
    if s is not None:
        return s
    # create new
    s = generate_device_secret_bytes(32)
    if KEYRING_AVAILABLE:
        try:
            keyring.set_password(SERVICE_NAME, USERNAME, s.hex())
            print("[*] Device secret stored in keyring.")
            return s
        except Exception:
            pass
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

# ---------- Base grouped S-box (your grouped mapping) ----------
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

# ---------- Helpers to flatten grouped S-box (6 -> 4 bits) ----------
def grouped_sbox_to_flat(grouped_sbox):
    flat = [None]*64
    for rule, vals in grouped_sbox.items():
        for idx, val in enumerate(vals):
            flat[val] = idx  # store 4-bit index as output
    missing = [i for i,v in enumerate(flat) if v is None]
    if missing:
        raise ValueError("Missing inputs in grouped S-box: " + str(missing))
    return flat

# ---------- Differential Distribution Table (DDT) ----------
def compute_ddt_for_grouped_sbox(grouped_sbox):
    """
    Return:
      - DDT: a list of 64 rows, each row is length 16 (counts)
      - stats: dict containing max_count (nonzero Δx), mean_count, worst pairs
    Explanation:
      For each input diff dx in 0..63:
        for x in 0..63:
          y = S(x); y2 = S(x ^ dx); dy = y ^ y2   (dy in 0..15)
          increment DDT[dx][dy]
    """
    flat = grouped_sbox_to_flat(grouped_sbox)  # S(x) ∈ 0..15
    N = 64
    M = 16
    # initialize DDT
    DDT = [ [0]*M for _ in range(N) ]
    for dx in range(N):
        for x in range(N):
            y = flat[x]
            y2 = flat[x ^ dx]
            dy = y ^ y2
            DDT[dx][dy] += 1

    # stats: for differential uniformity, consider dx != 0
    max_count = 0
    sum_counts = 0
    count_cells = 0
    worst = []
    for dx in range(1, N):
        row_max = max(DDT[dx])
        if row_max > max_count:
            max_count = row_max
        sum_counts += sum(DDT[dx])
        count_cells += len(DDT[dx])
        # record worst dy positions
        for dy, c in enumerate(DDT[dx]):
            worst.append((c, dx, dy))
    mean_count = sum_counts / count_cells if count_cells else 0
    worst.sort(reverse=True)
    top_bad = worst[:12]
    stats = {
        "max_count_nonzero_dx": max_count,
        "mean_count_nonzero_dx": mean_count,
        "top_bad": top_bad
    }
    return DDT, stats

# ---------- QR fit helper ----------
def try_embed_qr(data_str):
    try:
        qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H)
        qr.add_data(data_str)
        qr.make(fit=True)
        return True, qr.version
    except Exception:
        return False, None

# ---------- Main loop: generate many shuffled sboxes and test DDT ----------
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
    best_score = None  # prefer smaller max_count, then smaller mean_count
    results = []

    print(f"[*] Starting {MAX_TRIES} CA-shuffle attempts (CA30), computing DDT...")

    for attempt in range(1, MAX_TRIES+1):
        nonce = secrets.token_hex(12)
        seed_input = share_hash + nonce + device_secret_hex
        seed_hash = hashlib.sha256(seed_input.encode()).hexdigest()
        candidate = shuffle_sbox_with_ca(BASE_SBOX, seed_hash,
                                         key_bits_per_value=KEY_BITS,
                                         steps=CA_STEPS, cells=CA_CELLS)

        ddt, stats = compute_ddt_for_grouped_sbox(candidate)
        max_cnt = stats["max_count_nonzero_dx"]
        mean_cnt = stats["mean_count_nonzero_dx"]

        results.append({
            "attempt": attempt,
            "nonce": nonce,
            "max_count": max_cnt,
            "mean_count": mean_cnt
        })

        score = (max_cnt, mean_cnt)
        if best_candidate is None or score < best_score:
            best_candidate = candidate
            best_score = score
            best_stats = stats
            best_meta = {"attempt": attempt, "nonce": nonce, "max_count": max_cnt, "mean_count": mean_cnt}

        if attempt % report_every == 0 or attempt == 1:
            print(f" attempt {attempt:3d}: max_count={max_cnt}  mean_count={mean_cnt:.4f}  (nonce={nonce[:8]}...)")

    # Print chosen candidate summary
    print("\n--- Best CA-shuffled S-box (grouped) by DDT metric ---")
    for rule, vals in best_candidate.items():
        print(f" Rule {rule}: {vals}")
    print("--- end S-box ---\n")

    # Show some DDT diagnostics for best candidate
    ddt_best, stats_best = compute_ddt_for_grouped_sbox(best_candidate)
    print("Best candidate DDT stats:")
    print(f" max_count_nonzero_dx: {stats_best['max_count_nonzero_dx']}")
    print(f" mean_count_nonzero_dx: {stats_best['mean_count_nonzero_dx']:.4f}")
    print("Top worst (count, dx, dy):")
    for item in stats_best["top_bad"][:8]:
        print(" ", item)

    # Save results
    with open("sbox_ddt_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("best_sbox_by_ddt.json", "w") as f:
        json.dump({"meta": best_meta, "sbox": best_candidate, "best_stats": stats_best}, f, indent=2)
    # Save full DDT of best as JSON (rows are lists)
    with open("best_ddt.json", "w") as f:
        json.dump(ddt_best, f, indent=2)

    print("\n[*] Done. Best candidate saved to best_sbox_by_ddt.json and best_ddt.json")
    print("Best attempt:", best_meta)

if __name__ == "__main__":
    main()
