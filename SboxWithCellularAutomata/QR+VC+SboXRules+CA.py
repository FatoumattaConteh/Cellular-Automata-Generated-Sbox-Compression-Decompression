"""
Full Pipeline: CA-Rule-90–Driven Dynamic S-Box + Visual Cryptography + QR Integration
Updated: use Share1 hash as the 'message' for CA seed derivation.
"""

import qrcode
import numpy as np
import json
import random
from PIL import Image
import math
import hashlib
import secrets
import base64
import os

# ----------------------------
# Shannon Entropy
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
# QR generation helper
# ----------------------------
def generate_qr(data, filename="qr_code.png"):
    qr = qrcode.QRCode(version=None, error_correction=qrcode.constants.ERROR_CORRECT_H)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill="black", back_color="white")
    img.save(filename)
    return np.array(img, dtype=np.uint8)


# ----------------------------
# Visual cryptography shares
# ----------------------------
def create_shares(qr_array):
    rows, cols = qr_array.shape
    share1 = np.zeros((rows, cols), dtype=np.uint8)
    share2 = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            if qr_array[i, j] == 0:  # black pixel
                pattern = random.choice([[0, 0], [1, 1]])
            else:  # white pixel
                pattern = random.choice([[0, 1], [1, 0]])
            share1[i, j] = pattern[0]
            share2[i, j] = pattern[1]
    return share1, share2


# ----------------------------
# Base S-Box (covering 0..63 across 4 rules)
# ----------------------------
BASE_SBOX = {
    "00": [0,3,5,6,24,27,29,30,40,43,45,46,48,51,53,54],
    "01": [9,10,12,15,17,18,20,23,33,34,36,39,57,58,60,63],
    "10": [1,2,4,7,25,26,28,31,41,42,44,47,49,50,52,55],
    "11": [8,11,13,14,16,19,21,22,32,35,37,38,56,59,61,62],
}


# ----------------------------
# Enhanced CA Seed Generation
# ----------------------------
def generate_enhanced_ca_seed(device_secret: str, nonce: str, message: str, ca_length: int = 256):
    """
    Generate seed using SHA-256(device_secret || nonce || message_hash_or_value)
    message here is expected to be a string (we will pass share1_hash)
    """
    # H_in = SHA256(message) is computed inside as in your design
    H_in = hashlib.sha256(message.encode()).hexdigest()
    I = device_secret + nonce + H_in
    S = hashlib.sha256(I.encode()).hexdigest()
    # convert hex to bitstring
    S_bits = ''.join(f'{int(c, 16):04b}' for c in S)
    if ca_length <= len(S_bits):
        S_CA = S_bits[:ca_length]
    else:
        # repeat the bitstring to reach the requested length
        S_CA = (S_bits * ((ca_length // len(S_bits)) + 1))[:ca_length]
    return S_CA


# ----------------------------
# Rule Stream Generator (1–4)  -- kept (unused by default)
# ----------------------------
def generate_rule_stream(ca_seed: str, num_rules: int):
    rule_stream = []
    for t in range(num_rules):
        start_idx = 2 * t
        end_idx = 2 * t + 2
        if end_idx > len(ca_seed):
            remaining = end_idx - len(ca_seed)
            r_bits = ca_seed[start_idx:] + ca_seed[:remaining]
        else:
            r_bits = ca_seed[start_idx:end_idx]
        rule_index = 1 + int(r_bits, 2)  # maps '00'->1 ... '11'->4
        rule_stream.append(rule_index)
    return rule_stream


# ----------------------------
# Cellular Automata (Rule 90)
# ----------------------------
def generate_ca_bits_rule90(seed_bits: str, steps: int = 50, cells: int = 64):
    """
    Rule 90: s_i(t+1) = s_{i−1}(t) XOR s_{i+1}(t)
    Produces steps * cells output bits (concatenated)
    """
    # initial state: use the first 'cells' bits of seed_bits (repeat if shorter)
    state = [int(bit) for bit in seed_bits[:cells]]
    if len(state) < cells:
        state = (state * ((cells // len(state)) + 1))[:cells]

    output_bits = []
    for _ in range(steps):
        new_state = []
        for i in range(cells):
            left = state[(i - 1) % cells]
            right = state[(i + 1) % cells]
            new_val = left ^ right  # Rule 90
            new_state.append(new_val)
        state = new_state
        output_bits.extend(state)
    return output_bits


# ----------------------------
# Shuffle S-Box using CA output bits
# ----------------------------
def shuffle_sbox_with_ca(original_sbox, seed_bits: str, ca_steps: int = 50, ca_cells: int = 64):
    bits = generate_ca_bits_rule90(seed_bits, steps=ca_steps, cells=ca_cells)
    shuffled = {}
    idx = 0
    for rule, values in original_sbox.items():
        rand_nums = []
        for _ in range(len(values)):
            chunk = bits[idx: idx+4]
            idx += 4
            if len(chunk) < 4:
                # wrap-around if exhausted
                chunk = bits[:4]
            rand_nums.append(int(''.join(map(str, chunk)), 2))
        paired = list(zip(rand_nums, values.copy()))
        paired.sort(key=lambda x: x[0])
        shuffled[rule] = [val for _, val in paired]
    return shuffled


# ----------------------------
# Compression / Decompression
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
            raise ValueError(f"Value {val} not found in S-Box")
        rule, idx = rev[val]
        out_bits.append(format(idx, '04b'))
        rules.append(rule)
    return ''.join(out_bits), rules


def decompress_bits(compressed_bits, rules, sbox):
    out = []
    for i, rule in enumerate(rules):
        start = i * 4
        idx = int(compressed_bits[start:start+4], 2)
        val = sbox[rule][idx]
        out.append(format(val, '06b'))
    return ''.join(out)


# ----------------------------
# Base64 Wrappers
# ----------------------------
def bits_to_b64_json(bits):
    if bits == "":
        return {"b64": "", "bitlen": 0}
    b = int(bits, 2).to_bytes((len(bits) + 7)//8, byteorder='big')
    b64 = base64.b64encode(b).decode()
    return {"b64": b64, "bitlen": len(bits)}


def b64_json_to_bits(b64_json):
    b64 = b64_json["b64"]
    bitlen = b64_json["bitlen"]
    if b64 == "" or bitlen == 0:
        return ""
    raw = base64.b64decode(b64)
    bits_full = ''.join(f"{byte:08b}" for byte in raw)
    return bits_full[-bitlen:]


# ----------------------------
# QR Embedding Check
# ----------------------------
def try_embed_qr(data_str, label="data"):
    try:
        qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H)
        qr.add_data(data_str)
        qr.make(fit=True)
        version = qr.version
        qr.make_image(fill_color="black", back_color="white")
        return True, version
    except Exception:
        return False, None


# ----------------------------
# MAIN PIPELINE
# ----------------------------
if __name__ == "__main__":
    # Configuration (in production, store device_secret securely)
    device_secret = "my_secure_device_key_12345"
    data = "Transaction ID: 12345, Amount: $90" 

    # Generate nonce
    nonce = secrets.token_hex(16)
    print(f"[*] Generated nonce: {nonce}")

    # Generate QR
    qr_array = generate_qr(data, "original_qr.png")
    print("[*] QR generated")

    # Entropy after QR
    qr_flat = qr_array.ravel().tolist()
    print(f"[QR] Size: {len(qr_flat)} bits | Entropy: {shannon_entropy(qr_flat):.4f}")

    # Generate visual cryptography shares
    share1, share2 = create_shares(qr_array)
    Image.fromarray((share1*255).astype(np.uint8)).save("share1.png")
    Image.fromarray((share2*255).astype(np.uint8)).save("share2.png")
    print("[*] Visual cryptography shares saved (share1.png, share2.png)")

    # Flatten share1 and compute its hash (use hash as message input)
    share1_bits = ''.join(str(bit) for bit in share1.ravel().tolist())
    share1_hash = hashlib.sha256(share1_bits.encode()).hexdigest()
    print(f"[*] Share1 hash (used for seed): {share1_hash[:16]}...")

    # Prepare metadata (store share1_hash to allow deterministic decompression)
    metadata = {
        "nonce": nonce,
        "share1_hash": share1_hash,
        "device_secret_hash": hashlib.sha256(device_secret.encode()).hexdigest()[:16]
    }
    with open("metadata.json", "w") as f:
        json.dump(metadata, f)

    # Generate CA seed using share1_hash as the 'message' component
    ca_seed_bits = generate_enhanced_ca_seed(
        device_secret=device_secret,
        nonce=nonce,
        message=share1_hash,  # IMPORTANT: use share1_hash here
        ca_length=256
    )
    print(f"[*] Generated enhanced CA seed ({len(ca_seed_bits)} bits)")

    # Prepare share1 bitstring for compression, pad to multiple of 6
    flat_share1 = share1_bits
    original_length = len(flat_share1)
    while len(flat_share1) % 6 != 0:
        flat_share1 += '0'
    padding_added = len(flat_share1) - original_length
    with open("padding_info.json", "w") as f:
        json.dump({"padding": padding_added, "original_length": original_length}, f)

    # ---- CA S-box Generation with enhanced seed (Rule 90) ----
    shuffled_sbox = shuffle_sbox_with_ca(BASE_SBOX, seed_bits=ca_seed_bits, ca_steps=50, ca_cells=64)

    # Print shuffled S-boxes (for inspection)
    print("\n--- Shuffled S-BOX (by rule) ---")
    for r, vals in shuffled_sbox.items():
        print(f"Rule {r}: {vals}")
    print("--- end S-BOX ---\n")

    # Iterative compression (10 rounds as before)
    compressed_bits = flat_share1
    all_rules = []

    for i in range(10):
        compressed_bits, rules = compress_bits(compressed_bits, shuffled_sbox)
        all_rules.append(rules)

        # Save compressed as base64 JSON and rules
        b64meta = bits_to_b64_json(compressed_bits)
        with open(f"compressed_iter_{i+1}.json", "w") as f:
            json.dump(b64meta, f)
        with open(f"rules_iter_{i+1}.json", "w") as f:
            json.dump(rules, f)

        compressed_chars = len(compressed_bits)
        rules_chars = len(json.dumps(rules))
        b64_chars = len(b64meta["b64"])
        print(f"Iteration {i+1}: Compressed={compressed_chars}, Rules={rules_chars}, Base64={b64_chars}")

        fit_b64, ver_b64 = try_embed_qr(b64meta["b64"], label=f"iter{i+1}_b64")
        print(f"Iteration {i+1}: QR fit (base64) = {fit_b64}, version={ver_b64}")

        # Test decompression locally
        test_decompressed = decompress_bits(compressed_bits, rules, shuffled_sbox)
        with open(f"decompressed_iter_{i+1}.txt", "w") as f:
            f.write(test_decompressed)

    # ----------------------------
    # Full 10-round decompression (using saved metadata)
    # ----------------------------
    print("[*] Starting decompression...")

    metadata = json.load(open("metadata.json"))
    decompression_seed_bits = generate_enhanced_ca_seed(
        device_secret=device_secret,
        nonce=metadata["nonce"],
        message=metadata["share1_hash"],  # use stored share1_hash
        ca_length=256
    )
    decompression_sbox = shuffle_sbox_with_ca(BASE_SBOX, seed_bits=decompression_seed_bits, ca_steps=50, ca_cells=64)

    decompressed_bits = None
    for i in range(9, -1, -1):
        b64meta = json.load(open(f"compressed_iter_{i+1}.json"))
        rules = json.load(open(f"rules_iter_{i+1}.json"))
        compressed_bits_i = b64_json_to_bits(b64meta)
        decompressed_bits = decompress_bits(compressed_bits_i, rules, decompression_sbox)

    # Remove padding and restore original
    pad_info = json.load(open("padding_info.json"))
    final_decompressed = decompressed_bits[:pad_info["original_length"]]

    restored_array = np.array([int(b) for b in final_decompressed]).reshape(share1.shape)
    Image.fromarray((restored_array*255).astype(np.uint8)).save("restored_share1.png")
    with open("decompressed_share1.txt", "w") as f:
        f.write(final_decompressed)

    # Reconstruct QR
    reconstructed_qr_array = np.logical_xor(restored_array, share2).astype(np.uint8)
    Image.fromarray((reconstructed_qr_array*255).astype(np.uint8)).save("reconstructed_qr.png")

    # Validation: compare hashes of original share1 and restored share1
    restored_bits = ''.join(str(bit) for bit in restored_array.ravel().tolist())
    orig_hash = hashlib.sha256(share1_bits.encode()).hexdigest()
    restored_hash = hashlib.sha256(restored_bits.encode()).hexdigest()
    print(f"Original share1 hash:  {orig_hash[:16]}...")
    print(f"Restored share1 hash:  {restored_hash[:16]}...")
    print(f"Hashes match: {orig_hash == restored_hash}")

    print(f"Final decompressed share1: {len(final_decompressed)} bits")
    print("[*] Enhanced Rule 90 pipeline complete.")

