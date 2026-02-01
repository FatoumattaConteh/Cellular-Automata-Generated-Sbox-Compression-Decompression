import qrcode
import numpy as np
import random
import json
from PIL import Image
import math

# --- Shannon Entropy ---
def shannon_entropy(data):
    """Calculate Shannon entropy for binary data."""
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
# Step 1: Generate QR Code
# ----------------------------
def generate_qr(data, filename="qr_code.png"):
    qr = qrcode.QRCode(version=None, error_correction=qrcode.constants.ERROR_CORRECT_H)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill="black", back_color="white")
    img.save(filename)
    return np.array(img, dtype=np.uint8)

# ----------------------------
# Step 2: Create Visual Cryptography Shares
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
# Step 3: S-Box Table (Compression Mapping)
# ----------------------------
SBOX = {
    "00": [0,3,5,6,24,27,29,30,40,43,45,46,48,51,53,54],  # Rule 1: XOR/XOR
    "01": [9,10,12,15,17,18,20,23,33,34,36,39,57,58,60,63],  # Rule 2: XNOR/XNOR
    "10": [1,2,4,7,25,26,28,31,41,42,44,47,49,50,52,55],  # Rule 3: XOR/XNOR
    "11": [8,11,13,14,16,19,21,22,32,35,37,38,56,59,61,62],  # Rule 4: XNOR/XOR
}

# Reverse lookup for decompression
REVERSE_SBOX = {}
for rule, values in SBOX.items():
    for idx, val in enumerate(values):
        REVERSE_SBOX[val] = (rule, idx)

# ----------------------------
# Step 4: Compression Function
# ----------------------------
def compress(bits):
    compressed = []
    rules_used = []
    for i in range(0, len(bits), 6):
        chunk = bits[i:i+6]
        if len(chunk) < 6:
            continue
        val = int(chunk, 2)
        if val in REVERSE_SBOX:
            rule, idx = REVERSE_SBOX[val]
            compressed.append(format(idx, '04b'))
            rules_used.append(rule)
        else:
            raise ValueError(f"Value {val} not found in SBOX")
    return "".join(compressed), rules_used

# ----------------------------
# Step 5: Decompression Function
# ----------------------------
def decompress(compressed_bits, rules):
    decompressed = []
    for i, rule in enumerate(rules):
        idx = int(compressed_bits[i*4:(i+1)*4], 2)
        val = SBOX[rule][idx]
        decompressed.append(format(val, '06b'))
    return "".join(decompressed)

# ----------------------------
# Step 6: QR Embedding Test
# ----------------------------
def try_embed_qr(data, iteration):
    try:
        qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H)
        qr.add_data(data)
        qr.make(fit=True)
        version = qr.version
        qr_img = qr.make_image(fill_color="black", back_color="white")
        qr_img.save(f"compressed_iter_{iteration}_qr.png")
        print(f"Iteration {iteration}: ✅ Fits into QR (version {version})")
    except Exception:
        print(f"Iteration {iteration}: ❌ Too large for QR")


# ----------------------------
# Step X: Test S-box Bijectivity
# ----------------------------
def test_sbox_bijectivity_verbose(SBOX):
    """
    Proper bijectivity test and verbose display:
    - Prints mapping for values 0..63 if present.
    - Reports duplicates and missing values.
    - Returns True if bijective, False otherwise.
    """
    # Build value -> (rule, idx) map and detect duplicates
    value_map = {}       # v -> (rule, idx)
    duplicates = []      # list of (v, first_loc, second_loc,...)
    locations = {}       # v -> list of locations

    for rule, values in SBOX.items():
        for idx, v in enumerate(values):
            locations.setdefault(v, []).append((rule, idx))
            if v not in value_map:
                value_map[v] = (rule, idx)

    # Print mapping for values that exist in the SBOX
    print("\n--- S-Box value -> (rule, idx) mapping (present values) ---")
    for v in sorted(value_map.keys()):
        rule, idx = value_map[v]
        print(f"Value {v:2d} : {format(v, '06b')} -> (rule={rule}, col={idx:02d})")
    print("-----------------------------------------------------------\n")

    # Find duplicates
    dups = {v: locs for v, locs in locations.items() if len(locs) > 1}
    if dups:
        print("❌ Duplicate values found in SBOX (value : locations):")
        for v, locs in sorted(dups.items()):
            print(f"  {v:2d} : {locs}")
    else:
        print("No duplicate values detected in SBOX.")

    # Find missing values
    present = set(value_map.keys())
    missing = sorted(set(range(64)) - present)
    if missing:
        print("\n❌ Missing values (0..63) not present in SBOX:")
        print(missing)
    else:
        print("\nAll values 0..63 are present in SBOX (no missing values).")

    # Summary verdict
    bijective = (len(dups) == 0) and (len(missing) == 0)
    if bijective:
        print("\n✅ S-Box is bijective (permutation of 0..63).")
    else:
        print("\n❌ S-Box is NOT bijective. Fix duplicates/missing entries.")
    return bijective



# ----------------------------
# Step 7: Pipeline
# ----------------------------
if __name__ == "__main__":
    data = "Transaction ID: 12345, Amount: $90"

    # Generate QR
    qr_array = generate_qr(data, "original_qr.png")


     # --- Entropy after QR ---
    qr_flat = qr_array.ravel().tolist()
    qr_entropy = shannon_entropy(qr_flat)
    print(f"[QR] Size: {len(qr_flat)} bits | Entropy: {qr_entropy:.4f}")

    # Generate shares
    share1, share2 = create_shares(qr_array)
    Image.fromarray((share1*255).astype(np.uint8)).save("share1.png")
    Image.fromarray((share2*255).astype(np.uint8)).save("share2.png")


    # --- Entropy after VC ---
    share1_flat = share1.ravel().tolist()
    vc_entropy = shannon_entropy(share1_flat)
    print(f"[VC Share1] Size: {len(share1_flat)} bits | Entropy: {vc_entropy:.4f}")

    # Flatten share1 to binary string
    flat_share1 = "".join(str(bit) for row in share1 for bit in row)
    original_length = len(flat_share1)

    # Pad to multiple of 6
    while len(flat_share1) % 6 != 0:
        flat_share1 += "0"
    padding_added = len(flat_share1) - original_length
    with open("padding_info.json", "w") as f:
        json.dump({"padding": padding_added, "original_length": original_length}, f)

    # Iterative compression
compressed = flat_share1
all_rules = []

for i in range(10):
    compressed, rules = compress(compressed)
    all_rules.append(rules)

    # Save compressed and rules
    with open(f"compressed_iter_{i+1}.txt", "w") as f:
        f.write(compressed)
    with open(f"rules_iter_{i+1}.json", "w") as f:
        json.dump(rules, f)

    # Print sizes
    print(f"Iteration {i+1}: Compressed = {len(compressed)} chars, Rules = {len(json.dumps(rules))} chars")

    # QR embedding check
    if len(compressed) + len(json.dumps(rules)) < 2953*8:  # rough max for QR40-L
        print(f"Iteration {i+1}: ✅ Fits into QR")
    else:
        print(f"Iteration {i+1}: ❌ Too large for QR")

    # Save decompressed text per iteration (optional)
    test_decompressed = decompress(compressed, rules)
    with open(f"decompressed_iter_{i+1}.txt", "w") as f:
        f.write(test_decompressed)
    print(f"Iteration {i+1}: Decompressed = {len(test_decompressed)} chars\n")

# ----------------------------
# Full 10-round decompression
# ----------------------------
decompressed = compressed
for i in range(9, -1, -1):
    with open(f"compressed_iter_{i+1}.txt") as f:
        compressed_i = f.read().strip()
    with open(f"rules_iter_{i+1}.json") as f:
        rules_i = json.load(f)
    decompressed = decompress(compressed_i, rules_i)

# Remove padding
with open("padding_info.json") as f:
    pad_info = json.load(f)
final_decompressed = decompressed[:pad_info["original_length"]]

# Reshape and save final restored share
restored_array = np.array([int(b) for b in final_decompressed]).reshape(share1.shape)
Image.fromarray((restored_array*255).astype(np.uint8)).save("restored_share1.png")
with open("decompressed_share1.txt", "w") as f:
    f.write(final_decompressed)

# ----------------------------
# Step 9: Reconstruct Original QR from Shares
# ----------------------------
# share1_restored: final decompressed share1
share1_restored = restored_array  # from previous final decompression
share2_original = share2          # original share2

# Stack shares: simple OR operation (black=1, white=0)
reconstructed_qr_array = np.logical_xor(share1_restored, share2_original).astype(np.uint8)

# Save reconstructed QR image
Image.fromarray((reconstructed_qr_array*255).astype(np.uint8)).save("reconstructed_qr.png")

# Display number of characters in decompressed share1
print(f"Final decompressed share1: {len(final_decompressed)} characters")

 # Test S-box first
test_sbox_bijectivity_verbose(SBOX)

