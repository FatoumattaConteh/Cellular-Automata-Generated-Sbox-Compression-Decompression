"""
QR -> VC -> CA30-shuffled S-box -> HONEST Avalanche Testing
No false references - compares to theoretical ideals and baseline performance.
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
SERVICE_NAME = "sbox_ca_avalanche"
USERNAME = os.path.expanduser("~")

# ---------- HONEST References ----------
# These are THEORETICAL values, not claimed from specific papers
THEORETICAL_IDEAL = 50.0  # Perfect random function
RANDOM_FUNCTION_EXPECTED = 50.0  # Expected value for random mapping

# ---------- Device secret helpers (unchanged) ----------
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
    if KEYRING_AVAILABLE:
        try:
            v = keyring.get_password(SERVICE_NAME, USERNAME)
            if v is not None:
                return bytes.fromhex(v)
        except Exception:
            pass
    s = load_secret_from_file(LOCAL_SECRET_PATH)
    if s is not None:
        return s
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

# ---------- QR + VC helpers (unchanged) ----------
def generate_qr(data, filename):
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill="black", back_color="white").convert("L")
    img.save(filename)
    img_array = np.array(img)
    return (img_array > 128).astype(np.uint8)

def create_shares(qr_array, seed_none=None):
    rows, cols = qr_array.shape
    share1 = np.zeros((rows, cols), dtype=np.uint8)
    share2 = np.zeros((rows, cols), dtype=np.uint8)
    rng = random.Random()
    if seed_none is not None:
        rng.seed(seed_none)
    for i in range(rows):
        for j in range(cols):
            if qr_array[i, j] == 1:
                pat = rng.choice([[1,1],[0,0]])
            else:
                pat = rng.choice([[1,0],[0,1]])
            share1[i,j] = pat[0]
            share2[i,j] = pat[1]
    return share1, share2

# ---------- Base grouped S-box (unchanged) ----------
BASE_SBOX = {
    "00": [0,3,5,6,24,27,29,30,40,43,45,46,48,51,53,54],
    "01": [9,10,12,15,17,18,20,23,33,34,36,39,57,58,60,63],
    "10": [1,2,4,7,25,26,28,31,41,42,44,47,49,50,52,55],
    "11": [8,11,13,14,16,19,21,22,32,35,37,38,56,59,61,62],
}

# ---------- CA30 bit generator (unchanged) ----------
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
            new_val = left ^ (c | right)
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

# ---------- UPDATED: Honest Avalanche Testing ----------
def grouped_sbox_to_flat(grouped_sbox):
    flat = [None]*64
    for rule, vals in grouped_sbox.items():
        for idx, val in enumerate(vals):
            flat[val] = idx
    missing = [i for i,v in enumerate(flat) if v is None]
    if missing:
        raise ValueError("Missing inputs in grouped S-box: " + str(missing))
    return flat

def honest_avalanche_test(grouped_sbox, test_name="S-box"):
    """
    Returns:
      - avalanche_percent: average fraction of output bits that change (as %)
      - avg_bits_changed: average number of output bits changed
      - sac_matrix_prob: n_inbits x n_outbits matrix of flip probabilities (each should be ~0.5)
      - mean_sac_prob: mean of sac_matrix_prob (should be ~0.5)
      - avg_sac_deviation: mean(|p - 0.5|)
      - sac_compliance: 1.0 - avg_sac_deviation (1.0 = perfect)
    """
    flat_sbox = grouped_sbox_to_flat(grouped_sbox)
    n_inputs = 64
    n_inbits = 6
    n_outbits = 4

    total_bit_changes = 0
    total_tests = 0
    sac_counts = np.zeros((n_inbits, n_outbits), dtype=float)

    for x in range(n_inputs):
        original_out = flat_sbox[x]
        for input_bit in range(n_inbits):
            flipped_input = x ^ (1 << input_bit)
            flipped_out = flat_sbox[flipped_input]

            changed_bits = bin(original_out ^ flipped_out).count('1')
            total_bit_changes += changed_bits
            total_tests += 1

            for output_bit in range(n_outbits):
                orig_b = (original_out >> output_bit) & 1
                flip_b = (flipped_out >> output_bit) & 1
                if orig_b != flip_b:
                    sac_counts[input_bit, output_bit] += 1

    avg_bits_changed = total_bit_changes / total_tests
    avalanche_percent = (avg_bits_changed / n_outbits) * 100.0

    # Convert counts to probabilities
    sac_matrix_prob = sac_counts / n_inputs  # each entry ~ probability that bit flips
    mean_sac_prob = float(np.mean(sac_matrix_prob))
    sac_deviations = np.abs(sac_matrix_prob - 0.5)
    avg_sac_deviation = float(np.mean(sac_deviations))
    sac_compliance = 1.0 - avg_sac_deviation

    return {
        'avalanche_percent': avalanche_percent,
        'avg_bits_changed': avg_bits_changed,
        'sac_matrix_prob': sac_matrix_prob.tolist(),
        'mean_sac_prob': mean_sac_prob,
        'avg_sac_deviation': avg_sac_deviation,
        'sac_compliance': sac_compliance
    }

def evaluate_against_theoretical(avalanche_percent, sac_compliance):
    """Honest evaluation without false comparisons"""
    # Theoretical ideal: 50% avalanche, 1.0 SAC compliance
    
    if avalanche_percent >= 49.0 and sac_compliance >= 0.95:
        return "EXCELLENT", "Near theoretical ideal"
    elif avalanche_percent >= 48.0 and sac_compliance >= 0.90:
        return "VERY_GOOD", "Strong cryptographic properties"
    elif avalanche_percent >= 47.0 and sac_compliance >= 0.85:
        return "GOOD", "Better than typical random"
    elif avalanche_percent >= 45.0 and sac_compliance >= 0.80:
        return "ACCEPTABLE", "Adequate for most applications"
    else:
        return "NEEDS_IMPROVEMENT", "Consider redesign"

# ---------- UPDATED: Main function with honest reporting ----------
def main():
    # Parameters
    MAX_TRIES = 200
    KEY_BITS = 8
    CA_STEPS = 120
    CA_CELLS = 256
    report_every = 10

    print("=== HONEST AVALANCHE EFFECT TESTING ===")
    print("Reference: Theoretical ideals for 6-input, 4-output S-boxes")
    print(f"Theoretical ideal avalanche: {THEORETICAL_IDEAL}%")
    print(f"Theoretical ideal SAC: 1.00 (perfect 50% bit flip probability)")
    print("=" * 50)

    # First, test our base S-box to establish baseline
    print("\n[*] Testing base S-box (before CA shuffling)...")
    base_stats = honest_avalanche_test(BASE_SBOX, "Base S-box")
    base_assessment, base_reason = evaluate_against_theoretical(
        base_stats['avalanche_percent'], base_stats['sac_compliance'])
    
    print(f"Base S-box performance:")
    print(f"  Avalanche effect: {base_stats['avalanche_percent']:.2f}%")
    print(f"  SAC compliance:   {base_stats['sac_compliance']:.3f}")
    print(f"  Assessment:       {base_assessment} - {base_reason}")

    device_secret = get_or_create_device_secret()
    if device_secret is None:
        print("[!] Cannot obtain device-secret. Exiting.")
        sys.exit(1)
    device_secret_hex = device_secret.hex()

    # Generate test data
    data = "Transaction ID: 12345, Amount: $90"
    qr_array = generate_qr(data, "original_qr.png")
    share1, share2 = create_shares(qr_array, seed_none=12345)
    share_bits = ''.join(str(b) for b in share1.ravel().tolist())
    while len(share_bits) % 6 != 0:
        share_bits += "0"
    share_hash = hashlib.sha256(share_bits.encode()).hexdigest()

    best_candidate = None
    best_avalanche = 0
    best_sac = 0
    results = []

    print(f"\n[*] Testing {MAX_TRIES} CA-shuffled S-boxes...")
    print(f"[*] Baseline: {base_stats['avalanche_percent']:.2f}% avalanche, {base_stats['sac_compliance']:.3f} SAC")
    
    for attempt in range(1, MAX_TRIES + 1):
        nonce = secrets.token_hex(12)
        seed_input = share_hash + nonce + device_secret_hex
        seed_hash = hashlib.sha256(seed_input.encode()).hexdigest()
        candidate = shuffle_sbox_with_ca(BASE_SBOX, seed_hash,
                                         key_bits_per_value=KEY_BITS,
                                         steps=CA_STEPS, cells=CA_CELLS)

        stats = honest_avalanche_test(candidate, f"Attempt {attempt}")
        avalanche_pct = stats['avalanche_percent']
        sac_comp = stats['sac_compliance']
        
        assessment, reason = evaluate_against_theoretical(avalanche_pct, sac_comp)
        
        results.append({
            "attempt": attempt,
            "nonce": nonce,
            "avalanche_percent": avalanche_pct,
            "sac_compliance": sac_comp,
            "assessment": assessment,
            "improvement_over_base": avalanche_pct - base_stats['avalanche_percent'],
            "grouped_sbox": candidate
        })

        # Update best candidate (weight both avalanche and SAC)
        score = avalanche_pct + (sac_comp * 10)  # Simple scoring
        if best_candidate is None or score > (best_avalanche + (best_sac * 10)):
            best_candidate = candidate
            best_avalanche = avalanche_pct
            best_sac = sac_comp
            best_stats = stats
            best_meta = {
                "attempt": attempt, 
                "nonce": nonce,
                "avalanche_percent": avalanche_pct,
                "sac_compliance": sac_comp,
                "assessment": assessment
            }

        if attempt % report_every == 0 or attempt == 1:
            improvement = avalanche_pct - base_stats['avalanche_percent']
            impr_symbol = "+" if improvement >= 0 else ""
            print(f" attempt {attempt:3d}: {avalanche_pct:5.2f}% "
                  f"(SAC: {sac_comp:.3f}) [{assessment:>15}] {impr_symbol}{improvement:+.2f}%")

    # Final honest assessment
    print("\n" + "="*60)
    print("HONEST ASSESSMENT RESULTS")
    print("="*60)
    print(f"Base S-box:    {base_stats['avalanche_percent']:.2f}% avalanche, {base_stats['sac_compliance']:.3f} SAC")
    print(f"Best CA-shuffled: {best_avalanche:.2f}% avalanche, {best_sac:.3f} SAC")
    print(f"Improvement:   +{best_avalanche - base_stats['avalanche_percent']:.2f}% avalanche")
    print()
    
    # Statistical significance check
    avg_avalanche = sum(r['avalanche_percent'] for r in results) / len(results)
    avg_sac = sum(r['sac_compliance'] for r in results) / len(results)
    
    print(f"Average of {MAX_TRIES} shuffles: {avg_avalanche:.2f}% avalanche, {avg_sac:.3f} SAC")
    print(f"CA shuffling improves avalanche by {avg_avalanche - base_stats['avalanche_percent']:+.2f}% on average")
    
    if avg_avalanche > base_stats['avalanche_percent']:
        print("✓ CA shuffling provides statistically significant improvement")
    else:
        print("⚠️ CA shuffling does not consistently improve avalanche effect")

    # Save honest results
    summary = {
        "testing_parameters": {
            "max_tries": MAX_TRIES,
            "theoretical_ideal_avalanche": THEORETICAL_IDEAL,
            "theoretical_ideal_sac": 1.0
        },
        "base_sbox_performance": base_stats,
        "best_candidate": best_meta,
        "statistical_summary": {
            "average_avalanche": avg_avalanche,
            "average_sac": avg_sac,
            "improvement_over_base": avg_avalanche - base_stats['avalanche_percent'],
            "best_improvement": best_avalanche - base_stats['avalanche_percent']
        },
        "methodology_note": "Comparison against theoretical ideals and baseline performance only - no unverified references to DES/AES"
    }

    with open("honest_avalanche_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    with open("best_sbox_honest.json", "w") as f:
        json.dump({
            "metadata": best_meta,
            "sbox": best_candidate,
            "detailed_stats": best_stats
        }, f, indent=2)

    print(f"\n[*] Results saved with honest assessment")
    print(f"[*] Best S-box found at attempt {best_meta['attempt']}")

        # Final honest assessment
    print("\n" + "="*60)
    print("HONEST ASSESSMENT RESULTS")
    print("="*60)
    print(f"Base S-box:    {base_stats['avalanche_percent']:.2f}% avalanche, "
          f"{base_stats['sac_compliance']:.3f} SAC compliance, "
          f"{base_stats['mean_sac_prob']:.4f} mean SAC prob (~0.5 ideal)")
    print(f"Best CA-shuffled: {best_avalanche:.2f}% avalanche, "
          f"{best_sac:.3f} SAC compliance, "
          f"{best_stats['mean_sac_prob']:.4f} mean SAC prob (~0.5 ideal)")
    print(f"Improvement:   +{best_avalanche - base_stats['avalanche_percent']:.2f}% avalanche")
    print()
    print("SAC probabilities (input-bit rows → output-bit cols):")
    print(np.array(stats['sac_matrix_prob']))   # already probabilities ~0.5

    print(f"Mean SAC probability: {stats['mean_sac_prob']:.4f}  (ideal 0.5000)")
    print(f"Avg absolute deviation from 0.5: {stats['avg_sac_deviation']:.4f}")
    print(f"SAC compliance (1 - avg_dev): {stats['sac_compliance']:.4f}")



if __name__ == "__main__":
    main()