"""
QR -> VC -> CA30-shuffled S-box -> COMPLETE SAC + BIC Testing
Tests both Strict Avalanche Criterion AND Bit Independence Criterion
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

# ---------- Theoretical References ----------
THEORETICAL_IDEAL = 50.0

def complete_crypto_analysis(grouped_sbox):
    """
    Complete cryptographic analysis including both SAC and BIC tests
    """
    flat_sbox = grouped_sbox_to_flat(grouped_sbox)
    n_inputs = 64
    n_inbits = 6
    n_outbits = 4
    
    # SAC Analysis
    sac_matrix = np.zeros((n_inbits, n_outbits))
    avalanche_changes = []
    
    # BIC Analysis - we need to track changes per input bit flip
    # Structure: bic_data[input_bit] = list of output change vectors
    bic_data = [[] for _ in range(n_inbits)]
    
    for x in range(n_inputs):
        original_out = flat_sbox[x]
        
        for input_bit in range(n_inbits):
            flipped_input = x ^ (1 << input_bit)
            flipped_out = flat_sbox[flipped_input]
            
            # Calculate changed bits
            changed_bits = bin(original_out ^ flipped_out).count('1')
            avalanche_changes.append(changed_bits)
            
            # Track output bit changes for SAC
            change_vector = []  # For BIC: track which specific bits changed
            for output_bit in range(n_outbits):
                original_bit = (original_out >> output_bit) & 1
                flipped_bit = (flipped_out >> output_bit) & 1
                if original_bit != flipped_bit:
                    sac_matrix[input_bit, output_bit] += 1
                    change_vector.append(1)  # This bit flipped
                else:
                    change_vector.append(0)  # This bit didn't flip
            
            # Store for BIC analysis
            bic_data[input_bit].append(change_vector)
    
    # Convert SAC to probabilities and calculate metrics
    sac_matrix /= n_inputs
    sac_deviations = np.abs(sac_matrix - 0.5)
    avg_sac_deviation = np.mean(sac_deviations)
    sac_compliance = 1.0 - avg_sac_deviation
    
    # Avalanche effect
    avg_bits_changed = np.mean(avalanche_changes)
    avalanche_percent = (avg_bits_changed / n_outbits) * 100
    
    # BIC Analysis - Check independence between output bits
    bic_results = analyze_bic(bic_data, n_inbits, n_outbits)
    
    return {
        'avalanche_percent': avalanche_percent,
        'avg_bits_changed': avg_bits_changed,
        'sac_compliance': sac_compliance,
        'avg_sac_deviation': avg_sac_deviation,
        'sac_matrix': sac_matrix.tolist(),
        'bic_analysis': bic_results
    }

def analyze_bic(bic_data, n_inbits, n_outbits):
    """
    Analyze Bit Independence Criterion
    For each input bit flip, check if output bit changes are independent
    """
    bic_correlations = []
    bic_max_correlation = 0
    bic_violations = 0
    
    for input_bit in range(n_inbits):
        changes = np.array(bic_data[input_bit])  # 64x4 matrix of changes
        
        # For each pair of output bits, check correlation
        for bit_i in range(n_outbits):
            for bit_j in range(bit_i + 1, n_outbits):
                # Get the change patterns for these two bits
                changes_i = changes[:, bit_i]  # How bit_i changes across all inputs
                changes_j = changes[:, bit_j]  # How bit_j changes across all inputs
                
                # Calculate correlation coefficient
                correlation = np.corrcoef(changes_i, changes_j)[0, 1]
                if np.isnan(correlation):
                    correlation = 0  # Handle undefined correlation
                
                bic_correlations.append(abs(correlation))
                
                # Track maximum correlation and violations
                if abs(correlation) > bic_max_correlation:
                    bic_max_correlation = abs(correlation)
                
                # Consider correlation > 0.3 as a violation (adjustable threshold)
                if abs(correlation) > 0.3:
                    bic_violations += 1
    
    avg_bic_correlation = np.mean(bic_correlations) if bic_correlations else 0
    
    # BIC compliance: 1.0 = perfectly independent, 0.0 = completely correlated
    bic_compliance = 1.0 - avg_bic_correlation
    
    return {
        'bic_compliance': bic_compliance,
        'avg_correlation': avg_bic_correlation,
        'max_correlation': bic_max_correlation,
        'violation_count': bic_violations,
        'total_bit_pairs': len(bic_correlations),
        'assessment': evaluate_bic_strength(bic_compliance, bic_max_correlation)
    }

def evaluate_bic_strength(bic_compliance, max_correlation):
    """Evaluate BIC strength based on correlation metrics"""
    if bic_compliance >= 0.95 and max_correlation <= 0.2:
        return "EXCELLENT_BIC"
    elif bic_compliance >= 0.90 and max_correlation <= 0.3:
        return "GOOD_BIC" 
    elif bic_compliance >= 0.85 and max_correlation <= 0.4:
        return "ACCEPTABLE_BIC"
    else:
        return "WEAK_BIC"

def evaluate_overall_strength(avalanche_percent, sac_compliance, bic_compliance):
    """Overall evaluation considering all three criteria"""
    avalanche_score = min(avalanche_percent / 50.0, 1.0)  # Normalize to 0-1
    overall_score = (avalanche_score + sac_compliance + bic_compliance) / 3.0
    
    if overall_score >= 0.90:
        return "CRYPTOGRAPHICALLY_STRONG", overall_score
    elif overall_score >= 0.80:
        return "GOOD", overall_score
    elif overall_score >= 0.70:
        return "ACCEPTABLE", overall_score
    else:
        return "NEEDS_IMPROVEMENT", overall_score

# ---------- Rest of the code (helpers, CA, etc. remain the same) ----------
def grouped_sbox_to_flat(grouped_sbox):
    flat = [None]*64
    for rule, vals in grouped_sbox.items():
        for idx, val in enumerate(vals):
            flat[val] = idx
    missing = [i for i,v in enumerate(flat) if v is None]
    if missing:
        raise ValueError("Missing inputs in grouped S-box: " + str(missing))
    return flat

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
            return s
        except Exception:
            pass
    try:
        store_secret_in_file(s, LOCAL_SECRET_PATH)
    except Exception as e:
        print("[!] Could not store device secret to file:", e)
    return s

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

BASE_SBOX = {
    "00": [0,3,5,6,24,27,29,30,40,43,45,46,48,51,53,54],
    "01": [9,10,12,15,17,18,20,23,33,34,36,39,57,58,60,63],
    "10": [1,2,4,7,25,26,28,31,41,42,44,47,49,50,52,55],
    "11": [8,11,13,14,16,19,21,22,32,35,37,38,56,59,61,62],
}

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

def main():
    MAX_TRIES = 50  # Reduced for demonstration
    KEY_BITS = 8
    CA_STEPS = 120
    CA_CELLS = 256
    report_every = 5

    print("=== COMPLETE CRYPTOGRAPHIC ANALYSIS ===")
    print("Testing: Avalanche Effect + SAC + BIC")
    print("=" * 50)

    # Test base S-box first
    print("\n[*] Analyzing base S-box...")
    base_analysis = complete_crypto_analysis(BASE_SBOX)
    base_overall, base_score = evaluate_overall_strength(
        base_analysis['avalanche_percent'],
        base_analysis['sac_compliance'], 
        base_analysis['bic_analysis']['bic_compliance']
    )
    
    print(f"Base S-box Analysis:")
    print(f"  Avalanche: {base_analysis['avalanche_percent']:.2f}%")
    print(f"  SAC:       {base_analysis['sac_compliance']:.3f}")
    print(f"  BIC:       {base_analysis['bic_analysis']['bic_compliance']:.3f}")
    print(f"  Overall:   {base_overall} (score: {base_score:.3f})")

    device_secret = get_or_create_device_secret()
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
    best_overall_score = 0
    results = []

    print(f"\n[*] Testing {MAX_TRIES} CA-shuffled S-boxes...")
    
    for attempt in range(1, MAX_TRIES + 1):
        nonce = secrets.token_hex(12)
        seed_input = share_hash + nonce + device_secret_hex
        seed_hash = hashlib.sha256(seed_input.encode()).hexdigest()
        candidate = shuffle_sbox_with_ca(BASE_SBOX, seed_hash,
                                         key_bits_per_value=KEY_BITS,
                                         steps=CA_STEPS, cells=CA_CELLS)

        analysis = complete_crypto_analysis(candidate)
        overall_assessment, overall_score = evaluate_overall_strength(
            analysis['avalanche_percent'],
            analysis['sac_compliance'],
            analysis['bic_analysis']['bic_compliance']
        )
        
        results.append({
            "attempt": attempt,
            "nonce": nonce,
            "avalanche": analysis['avalanche_percent'],
            "sac": analysis['sac_compliance'],
            "bic": analysis['bic_analysis']['bic_compliance'],
            "overall_score": overall_score,
            "assessment": overall_assessment,
            "bic_violations": analysis['bic_analysis']['violation_count'],
            "grouped_sbox": candidate
        })

        if overall_score > best_overall_score:
            best_candidate = candidate
            best_overall_score = overall_score
            best_analysis = analysis
            best_meta = {
                "attempt": attempt,
                "nonce": nonce,
                "overall_score": overall_score,
                "assessment": overall_assessment
            }

        if attempt % report_every == 0 or attempt == 1:
            bic_status = analysis['bic_analysis']['assessment']
            print(f" attempt {attempt:2d}: "
                  f"Av={analysis['avalanche_percent']:5.2f}% "
                  f"SAC={analysis['sac_compliance']:.3f} "
                  f"BIC={analysis['bic_analysis']['bic_compliance']:.3f} "
                  f"[{overall_assessment:>10}]")

    # Final comprehensive results
    print("\n" + "="*70)
    print("COMPREHENSIVE CRYPTOGRAPHIC ANALYSIS RESULTS")
    print("="*70)
    
    print(f"\nBase S-box performance:")
    print(f"  Overall score: {base_score:.3f} - {base_overall}")
    
    print(f"\nBest CA-shuffled S-box (attempt {best_meta['attempt']}):")
    print(f"  Avalanche effect: {best_analysis['avalanche_percent']:.2f}%")
    print(f"  SAC compliance:   {best_analysis['sac_compliance']:.3f}")
    print(f"  BIC compliance:   {best_analysis['bic_analysis']['bic_compliance']:.3f}")
    print(f"  BIC assessment:   {best_analysis['bic_analysis']['assessment']}")
    print(f"  BIC violations:   {best_analysis['bic_analysis']['violation_count']} "
          f"/ {best_analysis['bic_analysis']['total_bit_pairs']} pairs")
    print(f"  Overall score:    {best_overall_score:.3f} - {best_meta['assessment']}")
    
    # Improvement analysis
    avalanche_impr = best_analysis['avalanche_percent'] - base_analysis['avalanche_percent']
    sac_impr = best_analysis['sac_compliance'] - base_analysis['sac_compliance']
    bic_impr = best_analysis['bic_analysis']['bic_compliance'] - base_analysis['bic_analysis']['bic_compliance']
    
    print(f"\nImprovement over base S-box:")
    print(f"  Avalanche: +{avalanche_impr:+.2f}%")
    print(f"  SAC:       +{sac_impr:+.3f}")
    print(f"  BIC:       +{bic_impr:+.3f}")

    # Save comprehensive results
    summary = {
        "cryptographic_analysis": {
            "base_sbox": base_analysis,
            "best_candidate": {
                "metadata": best_meta,
                "analysis": best_analysis
            },
            "improvement": {
                "avalanche": avalanche_impr,
                "sac": sac_impr, 
                "bic": bic_impr
            }
        },
        "all_results": [{k: v for k, v in r.items() if k != "grouped_sbox"} 
                        for r in results]
    }

    with open("comprehensive_crypto_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    with open("best_sbox_comprehensive.json", "w") as f:
        json.dump({
            "metadata": best_meta,
            "sbox": best_candidate,
            "detailed_analysis": best_analysis
        }, f, indent=2)

    print(f"\n[*] Comprehensive analysis saved")
    print(f"[*] Best S-box meets: {best_meta['assessment']}")

if __name__ == "__main__":
    main()