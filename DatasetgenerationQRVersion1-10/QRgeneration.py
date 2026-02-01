# generate_qr_dataset.py
import os, random, csv
import qrcode
from PIL import Image
import numpy as np

OUTDIR = "qr_images"
os.makedirs(OUTDIR, exist_ok=True)
random_dir = os.path.join(OUTDIR, "random")
pay_dir = os.path.join(OUTDIR, "payments")
os.makedirs(random_dir, exist_ok=True)
os.makedirs(pay_dir, exist_ok=True)

def shannon_entropy(arr):
    a = np.asarray(arr).ravel()
    vals, counts = np.unique(a, return_counts=True)
    probs = counts / counts.sum()
    return - (probs * np.log2(probs)).sum()

rows = []
# 1) Random binary payloads (100 each version)
for version in range(5, 11):
    ver_dir = os.path.join(random_dir, f"v{version}")
    os.makedirs(ver_dir, exist_ok=True)
    for i in range(1, 501):
        payload = "R_" + str(version) + "_" + "".join(random.choice("01") for _ in range(version * 20))
        qr = qrcode.QRCode(version=version, error_correction=qrcode.constants.ERROR_CORRECT_L)
        qr.add_data(payload)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        fname = f"qr_v{version}_rand_{i}.png"
        path = os.path.join(ver_dir, fname)
        img.save(path)
        arr = np.array(img.convert("1"))
        rows.append([path, version, "random", len(payload), shannon_entropy(arr)])

# 2) Payment-like payloads (small realistic strings)
payments = []
for i in range(1, 101):  # 100 payment samples total — distribute across versions
    tid = f"TX{100000+i}"
    amt = f"{random.randint(1,9999)}.{random.randint(0,99):02d}"
    ts = f"2025-09-{random.randint(1,28):02d}T{random.randint(0,23):02d}:{random.randint(0,59):02d}"
    payload = f"ID:{tid};AMT:{amt};TS:{ts};MERCH:Shop{random.randint(1,200)}"
    payments.append(payload)

# distribute payments across versions 1..10 (10 each)
for idx, payload in enumerate(payments, start=1):
    version = ((idx-1) % 10) + 1
    ver_dir = os.path.join(pay_dir, f"v{version}")
    os.makedirs(ver_dir, exist_ok=True)
    qr = qrcode.QRCode(version=version, error_correction=qrcode.constants.ERROR_CORRECT_L)
    qr.add_data(payload)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    fname = f"qr_v{version}_pay_{idx}.png"
    path = os.path.join(ver_dir, fname)
    img.save(path)
    arr = np.array(img.convert("1"))
    rows.append([path, version, "payment", len(payload), shannon_entropy(arr)])

# write metadata CSV
csv_path = os.path.join(OUTDIR, "dataset_metadata.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["filepath", "qr_version", "payload_type", "payload_len_chars", "qr_entropy"])
    w.writerows(rows)

print("Dataset done:", csv_path)
