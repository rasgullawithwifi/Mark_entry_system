"""
Script 1: 1_prepare_dataset.py
────────────────────────────────
Reads your answer sheet JPGs + labels.csv
Crops each individual cell from the marks row
Augments each cell 20 times
Saves into labeled class folders ready for training

Run: python3 1_prepare_dataset.py
"""

import cv2
import numpy as np
import os
import csv
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
SHEETS_DIR  = "answer_sheets"
CSV_FILE    = "answer_sheets/labels.csv"
DATASET_DIR = "cell_data/dataset"
CELL_SIZE   = 64
NUM_AUG     = 20

CLASS_NAMES = ["blank", "0", "0.5", "1", "1.5", "2", "2.5", "3"]
COLUMNS     = ['1a','1b','2a','2b','3a','3b','4a','4b','5a','5b','6a','6b']

# ── Create class folders ──────────────────────────────────────────────────────
for cls in CLASS_NAMES:
    os.makedirs(os.path.join(DATASET_DIR, cls), exist_ok=True)
print(f"✅ Created class folders in {DATASET_DIR}/")

# ── Helpers ───────────────────────────────────────────────────────────────────
def to_class(val_str):
    v = val_str.strip()
    if v == "" or v == "-1":
        return "blank"
    try:
        f = float(v)
        rounded = round(f * 2) / 2
        s = str(rounded)
        if s.endswith(".0"):
            s = s[:-2]
        return s
    except:
        return None

def augment(img, n=20):
    results = [img.copy()]
    h, w = img.shape[:2]
    for _ in range(n - 1):
        aug = img.copy()
        angle = np.random.uniform(-20, 20)
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        alpha = np.random.uniform(0.6, 1.4)
        aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=0)
        if np.random.random() > 0.5:
            k = np.random.choice([3, 5])
            aug = cv2.GaussianBlur(aug, (k, k), 0)
        if np.random.random() > 0.5:
            zoom = np.random.uniform(0.85, 1.0)
            ch, cw = int(h*zoom), int(w*zoom)
            y0 = (h - ch) // 2
            x0 = (w - cw) // 2
            aug = cv2.resize(aug[y0:y0+ch, x0:x0+cw], (w, h))
        if np.random.random() > 0.6:
            noise = np.random.randint(-15, 15, aug.shape, dtype=np.int16)
            aug = np.clip(aug.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        results.append(aug)
    return results

def find_4_tables(img):
    """Divide image into 4 equal horizontal strips — one per table."""
    h, w = img.shape[:2]
    strip_h = h // 4
    return [(0, i * strip_h, w, strip_h) for i in range(4)]

def find_marks_row(table_img):
    """Find the row with red handwriting inside a table."""
    h, w = table_img.shape[:2]
    hsv = cv2.cvtColor(table_img, cv2.COLOR_BGR2HSV)
    m1  = cv2.inRange(hsv, np.array([0,   40, 40]), np.array([15,  255, 255]))
    m2  = cv2.inRange(hsv, np.array([160, 40, 40]), np.array([180, 255, 255]))
    red = cv2.bitwise_or(m1, m2)
    row_sums = np.sum(red, axis=1)
    if row_sums.max() == 0:
        y1 = int(h * 0.55)
        y2 = int(h * 0.82)
        return table_img[y1:y2, :], y1
    peak = int(np.argmax(row_sums))
    margin = max(22, int(h * 0.14))
    y1 = max(0, peak - margin)
    y2 = min(h, peak + margin)
    return table_img[y1:y2, :], y1

def crop_12_cells(marks_row_img):
    """Split marks row into 12 cells: 1a 1b 2a 2b ... 6a 6b"""
    h, w = marks_row_img.shape[:2]
    cell_w = w // 12
    cells = []
    for i in range(12):
        x1 = i * cell_w + 3
        x2 = min(x1 + cell_w - 6, w)
        cell = marks_row_img[3:h-3, x1:x2]
        if cell.size > 0:
            cell = cv2.resize(cell, (CELL_SIZE, CELL_SIZE))
        else:
            cell = np.ones((CELL_SIZE, CELL_SIZE, 3), dtype=np.uint8) * 255
        cells.append(cell)
    return cells

# ── Read CSV ──────────────────────────────────────────────────────────────────
print(f"\nReading {CSV_FILE}...")
label_rows = []
with open(CSV_FILE, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        label_rows.append(row)
print(f"✅ {len(label_rows)} labeled tables found\n")

rows_by_file = defaultdict(list)
for row in label_rows:
    rows_by_file[row['filename'].strip()].append(row)

# ── Process ───────────────────────────────────────────────────────────────────
total_saved  = 0
class_counts = {c: 0 for c in CLASS_NAMES}

for filename, tables in sorted(rows_by_file.items()):
    sheet_path = os.path.join(SHEETS_DIR, filename)
    if not os.path.exists(sheet_path):
        print(f"  ❌ Not found: {filename}")
        continue

    img = cv2.imread(sheet_path)
    if img is None:
        print(f"  ❌ Cannot read: {filename}")
        continue

    print(f"  📄 {filename}", end="  ")

    table_regions = find_4_tables(img)
    sheet_num = int(filename.split('_')[1].split('.')[0])

    for row in tables:
        table_no = int(row['table_no'].strip())
        t_idx    = table_no - 1
        if t_idx >= len(table_regions):
            continue

        tx, ty, tw, th = table_regions[t_idx]
        table_img = img[ty:ty+th, tx:tx+tw]
        if table_img.size == 0:
            continue

        marks_row, _ = find_marks_row(table_img)
        cells = crop_12_cells(marks_row)

        for i, col in enumerate(COLUMNS):
            if i >= len(cells):
                break
            val_str    = row.get(col, '').strip()
            class_name = to_class(val_str)
            if class_name is None or class_name not in CLASS_NAMES:
                continue

            cell_img  = cells[i]
            base_name = f"s{sheet_num:03d}_t{table_no}_{col}"
            save_dir  = os.path.join(DATASET_DIR, class_name)
            augmented = augment(cell_img, NUM_AUG)

            for aug_i, aug_img in enumerate(augmented):
                cv2.imwrite(
                    os.path.join(save_dir, f"{base_name}_aug{aug_i:02d}.jpg"),
                    aug_img)
                total_saved += 1

            class_counts[class_name] += len(augmented)

    print("✅")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  DATASET READY")
print(f"{'='*50}")
print(f"  Total images : {total_saved}")
print(f"\n  Per class:")
for cls in CLASS_NAMES:
    count = class_counts[cls]
    bar   = "█" * (count // 50)
    print(f"    {cls:>6}  {count:>5}  {bar}")
print(f"\n  Saved to: {DATASET_DIR}/")
print(f"  ✅ Run next: python3 2_train_model.py"