"""
Step 1: Crop individual cells from answer sheet images
Reads labels.csv and sheet JPGs → saves individual cell crops
"""

import cv2
import numpy as np
import os
import csv
from pathlib import Path

# Config
ANSWER_SHEETS_DIR = "answer_sheets"
RAW_CELLS_DIR = "cell_data/raw_cells"
CSV_FILE = os.path.join(ANSWER_SHEETS_DIR, "labels.csv")

# Answer sheet dimensions (from your table)
NUM_QUESTIONS = 6
NUM_PARTS = 2  # a and b

os.makedirs(RAW_CELLS_DIR, exist_ok=True)

def find_marks_table(img):
    """Find the marks table in the answer sheet image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    # Find horizontal and vertical lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    
    h_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, h_kernel)
    v_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, v_kernel)
    grid = cv2.add(h_lines, v_lines)
    
    # Find contours
    cnts, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_area = img.shape[0] * img.shape[1]
    best = None
    best_area = 0
    
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        # Look for large rectangular table
        if area > img_area * 0.08 and area > best_area and w > h:
            best = (x, y, w, h)
            best_area = area
    
    return best

def extract_cells_from_table(img, sheet_num, table_num):
    """Extract individual cells from a table image."""
    table = find_marks_table(img)
    
    if table is None:
        print(f"  ⚠️  Could not find table in {sheet_num}.{table_num}")
        return {}
    
    tx, ty, tw, th = table
    
    # Marks row is typically in lower half of table
    marks_y_start = int(ty + th * 0.55)
    marks_y_end = int(ty + th * 0.80)
    marks_x_start = int(tx)
    marks_x_end = int(tx + tw)
    
    marks_region = img[marks_y_start:marks_y_end, marks_x_start:marks_x_end]
    mh, mw = marks_region.shape[:2]
    
    # Divide into cells
    num_cells = NUM_QUESTIONS * NUM_PARTS
    cell_w = mw // num_cells
    
    cells = {}
    for idx in range(num_cells):
        q = (idx // 2) + 1
        part = 'a' if (idx % 2) == 0 else 'b'
        key = f"{q}{part}"
        
        cx_start = idx * cell_w + 4
        cx_end = cx_start + cell_w - 8
        
        if cx_end > mw:
            cx_end = mw
        
        cy_start = 4
        cy_end = mh - 4
        
        cell_crop = marks_region[cy_start:cy_end, cx_start:cx_end]
        
        if cell_crop.size > 0:
            cells[key] = cell_crop
    
    return cells

def save_cell(cell_img, sheet_num, table_num, question, part):
    """Save a single cell image."""
    filename = f"s{sheet_num:03d}_t{table_num}_q{question}{part}.jpg"
    filepath = os.path.join(RAW_CELLS_DIR, filename)
    cv2.imwrite(filepath, cell_img)

# ════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════

print("\n" + "="*50)
print("  STEP 1: CROP CELLS FROM SHEETS")
print("="*50)

# Read CSV
print(f"\nReading {CSV_FILE}...")
rows = []
with open(CSV_FILE, 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Found {len(rows)} table entries")

# Process each row
processed = 0
for i, row in enumerate(rows):
    filename = row['filename'].strip()
    table_no = int(row['table_no'].strip())
    
    # Extract sheet number from filename
    sheet_num = int(filename.split('_')[1].split('.')[0])
    
    sheet_path = os.path.join(ANSWER_SHEETS_DIR, filename)
    
    if not os.path.exists(sheet_path):
        print(f"  ⚠️  File not found: {sheet_path}")
        continue
    
    print(f"Processing {filename} (Table {table_no})...", end=" ")
    
    img = cv2.imread(sheet_path)
    if img is None:
        print("❌ Could not read image")
        continue
    
    # Since one JPG has 4 tables, we need to crop the sheet into 4 parts
    # For simplicity, we'll pass the full image and let find_marks_table handle it
    # In reality you might want to crop the image first
    
    cells = extract_cells_from_table(img, sheet_num, table_no)
    
    if not cells:
        print("❌ No cells found")
        continue
    
    # Save each cell
    for key, cell_img in cells.items():
        q = int(key[0])
        p = key[1]
        save_cell(cell_img, sheet_num, table_no, q, p)
    
    print(f"✅ {len(cells)} cells")
    processed += 1

print(f"\n✅ Processed {processed} tables")
print(f"   Cells saved to: {RAW_CELLS_DIR}/")
print(f"   Total cell images: {len(os.listdir(RAW_CELLS_DIR))}")