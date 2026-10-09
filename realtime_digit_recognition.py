import tensorflow as tf
import numpy as np
import cv2
import os
from datetime import datetime

model = tf.keras.models.load_model("digit_model.h5")
print("✅ Model loaded")

# Try all cameras and pick the best one available
cap = None
for i in range(3):
    test = cv2.VideoCapture(i)
    if test.isOpened():
        ret, frame = test.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            print(f"Camera {i}: {w}x{h}")
            if cap is None or w > best_w:
                if cap is not None:
                    cap.release()
                cap = test
                best_w = w
                best_i = i
            else:
                test.release()
        else:
            test.release()

if cap is None:
    print("❌ Cannot open any camera"); exit()

print(f"✅ Using camera {best_i}")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

RESULTS_FOLDER = "digit_results"
os.makedirs(RESULTS_FOLDER, exist_ok=True)
capture_count = 0

print("\n==============================================")
print("  S = capture & recognise  |  Q = quit")
print("==============================================\n")

def find_paper(frame):
    h_f, w_f = frame.shape[:2]
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    _, th   = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY)
    kernel  = np.ones((20, 20), np.uint8)
    closed  = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    min_area = w_f * h_f * 0.15
    best, best_area = None, 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if 0.4 < w/h < 2.5 and area > best_area:
            best_area = area
            best = (x, y, w, h)
    if best is None:
        return None, None
    x, y, w, h = best
    pad = 50
    x1, y1 = max(0, x+pad), max(0, y+pad)
    x2, y2 = min(w_f, x+w-pad), min(h_f, y+h-pad)
    return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)

def get_ink_mask(paper):
    hsv  = cv2.cvtColor(paper, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)

    # Red ink
    m1  = cv2.inRange(hsv, np.array([0,   50, 50]), np.array([10,  255, 255]))
    m2  = cv2.inRange(hsv, np.array([160, 50, 50]), np.array([180, 255, 255]))
    red = cv2.bitwise_or(m1, m2)

    # Blue ink — wider range to catch dark blue/navy
    blue = cv2.inRange(hsv, np.array([90, 40, 30]), np.array([140, 255, 255]))

    # Black/pencil
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    black   = cv2.adaptiveThreshold(blurred, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 25, 18)
    black   = cv2.morphologyEx(black, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))

    red_px  = cv2.countNonZero(red)
    blue_px = cv2.countNonZero(blue)

    print(f"red={red_px} blue={blue_px}")  # debug line

    if blue_px > red_px and blue_px > 100:
        mask, ink = blue, "Blue"
    elif red_px > 100:
        mask, ink = red, "Red"
    else:
        mask, ink = black, "Black/Pencil"

    k    = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.dilate(mask, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask, ink

def prepare_digit(mask_crop):
    _, binary = cv2.threshold(mask_crop, 127, 255, cv2.THRESH_BINARY)
    ph, pw = binary.shape
    padded = cv2.copyMakeBorder(binary,
                                int(ph*0.2), int(ph*0.2),
                                int(pw*0.2), int(pw*0.2),
                                cv2.BORDER_CONSTANT, value=0)
    resized    = cv2.resize(padded, (20, 20), interpolation=cv2.INTER_AREA)
    _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    final      = cv2.copyMakeBorder(resized, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
    normalized = final / 255.0
    return normalized.reshape(1, 28, 28, 1)

def recognise(paper):
    result = paper.copy()
    ph, pw = paper.shape[:2]
    mask, ink = get_ink_mask(paper)
    MAX_W = int(pw * 0.20)
    MAX_H = int(ph * 0.40)
    MIN_W = max(8,  int(pw * 0.010))
    MIN_H = max(15, int(ph * 0.025))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if (cv2.contourArea(c) >= 100 and
                MIN_W <= w <= MAX_W and
                MIN_H <= h <= MAX_H and
                w/h <= 2.0):
            valid.append(c)
    if not valid:
        cv2.putText(result, "No digits found", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        return result, [], ink
    valid  = sorted(valid, key=lambda c: cv2.boundingRect(c)[0])
    digits = []
    for c in valid:
        x, y, w, h = cv2.boundingRect(c)
        if w/h < 0.25:
            digit, conf = 1, 99.0
        else:
            crop = mask[y:y+h, x:x+w]
            inp  = prepare_digit(crop)
            pred = model.predict(inp, verbose=0)
            digit = int(np.argmax(pred))
            conf  = float(np.max(pred)) * 100
        if conf < 60.0:
            cv2.rectangle(result, (x,y), (x+w,y+h), (150,150,150), 1)
            cv2.putText(result, f"?{conf:.0f}%", (x,y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)
            continue
        digits.append(digit)
        cv2.rectangle(result, (x,y), (x+w,y+h), (0,220,0), 2)
        cv2.putText(result, f"{digit}({conf:.0f}%)", (x,y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,200,255), 2)
    summary = "  ".join(str(d) for d in digits) if digits else "Nothing recognised"
    cv2.rectangle(result, (0,0), (pw,42), (0,0,0), -1)
    cv2.putText(result, f"Digits: {summary}  [{ink}]", (8,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,255,0), 2)
    return result, digits, ink

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame grab failed"); break
    preview = frame.copy()
    _, coords = find_paper(frame)
    if coords:
        x1, y1, x2, y2 = coords
        cv2.rectangle(preview, (x1,y1), (x2,y2), (0,200,255), 2)
        cv2.putText(preview, "Paper found — press S to capture",
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)
    else:
        cv2.putText(preview, "Hold white paper up to camera",
                    (30,50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,80,255), 2)
    cv2.putText(preview, "S = capture   Q = quit",
                (10, preview.shape[0]-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)
    cv2.imshow("Digit Recognition", preview)
    key = cv2.waitKey(30) & 0xFF
    if key == ord('s'):
        paper, coords = find_paper(frame)
        if paper is None or paper.size == 0:
            warn = frame.copy()
            cv2.putText(warn, "No paper detected!", (30,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
            cv2.imshow("Digit Recognition", warn)
            cv2.waitKey(1500)
            continue
        annotated, digits, ink = recognise(paper)
        capture_count += 1
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname    = f"capture_{capture_count}_{ts}"
        img_path = os.path.join(RESULTS_FOLDER, fname+".jpg")
        txt_path = os.path.join(RESULTS_FOLDER, fname+".txt")
        cv2.imwrite(img_path, annotated)

# Save to text file
with open(txt_path, "w") as f:
    f.write(f"Time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Ink    : {ink}\n")
    f.write(f"Digits : {' '.join(str(d) for d in digits)}\n")
    f.write(f"Count  : {len(digits)}\n")

# Save to Excel file
import openpyxl
excel_path = os.path.join(RESULTS_FOLDER, "results.xlsx")
if os.path.exists(excel_path):
    wb = openpyxl.load_workbook(excel_path)
    ws = wb.active
else:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Digit Results"
    ws.append(["Capture #", "Ink Type", "Digit 1", "Digit 2", "Digit 3", "Digit 4", "Digit 5", "Digit 6", "Digit 7", "Digit 8", "Digit 9", "Digit 10"])

row = [capture_count, ink] + digits + [""] * (10 - len(digits))
ws.append(row)
wb.save(excel_path)
print(f"📊 Excel updated → {excel_path}")

cap.release()
cv2.destroyAllWindows()
print(f"\n✅ Done. {capture_count} capture(s) saved to '{RESULTS_FOLDER}/'")