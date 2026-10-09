import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load trained model
model = tf.keras.models.load_model("digit_model.h5")
print("✅ Model loaded successfully")

# Load image
img = cv2.imread('digit.JPG', cv2.IMREAD_COLOR)
if img is None:
    print("❌ Image not found.")
    exit()
print("✅ Image loaded")
original_img = img.copy()

# STEP 1: Detect ink type and create mask
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Red ink
mask1    = cv2.inRange(img_hsv, np.array([0, 40, 40]),   np.array([15, 255, 255]))
mask2    = cv2.inRange(img_hsv, np.array([160, 40, 40]), np.array([180, 255, 255]))
red_mask = cv2.bitwise_or(mask1, mask2)

# Blue ink
blue_mask = cv2.inRange(img_hsv, np.array([100, 40, 40]), np.array([130, 255, 255]))

red_pixels  = cv2.countNonZero(red_mask)
blue_pixels = cv2.countNonZero(blue_mask)

if red_pixels > 50:
    ink_mask = red_mask
    ink_type = "Red ink"
elif blue_pixels > 50:
    ink_mask = blue_mask
    ink_type = "Blue ink"
else:
    gray     = cv2.GaussianBlur(gray, (5, 5), 0)
    ink_mask = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=25, C=18
    )
    kernel_noise = np.ones((2, 2), np.uint8)
    ink_mask     = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, kernel_noise)
    ink_type     = "Black/Pencil"

print(f"🖊️  Detected: {ink_type}")

# STEP 2: Clean up mask
kernel   = np.ones((3, 3), np.uint8)
ink_mask = cv2.dilate(ink_mask, kernel, iterations=2)
ink_mask = cv2.morphologyEx(ink_mask, cv2.MORPH_CLOSE, kernel)

# STEP 3: Find ALL contours
contours, _ = cv2.findContours(ink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours:
    print("❌ No digits detected!")
    exit()

# STEP 4: Filter out noise contours
MIN_AREA   = 40
MIN_WIDTH  = 5
MIN_HEIGHT = 15

valid_contours = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    aspect_ratio = w / h

    if (area >= MIN_AREA and
        w >= MIN_WIDTH and
        h >= MIN_HEIGHT and
        w < 400 and          # ignore huge wide contours (table edges, lines)
        h < 400 and          # ignore huge tall contours
        aspect_ratio < 3.0): # ignore anything wider than 3x its height
        valid_contours.append(c)

if not valid_contours:
    print("❌ No valid digit contours found!")
    exit()

# STEP 5: Sort contours left to right
valid_contours = sorted(valid_contours, key=lambda c: cv2.boundingRect(c)[0])

print(f"\n✅ Found {len(valid_contours)} digit(s)")

# STEP 6: Predict each digit
predicted_digits = []
result_img       = original_img.copy()

for i, contour in enumerate(valid_contours):
    x, y, w, h   = cv2.boundingRect(contour)
    aspect_ratio = w / h
    area         = cv2.contourArea(contour)
    fill_ratio   = area / (w * h)

    # 👇 THIS IS THE NEW LINE
    print(f"  Digit {i+1}: aspect_ratio={aspect_ratio:.2f}, fill_ratio={fill_ratio:.2f}, size={w}x{h}")

    cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if aspect_ratio < 0.35:
        predicted_digit = 1
        confidence      = 0.0
        print(f"  → thin line rule → 1")
    else:
        digit_crop       = ink_mask[y:y+h, x:x+w]
        digit_resized    = cv2.resize(digit_crop, (20, 20))
        digit_padded     = cv2.copyMakeBorder(digit_resized, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
        digit_normalized = digit_padded / 255.0
        digit_input      = digit_normalized.reshape(1, 28, 28, 1)

        prediction       = model.predict(digit_input, verbose=0)
        predicted_digit  = int(np.argmax(prediction))
        confidence       = float(np.max(prediction)) * 100
        print(f"  → model → {predicted_digit} ({confidence:.1f}%)")

    predicted_digits.append(predicted_digit)

    cv2.putText(result_img, str(predicted_digit),
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 0, 255), 2)

print(f"\n🎯 All digits detected: {predicted_digits}")

# STEP 7: Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
axes[1].set_title(f"Detected Digits: {predicted_digits}")
axes[1].axis('off')

plt.tight_layout()
plt.show()