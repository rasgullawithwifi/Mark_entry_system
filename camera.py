import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        print(f"Index {i}: opened={cap.isOpened()}, frame_ok={ret}")
        cap.release()