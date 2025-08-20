import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Cannot open camera")
    # Try other indices if 0 fails
    for i in [-1, 1, 2]: # -1 for any available camera, 1 and 2 for specific devices
        print(f"Trying camera index: {i}")
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"✅ Camera opened with index {i}")
            break
    if not cap.isOpened():
        print("❌ All attempts to open camera failed.")
        exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Cannot grab frame")
        break

    print("Mean pixel value:", frame.mean())  # Debug line
    cv2.imshow("Debug Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
