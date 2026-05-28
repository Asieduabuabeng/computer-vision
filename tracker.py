import cv2
import numpy as np
import time

# ── Colour range for RED in HSV ──────────────────────────
# Red wraps around the HSV hue circle so we need two ranges
# Lower red range — hue values 0 to 10
lower_red1 = np.array([0,   150, 80])
upper_red1 = np.array([10,  255, 255])

# Upper red range — hue values 170 to 180
lower_red2 = np.array([170, 150, 80])
upper_red2 = np.array([180, 255, 255])

# ── Open webcam ──────────────────────────────────────────
# Camera 1 is the working camera on this M2 Mac
# Camera 0 opens but delivers no frames
cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

# Set camera resolution to match Camera 1 output
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Check webcam opened successfully
if not cap.isOpened():
    print("Error: could not open webcam")
    exit()

print("Webcam opened.")

# ── Warm up — wait for real frames ───────────────────────
# Keep reading until we get a non-black frame
# frame.sum() > 0 confirms real pixel data is arriving
print("Waiting for camera to initialise...")
attempts = 0
while True:
    ret, frame = cap.read()
    attempts += 1
    if ret and frame.sum() > 0:
        print(f"Camera ready after {attempts} attempts.")
        break
    if attempts > 100:
        print("Error: camera not delivering frames after 100 attempts.")
        cap.release()
        exit()
    time.sleep(0.1)

print("Point the yellow crosshair at your red phone screen.")
print("Watch Terminal for HSV values.")
print("Press Q in the Colour Tracker window to quit.")

# ── Main loop ────────────────────────────────────────────
while True:

    # Read one frame from the webcam
    # ret = True if frame read successfully
    # frame = image as a NumPy array (height, width, 3)
    ret, frame = cap.read()

    # Skip if frame not read correctly
    if not ret:
        continue

    # Skip black frames — camera still adjusting
    if frame.sum() == 0:
        continue

    # ── Process frame ────────────────────────────────────

    # Blur to reduce noise before colour detection
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    # Convert BGR to HSV
    # HSV separates colour from brightness — better for detection
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Create two masks — one per red HSV range
    # inRange returns white where colour matches, black elsewhere
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine both red masks into one
    # bitwise_or — white if either mask is white
    mask = cv2.bitwise_or(mask1, mask2)

    # ── Clean up mask ────────────────────────────────────

    # Kernel — 5x5 grid of ones for morphological operations
    kernel = np.ones((5, 5), np.uint8)

    # Erode — removes small noise dots
    mask = cv2.erode(mask, kernel, iterations=2)

    # Dilate — fills small holes back in
    mask = cv2.dilate(mask, kernel, iterations=2)

    # ── Find contours ────────────────────────────────────
    contours, _ = cv2.findContours(
        mask,                   # binary image to search
        cv2.RETR_EXTERNAL,      # only outermost contours
        cv2.CHAIN_APPROX_SIMPLE # store only corner points
    )

    # ── Debug — read HSV at center of frame ──────────────

    # Get frame dimensions
    height, width = frame.shape[:2]

    # Calculate center pixel position
    center_y = height // 2
    center_x = width  // 2

    # Read HSV value of center pixel
    center_hsv = hsv[center_y, center_x]

    # Print to Terminal
    print(f"Center HSV: H={center_hsv[0]}  S={center_hsv[1]}  V={center_hsv[2]}")

    # Draw yellow crosshair at center
    cv2.circle(frame, (center_x, center_y), 10, (0, 255, 255), 2)
    cv2.line(frame, (center_x-20, center_y), (center_x+20, center_y), (0, 255, 255), 2)
    cv2.line(frame, (center_x, center_y-20), (center_x, center_y+20), (0, 255, 255), 2)

    # Show HSV reading as text on frame
    cv2.putText(
        frame,
        f"HSV: H={center_hsv[0]} S={center_hsv[1]} V={center_hsv[2]}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # ── Draw detection if found ──────────────────────────
    if len(contours) > 0:

        # Find largest contour — most likely your red object
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        # Only draw if large enough — filters out noise
        if area > 500:

            # Get minimum enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(largest)

            # Convert to integers — drawing needs whole numbers
            center = (int(x), int(y))
            radius = int(radius)

            # Draw green circle around detected object
            cv2.circle(frame, center, radius, (0, 255, 0), 3)

            # Draw small filled dot at exact center
            cv2.circle(frame, center, 5, (0, 255, 0), -1)

            # Show detection text
            cv2.putText(
                frame,
                f"Red detected — area: {int(area)}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    # ── Show windows ─────────────────────────────────────
    cv2.imshow("Colour Tracker", frame)
    cv2.imshow("Mask", mask)

    # Force windows to specific positions on screen
    cv2.moveWindow("Colour Tracker", 100, 100)
    cv2.moveWindow("Mask", 100, 500)

    # ── Quit on Q ────────────────────────────────────────
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── Clean up ─────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
print("Tracker stopped.")