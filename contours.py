import matplotlib
matplotlib.use('MacOSX')
import cv2
import matplotlib.pyplot as plt

# ── Load and prepare ─────────────────────────────────────
# Load the image from disk
image = cv2.imread("test.jpg")

# Make a copy to draw on — we don't want to modify the original
# image.copy() creates an independent duplicate of the array
image_contours    = image.copy()
image_boxes       = image.copy()

# Convert to greyscale — contours need single channel input
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Blur to reduce noise before thresholding
# Noise creates false contours — blurring prevents this
blurred = cv2.GaussianBlur(grey, (5, 5), 0)

# ── Create binary image ──────────────────────────────────
# Threshold to get black and white image
# Adaptive handles the uneven lighting in this image better
thresh = cv2.adaptiveThreshold(
    blurred,                         # input — blurred greyscale
    255,                             # value for pixels above threshold
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # gaussian weighted neighbourhood
    cv2.THRESH_BINARY,               # binary output only
    11,                              # block size — must be odd
    2                                # constant subtracted from mean
)

# ── Find contours ────────────────────────────────────────
# findContours returns a list of contours
# Each contour is a list of points describing one boundary
contours, hierarchy = cv2.findContours(
    thresh,                 # binary image to find contours in
    cv2.RETR_EXTERNAL,      # only find outermost contours
    cv2.CHAIN_APPROX_SIMPLE # store only corner points, not every point
)

print(f"Total contours found: {len(contours)}")

# ── Filter small contours ────────────────────────────────
# Small contours are usually noise — filter by minimum area
# Only keep contours with area greater than 100 pixels
min_area = 100
large_contours = [c for c in contours if cv2.contourArea(c) > min_area]
print(f"Contours after filtering small ones: {len(large_contours)}")

# ── Draw contours ────────────────────────────────────────
# Draw all large contours onto image_contours
# -1 means draw every contour in the list
# (0, 255, 0) is green in BGR
# 2 is the line thickness
cv2.drawContours(image_contours, large_contours, -1, (0, 255, 0), 2)

# ── Draw bounding boxes ──────────────────────────────────
# For each contour, calculate and draw its bounding rectangle
for contour in large_contours:
    # boundingRect returns top-left corner (x,y) and size (w,h)
    x, y, w, h = cv2.boundingRect(contour)

    # Draw rectangle on image_boxes
    # (x, y) = top left corner
    # (x+w, y+h) = bottom right corner
    # (255, 0, 0) = blue in BGR
    # 2 = line thickness
    cv2.rectangle(image_boxes, (x, y), (x+w, y+h), (255, 0, 0), 2)

# ── Display ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 5))

# Panel 1 — original image converted to RGB for Matplotlib
axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original")
axes[0].axis("off")

# Panel 2 — binary threshold image used for finding contours
axes[1].imshow(thresh, cmap="gray")
axes[1].set_title("Binary threshold")
axes[1].axis("off")

# Panel 3 — contours drawn as green lines
axes[2].imshow(cv2.cvtColor(image_contours, cv2.COLOR_BGR2RGB))
axes[2].set_title(f"Contours ({len(large_contours)} found)")
axes[2].axis("off")

# Panel 4 — bounding boxes drawn as blue rectangles
axes[3].imshow(cv2.cvtColor(image_boxes, cv2.COLOR_BGR2RGB))
axes[3].set_title("Bounding boxes")
axes[3].axis("off")

plt.tight_layout()

# Save output before displaying
plt.savefig("outputs/output_contours.png", dpi=150, bbox_inches='tight')
plt.show()