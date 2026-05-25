import matplotlib
matplotlib.use('MacOSX')
import cv2
import matplotlib.pyplot as plt

# ── Load and prepare ─────────────────────────────────────
# Load the image from disk as a NumPy array
# OpenCV loads images in BGR order (not RGB)
image = cv2.imread("test.jpg")

# Convert to greyscale — Canny works on a single channel
# Reduces 3 channels down to 1 — faster and sufficient for edges
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Blur before edge detection — smooths out noise
# Noise causes false edges — blurring prevents that
blurred = cv2.GaussianBlur(grey, (5, 5), 0)

# ── Apply Canny with different thresholds ────────────────
# Low thresholds — catches more edges including weak/noisy ones
edges_low = cv2.Canny(blurred, 10, 50)

# Medium thresholds — balanced, good default starting point
edges_mid = cv2.Canny(blurred, 50, 150)

# High thresholds — only the strongest edges survive
edges_high = cv2.Canny(blurred, 100, 200)

# ── Display ──────────────────────────────────────────────
# Create a figure with 4 side-by-side panels
# figsize=(18, 5) sets the window size in inches
fig, axes = plt.subplots(1, 4, figsize=(18, 5))

# Panel 1 — original image
# Convert BGR to RGB so colours display correctly in Matplotlib
axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original")
axes[0].axis("off")   # hides the x and y axis numbers

# Panel 2 — low threshold edges
# cmap="gray" tells Matplotlib to display single-channel as black/white
axes[1].imshow(edges_low, cmap="gray")
axes[1].set_title("Low thresholds (10, 50)")
axes[1].axis("off")

# Panel 3 — medium threshold edges
axes[2].imshow(edges_mid, cmap="gray")
axes[2].set_title("Medium thresholds (50, 150)")
axes[2].axis("off")

# Panel 4 — high threshold edges
axes[3].imshow(edges_high, cmap="gray")
axes[3].set_title("High thresholds (100, 200)")
axes[3].axis("off")

# tight_layout adjusts spacing so panels don't overlap
plt.tight_layout()
plt.show()