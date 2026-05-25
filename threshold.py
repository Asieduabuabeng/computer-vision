import matplotlib
matplotlib.use('MacOSX')
import cv2 
import matplotlib.pyplot as plt 

image = cv2.imread("test.jpg")

grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret, thresh_simple = cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY)

ret, thresh_inverse = cv2.threshold(grey, 127, 225, cv2.THRESH_BINARY_INV)

thresh_adaptive = cv2.adaptiveThreshold(
    grey,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)

fig, axes = plt.subplots(1, 4, figsize=(18, 5))

axes[0].imshow(grey, cmap="gray")
axes[0].set_title("Greyscale original")
axes[0].axis("off")

axes[1].imshow(thresh_simple, cmap="gray")
axes[1].set_title("Simple threshold (127)")
axes[1].axis("off")

axes[2].imshow(thresh_inverse, cmap="gray")
axes[2].set_title("Inverse threshold (127)")
axes[2].axis("off")

axes[3].imshow(thresh_adaptive, cmap="gray")
axes[3].set_title("Adaptive threshold")
axes[3].axis("off")

plt.tight_layout()
plt.savefig("outputs/output_threshold.png", dpi=150, bbox_inches='tight')
plt.show()