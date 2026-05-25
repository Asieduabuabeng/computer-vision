import cv2
import matplotlib.pyplot as plt
import numpy as np

image_bgr = cv2.imread("test.jpg")

if image_bgr is None:
    print("Error: image not found. Check the file path.")
else: 
    print("Image loaded successfully")
    print("Shape:", image_bgr.shape)
    print("Height:", image_bgr.shape[0])
    print("Width:", image_bgr.shape[1])


image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
image_grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(image_rgb)
axes[0].set_title("Original (RGB)")
axes[0].axis("off")

axes[1].imshow(image_grey, cmap="gray")
axes[1].set_title("Greyscale")
axes[1].axis("off")

axes[2].imshow(image_hsv)
axes[2].set_title("HSV")
axes[2].axis("off")

axes[3].imshow(image_hsv[:, :, 0], cmap="hsv")
axes[3].set_title("Hue channel only")
axes[3].axis("off")

plt.tight_layout()
plt.show()