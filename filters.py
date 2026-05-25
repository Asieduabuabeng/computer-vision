import cv2
import matplotlib.pyplot as plt

image = cv2.imread("test.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

blur_average = cv2.blur(image_rgb, (5, 5))
blur_gaussian = cv2.GaussianBlur(image_rgb, (5, 5), 0)
blur_median = cv2.medianBlur(image_rgb, 5)

blur_heavy = cv2.GaussianBlur(image_rgb, (21, 21), 0)

fig, axes = plt.subplots(1, 5, figsize=(20, 4))

axes[0].imshow(image_rgb)
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(blur_average)
axes[1].set_title("Average Blur (5x5)")
axes[1].axis("off")

axes[2].imshow(blur_gaussian)
axes[2].set_title("Gaussian Blur (5x5)")
axes[2].axis("off")

axes[3].imshow(blur_median)
axes[3].set_title("Median Blur (5)")
axes[3].axis("off")

axes[4].imshow(blur_heavy)
axes[4].set_title("Heavy Blur (3x3)")
axes[4].axis("off")

plt.tight_layout()
plt.show()