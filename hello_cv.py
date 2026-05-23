import numpy as np
import cv2
import matplotlib.pyplot as plt

image = np.zeros((300, 300, 3), dtype=np.uint8)

image[50:150, 50:250] = [255, 0, 0]
image[150:250, 50:250] = [0, 0, 255]

print("Image shape:", image.shape)
print("Height:", image.shape[0])
print("Width:", image.shape[1])
print("Channels:", image.shape[2])
print("Total numbers:", image.shape[0] * image.shape[1] * image.shape[2])

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("My First CV image")
plt.axis("off")
plt.show()