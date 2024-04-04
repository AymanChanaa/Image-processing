import cv2
import matplotlib.pyplot as plt
import numpy as np

path_input = "Image Processing\TP3\InputImages\hand.png"
input_img = cv2.imread(path_input)
resized_img = cv2.resize(input_img, (300, 300))
blur_img = cv2.GaussianBlur(resized_img, (13, 13), 0)

# Apply Canny edge detection
Canny_edges = cv2.Canny(blur_img, threshold1=100, threshold2=100)

# Sobel edge detection
sobel_x = cv2.Sobel(blur_img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blur_img, cv2.CV_64F, 0, 1, ksize=3)
sobel_edges = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

# Prewitt edge detection
kernel_prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
kernel_prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
prewitt_x = cv2.filter2D(blur_img, -1, kernel_prewitt_x)
prewitt_y = cv2.filter2D(blur_img, -1, kernel_prewitt_y)
prewitt_edges = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)

# Roberts edge detection
kernel_roberts_x = np.array([[1, 0], [0, -1]])
kernel_roberts_y = np.array([[0, 1], [-1, 0]])
roberts_x = cv2.filter2D(blur_img, -1, kernel_roberts_x)
roberts_y = cv2.filter2D(blur_img, -1, kernel_roberts_y)
roberts_edges = cv2.addWeighted(roberts_x, 0.5, roberts_y, 0.5, 0)

# Laplacian edge detection
laplacian_edges = cv2.Laplacian(blur_img, cv2.CV_64F)

# Display images using Matplotlib
plt.figure(figsize=(8, 4))

plt.subplot(2, 3, 1)
plt.xticks([])
plt.yticks([])
plt.imshow(resized_img)
cv2.imwrite(
    'Image Processing\TP3\OutputImages\\resized_img.jpg', resized_img)
plt.title("Original Image")

plt.subplot(2, 3, 2)
plt.xticks([])
plt.yticks([])
plt.imshow(Canny_edges, cmap='gray')
cv2.imwrite(
    'Image Processing\TP3\OutputImages\\Canny_edges.jpg', Canny_edges)
plt.title("Canny Edges")

plt.subplot(2, 3, 3)
plt.xticks([])
plt.yticks([])
plt.imshow(sobel_edges, cmap='gray')
cv2.imwrite(
    'Image Processing\TP3\OutputImages\\sobel_edges.jpg', sobel_edges)
plt.title("Sobel Edges")

plt.subplot(2, 3, 4)
plt.xticks([])
plt.yticks([])
plt.imshow(prewitt_edges, cmap='gray')
cv2.imwrite(
    'Image Processing\TP3\OutputImagess\\prewitt_edges.jpg', prewitt_edges)
plt.title("Prewitt Edges")

plt.subplot(2, 3, 5)
plt.xticks([])
plt.yticks([])
plt.imshow(roberts_edges, cmap='gray')
cv2.imwrite(
    'Image Processing\TP3\OutputImages\\roberts_edges.jpg', roberts_edges)
plt.title("Roberts Edges")

plt.subplot(2, 3, 6)
plt.xticks([])
plt.yticks([])
plt.imshow(laplacian_edges, cmap='gray')
cv2.imwrite(
    'Image Processing\TP3\OutputImages\\laplacian_edges.jpg', laplacian_edges)
plt.title("Laplacian Edges")

plt.tight_layout()
plt.show()
