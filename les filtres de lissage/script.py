import cv2
import numpy as np
import matplotlib.pyplot as plt

path_input = "TP2\InputImages\img.jpg"

input_img = cv2.imread(path_input)

resized_img = cv2.resize(input_img, (300, 300))

output_img_colored = cv2.imwrite(
    "Image Processing\\TP2\\OutputImages\\resize_img.jpg", resized_img)

# Gaussian kernel size 5,5
filtrage_gaussian5 = cv2.GaussianBlur(resized_img, (5, 5), 2)

filtrage_average5 = cv2.blur(resized_img, (5, 5))

filtrage_median5 = cv2.medianBlur(resized_img, 5)


# Showing and Saving all filters Gaussien, Average and Median
cv2.imshow("Gaussian blur", filtrage_gaussian5)
output_gaussian_img = cv2.imwrite(
    "Image Processing\\TP2\\OutputImages\\gaussian5_img.jpg", filtrage_gaussian5)
cv2.imshow("Average filtering", filtrage_average5)
output_average_img = cv2.imwrite(
    "Image Processing\\TP2\\OutputImages\\average5_img.jpg", filtrage_average5)
cv2.imshow("Median filtering", filtrage_median5)
output_median_img = cv2.imwrite(
    "Image Processing\\TP2\\OutputImages\\median5_img.jpg", filtrage_median5)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Gaussian kernel size 3,3

filtrage_gaussian3 = cv2.GaussianBlur(resized_img, (3, 3), 2)

filtrage_average3 = cv2.blur(resized_img, (3, 3))

filtrage_median3 = cv2.medianBlur(resized_img, 3)


# Showing and Saving all filters Gaussien, Average and Median
cv2.imshow("Gaussian blur", filtrage_gaussian3)
cv2.imwrite(
    "Image Processing\\TP2\\OutputImages\\gaussian3_img.jpg", filtrage_gaussian3)
cv2.imshow("Average filtering", filtrage_average3)
cv2.imwrite(
    "Image Processing\\TP2\\OutputImages\\average3_img.jpg", filtrage_average3)
cv2.imshow("Median filtering", filtrage_median3)
cv2.imwrite(
    "Image Processing\\TP2\\OutputImages\\median3_img.jpg", filtrage_median3)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Gaussian kernel size 7,7

filtrage_gaussian7 = cv2.GaussianBlur(resized_img, (7, 7), 2)

filtrage_average7 = cv2.blur(resized_img, (7, 7))

filtrage_median7 = cv2.medianBlur(resized_img, 7)


# Showing and Saving all filters Gaussien, Average and Median
cv2.imshow("Gaussian blur", filtrage_gaussian7)
cv2.imwrite(
    "Image Processing\\TP2\\OutputImages\\gaussian7_img.jpg", filtrage_gaussian7)
cv2.imshow("Average filtering", filtrage_average7)
cv2.imwrite(
    "Image Processing\\TP2\\OutputImages\\average7_img.jpg", filtrage_average7)
cv2.imshow("Median filtering", filtrage_median7)
cv2.imwrite(
    "Image Processing\\TP2\\OutputImages\\median7_img.jpg", filtrage_median7)

cv2.waitKey(0)
cv2.destroyAllWindows()


# Function to apply Gaussian filter to an image

def gaussian_filter(image, sigma):
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape

    filtered_image = np.zeros_like(image)

    # Calculate kernel size based on sigma
    kernel_size = int(6 * sigma + 1)

    # Construct Gaussian kernel
    kernel = np.fromfunction(lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x - kernel_size // 2)
                             ** 2 + (y - kernel_size // 2) ** 2) / (2 * sigma ** 2)), (kernel_size, kernel_size))

    # Normalize kernel
    kernel /= np.sum(kernel)

    # Apply filter
    for i in range(height - kernel_size + 1):
        for j in range(width - kernel_size + 1):
            if len(image.shape) == 3:
                for c in range(image.shape[2]):
                    filtered_image[i + kernel_size // 2, j + kernel_size // 2, c] = np.sum(image[i:i+kernel_size, j:j+kernel_size, c] * kernel)
            else:
                filtered_image[i + kernel_size // 2, j + kernel_size // 2] = np.sum(image[i:i+kernel_size, j:j+kernel_size] * kernel)

    return filtered_image

def average_filter(image, size):
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape

    filtered_image = np.zeros_like(image)

    for i in range(height - size + 1):
        for j in range(width - size + 1):
            if len(image.shape) == 3:
                for c in range(image.shape[2]):
                    filtered_image[i + size // 2, j + size // 2, c] = np.mean(image[i:i+size, j:j+size, c])
            else:
                filtered_image[i + size // 2, j + size // 2] = np.mean(image[i:i+size, j:j+size])

    return filtered_image

def median_filter(image, size):
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape

    filtered_image = np.zeros_like(image)

    for i in range(height - size + 1):
        for j in range(width - size + 1):
            if len(image.shape) == 3:
                for c in range(image.shape[2]):
                    filtered_image[i + size // 2, j + size // 2, c] = np.median(image[i:i+size, j:j+size, c])
            else:
                filtered_image[i + size // 2, j + size // 2] = np.median(image[i:i+size, j:j+size])

    return filtered_image

# Testing
# image = np.random.rand(200, 200)
sigma = 2  # Standard deviation for Gaussian filter
size = 5   # Kernel size for average and median filters
filtered_gaussian = gaussian_filter(resized_img, sigma)
filtered_average = average_filter(resized_img, size)
filtered_median = median_filter(resized_img, size)
cv2.imwrite('Image Processing\\TP2\\OutputImages\\filtered_median.jpg', filtered_median)
cv2.imwrite('Image Processing\\TP2\\OutputImages\\filtered_average.jpg', filtered_average)
cv2.imwrite('Image Processing\\TP2\\OutputImages\\filtered_gaussian.jpg', filtered_gaussian)
# Display results
plt.figure(figsize=(10, 5))
plt.subplot(1, 4, 1)
plt.imshow(resized_img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(filtered_gaussian, cmap='gray')
cv2.imwrite(
    'TP2\OutputImages\\filtered_gaussian.jpg', filtered_gaussian)
plt.title('Gaussian Filter')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(filtered_average, cmap='gray')
cv2.imwrite(
    'TP2\OutputImages\\filtered_average.jpg', filtered_average)
plt.title('Average Filter')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(filtered_median, cmap='gray')
cv2.imwrite(
    'TP2\OutputImages\\filtered_median.jpg', filtered_median)
plt.title('Median Filter')
plt.axis('off')

plt.show()
