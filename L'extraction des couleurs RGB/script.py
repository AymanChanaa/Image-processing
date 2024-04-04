import cv2
import matplotlib.pyplot as plt
import numpy as np

# Getting images from InputImages folder
path_input_bw = 'Image Processing\\TP1\\InputImages\\black_white_img.jpg'
path_input = 'Image Processing\\TP1\\InputImages\\color_img.jpg'

# Storage of the images
input_img_bw = cv2.imread(path_input_bw)
input_img = cv2.imread(path_input)

# Resize images
resized_img_bw = cv2.resize(input_img_bw, (300, 300))
resized_img = cv2.resize(input_img, (300, 300))

# Extract (r,g,b) from images
r1, g1, b1 = cv2.split(resized_img_bw)
r2, g2, b2 = cv2.split(resized_img)

# Seeing rezied images
cv2.imshow("Resized Image Bw", resized_img_bw)
cv2.imshow("Resized Image", resized_img)

# Insert my rezied images in outputImages folder
output_img_blackwhite = cv2.imwrite(
    'Image Processing\\TP1\\OutputImages\\resize_img_bw.jpg', resized_img_bw)
output_img_colored = cv2.imwrite(
    "Image Processing\\TP1\\OutputImages\\resize_img.jpg", resized_img)

# Utilisons les fonctions de OpenCV
# Black and White histogramme
hist_r_bw_openCv = cv2.calcHist([r1], [0], None, [256], [0, 256])
hist_g_bw_openCv = cv2.calcHist([g1], [0], None, [256], [0, 256])
hist_b_bw_openCv = cv2.calcHist([b1], [0], None, [256], [0, 256])

# Color image histogramme
hist_r_openCv = cv2.calcHist([r2], [0], None, [256], [0, 256])
hist_g_openCv = cv2.calcHist([g2], [0], None, [256], [0, 256])
hist_b_openCv = cv2.calcHist([b2], [0], None, [256], [0, 256])

# Our function my_histogram
def my_histogram(image):
    hist_r = np.zeros(256)
    hist_g = np.zeros(256)
    hist_b = np.zeros(256)

    # We go through each pixel
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel = image[i, j]

            hist_r[pixel[0]] += 1
            hist_g[pixel[1]] += 1
            hist_b[pixel[2]] += 1

    return hist_r, hist_g, hist_b


# Black and White histogram
hist_r_bw, hist_g_bw, hist_b_bw = my_histogram(resized_img_bw)
# Color image histogram
hist_r, hist_g, hist_b = my_histogram(resized_img)

# Black and white histogram
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Histogram RGB BW Image - my function")
plt.xlabel("Intensity")
plt.ylabel("Number of pixels")
plt.plot(hist_r_bw, color='r', label='R')
plt.plot(hist_g_bw, color='g', label='G')
plt.plot(hist_b_bw, color='b', label='B')
plt.xlim([0, 256])
plt.legend()


plt.subplot(1, 2, 2)
plt.title("Histogram RGB BW Image - opencv")
plt.xlabel("Intensity")
plt.ylabel("Number of pixels")
plt.plot(hist_r_bw_openCv, color='r', label='R')
plt.plot(hist_g_bw_openCv, color='g', label='G')
plt.plot(hist_b_bw_openCv, color='b', label='B')
plt.xlim([0, 256])
plt.legend()
plt.tight_layout()


# Color Image
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Histogram RGB color Image - my function")
plt.xlabel("Intensity")
plt.ylabel("Number of pixels")
plt.plot(hist_r, color='r', label='R')
plt.plot(hist_g, color='g', label='G')
plt.plot(hist_b, color='b', label='B')
plt.xlim([0, 256])
plt.legend()

plt.subplot(1, 2, 2)
plt.title("Histogram RGB color Image - opencv")
plt.xlabel("Intensity")
plt.ylabel("Number of pixels")
plt.plot(hist_r_openCv, color='r', label='R')
plt.plot(hist_g_openCv, color='g', label='G')
plt.plot(hist_b_openCv, color='b', label='B')
plt.xlim([0, 256])
plt.legend()
plt.tight_layout()
plt.show()
