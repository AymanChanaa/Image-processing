import os
from PIL import Image, ImageDraw
import cv2
from matplotlib import pyplot as plt
import numpy as np


def log_power_spectrum(image):
    # Convertir l'image en niveaux de gris
    if image.mode != 'L':
        image = image.convert('L')

    # Calculer la transformation de Fourier 2D de l'image
    fft_img = np.fft.fft2(image)

    # Décalage du centre de la transformée de Fourier
    fft_shifted = np.fft.fftshift(fft_img)

    # Calculer le logarithme de la magnitude de la transformée de Fourier
    log_magnitude_fft = np.log1p(np.abs(fft_shifted))

    # Mettre à l'échelle les valeurs pour les convertir en entiers 8 bits
    log_magnitude_fft_scaled = (
        log_magnitude_fft / np.max(log_magnitude_fft)) * 255

    # Créer une image PIL à partir des valeurs échelonnées
    log_magnitude_fft_image = Image.fromarray(
        log_magnitude_fft_scaled.astype(np.uint8))

    return log_magnitude_fft_image

def segment_and_reconstruct_grid(image):
    # Convertir l'image en niveaux de gris
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Appliquer un flou gaussien pour réduire le bruit
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Appliquer un seuillage adaptatif pour binariser l'image
    _, threshold_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Appliquer la détection de contours
    contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Créer une image masquée pour stocker les contours détectés
    mask = np.zeros_like(gray_image)

    # Dessiner les contours détectés sur l'image masquée
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    # Reconstruire la grille en utilisant la transformée de Fourier inverse
    grid_reconstructed = cv2.bitwise_and(gray_image, mask)

    return grid_reconstructed, mask

# Charger l'image
image_path = "Manip2\InputImages\MickyMouse.png"
image = Image.open(image_path)

log_magnitude_fft_image = log_power_spectrum(image)

output_path = os.path.join("Manip2\OutputImages\\",
                           "MickyMouse_log.png")
log_magnitude_fft_image.save(output_path)

# Afficher l'image générée avec Matplotlib
plt.imshow(log_magnitude_fft_image, cmap='gray')
plt.title('Log Power Spectrum')
plt.axis('off')  # Masquer les axes


# Appliquer la fonction pour segmenter et reconstruire la grille
image2 = cv2.imread("Manip2\InputImages\MickyMouse2.jpg")
grid_reconstructed, mask = segment_and_reconstruct_grid(image2)

# Afficher l'image originale, l'image masquée et l'image reconstruite de la grille
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
plt.title('Image Originale')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap='gray')
plt.title('Contours Détectés (Masque)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(grid_reconstructed, cmap='gray')
plt.title('Grille Reconstruite')
plt.axis('off')
plt.savefig("Manip2\OutputImages\\Segmentation.png")
plt.show()