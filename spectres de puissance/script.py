import cv2
import numpy as np
import matplotlib.pyplot as plt


def Power_Spectrum(img, subplot_index):
    fourier = np.fft.fft2(img)
    fourier_shifted = np.fft.fftshift(fourier)
    power_spectrum = np.abs(fourier_shifted) ** 2

    plt.subplot(2, 4, subplot_index)
    plt.imshow(img, cmap='gray')
    plt.title('Image ' + str(subplot_index))
    plt.axis('off')

    plt.subplot(2, 4, subplot_index + 4)
    plt.imshow(np.log(1 + power_spectrum), cmap='gray')
    plt.title('Power Spectrum ' + str(subplot_index))
    plt.colorbar()


def radial_power_spectrum(image):
    # Convertir l'image en niveaux de gris
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Effectuer la transformation de Fourier
    f = np.fft.fft2(gray_image)
    fshift = np.fft.fftshift(f)
    
    # Calculer le spectre de puissance
    power_spectrum = np.abs(fshift) ** 2
    
    # Calculer le spectre de puissance radial
    rows, cols = power_spectrum.shape
    center_row, center_col = rows // 2, cols // 2
    
    max_radius = int(np.sqrt(center_row**2 + center_col**2)) // 2
    radial_power_spectrum = np.zeros(max_radius)
    
    for r in range(rows):
        for c in range(cols):
            radius = int(np.sqrt((r - center_row) ** 2 + (c - center_col) ** 2))
            if radius < max_radius:
                radial_power_spectrum[radius] += power_spectrum[r, c]
    
    return radial_power_spectrum

def angular_power_spectrum(image):
    # Convertir l'image en niveaux de gris
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Effectuer la transformation de Fourier
    f = np.fft.fft2(gray_image)
    fshift = np.fft.fftshift(f)
    
    # Calculer le spectre de puissance
    power_spectrum = np.abs(fshift) ** 2
    
    # Calculer le spectre de puissance angulaire
    rows, cols = power_spectrum.shape
    center_row, center_col = rows // 2, cols // 2
    
    max_radius = int(np.sqrt(center_row**2 + center_col**2)) // 2
    angular_power_spectrum = np.zeros(max_radius)
    
    for r in range(rows):
        for c in range(cols):
            radius = int(np.sqrt((r - center_row) ** 2 + (c - center_col) ** 2))
            if radius < max_radius:
                theta = np.arctan2(r - center_row, c - center_col)
                if theta < 0:
                    theta += 2 * np.pi
                angular_bin = int((theta / (2 * np.pi)) * max_radius)
                angular_power_spectrum[angular_bin] += power_spectrum[r, c]
    
    return angular_power_spectrum


image = cv2.imread('Manip1\InputImages\horizontal_8.png')

# Calculer le spectre de puissance radial
radial_spectrum = radial_power_spectrum(image)

# Calculer le spectre de puissance angulaire
angular_spectrum = angular_power_spectrum(image)

plt.plot(angular_spectrum)
plt.title('Spectre de Puissance Angulaire')
plt.xlabel('Angle')
plt.ylabel('Amplitude')
plt.savefig("Manip1\outputImages\\angular_spectrum.png")

# Afficher les spectres fusionnés
plt.plot(radial_spectrum, label='Radial Spectrum')
plt.plot(angular_spectrum, label='Angular Spectrum')
plt.title('Spectre de Puissance Radial et Angulaire')
plt.xlabel('Fréquence / Angle')
plt.ylabel('Amplitude')
plt.legend()
plt.savefig("Manip1\outputImages\spectrum.png")

plt.figure(figsize=(14, 6))
# Charger l'image
image1 = cv2.imread('Manip1\InputImages\horizontal_8.png',
                    cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('Manip1\InputImages\horizontal_16.png',
                    cv2.IMREAD_GRAYSCALE)
image3 = cv2.imread('Manip1\InputImages\diagonal_8.png',
                    cv2.IMREAD_GRAYSCALE)
image4 = cv2.imread('Manip1\InputImages\\vertical_8.png',
                    cv2.IMREAD_GRAYSCALE)

Power_Spectrum(image1, 1)
Power_Spectrum(image2, 2)
Power_Spectrum(image3, 3)
Power_Spectrum(image4, 4)

plt.show()
