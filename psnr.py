from PIL import Image
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio

# Load the images
img1 = cv2.imread('./images/img10.jpg')
img2 = cv2.imread('./images/ccimg10.jpg')

# Convert the images to NumPy arrays
img1_array = np.array(img1)
img2_array = np.array(img2)

# Calculate the peak signal-to-noise ratio (PSNR)
psnr = peak_signal_noise_ratio(img1_array, img2_array, data_range=img1_array.max())

print('Peak Signal-to-Noise Ratio:', psnr)
