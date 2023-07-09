from PIL import Image
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

# Load the images
img1 = cv2.imread('./images/img1.jpg')
img2 = cv2.imread('./images/ccimg1.jpg')

# Convert the images to NumPy arrays
img1_array = np.array(img1)
img2_array = np.array(img2)

# Calculate the structural similarity index (SSIM)
ssim_index = ssim(img1_array, img2_array, multichannel=True)

print('Structural Similarity Index:', ssim_index)
