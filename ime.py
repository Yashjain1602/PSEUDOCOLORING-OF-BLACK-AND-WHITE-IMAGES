from PIL import Image
import numpy as np
import cv2
from scipy.signal import correlate2d
from skimage.metrics import mean_squared_error, structural_similarity#, mutual_info_score

# Load the images
img1 = cv2.imread('./images/img1.jpg')
img2 = cv2.imread('./images/ccimg1.jpg')

# Convert the images to NumPy arrays
img1_array = np.array(img1)
img2_array = np.array(img2)

# Calculate the mean squared error (MSE)
mse = mean_squared_error(img1_array, img2_array)

# Calculate the normalized cross-correlation (NCC)
ncc = np.max(correlate2d(img1_array, img2_array))

# Calculate the mutual information (MI)
#mi = mutual_info_score(img1_array.ravel(), img2_array.ravel())

print('Mean Squared Error:', mse)
print('Normalized Cross-Correlation:', ncc)
#print('Mutual Information:', mi)
