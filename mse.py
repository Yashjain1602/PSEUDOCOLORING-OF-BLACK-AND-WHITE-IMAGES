import numpy as np
import cv2

# Load the two images
img1 = cv2.imread('./images/img10.jpg')
img2 = cv2.imread('./images/ccimg10.jpg')

# Convert the images to arrays
arr1 = np.array(img1, dtype=np.float32)
arr2 = np.array(img2, dtype=np.float32)

# Compute the element-wise difference
diff = arr1 - arr2

# Square each element of the difference array
squared_diff = np.square(diff)

# Compute the average of the squared difference array
mse = np.mean(squared_diff)

print("Mean Squared Error:", mse)