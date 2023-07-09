import numpy as np
from scipy.stats import entropy
from PIL import Image

# Load the image
img = Image.open('./images/ccimg10.jpg')

# Convert the image to grayscale
img_gray = img.convert('L')

# Convert the image to a NumPy array
img_array = np.array(img_gray)

# Calculate the histogram of pixel values
histogram, _ = np.histogram(img_array, bins=np.arange(0, 257))

# Normalize the histogram
histogram = histogram / float(img_array.size)

# Calculate the entropy
entropy_val = entropy(histogram)

print('Entropy:', entropy_val)
