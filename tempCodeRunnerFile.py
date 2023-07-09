import numpy as np 
import cv2
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt
import numpy as np

def mse(imageA, imageB):
    	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err
def compare_images(imageA, imageB, title):
    	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	# s = ssim(imageA, imageB,multichannel=True)
	# setup the figure
	fig = plt.figure(title)
	# plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA, cmap = plt.cm.gray)
	plt.axis("off")
	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB, cmap = plt.cm.gray)
	plt.axis("off")
	# show the images
	plt.show()

print("loading models.....")
net = cv2.dnn.readNetFromCaffe('./model/colorization_deploy_v2.prototxt','./model/colorization_release_v2.caffemodel')
pts = np.load('./model/pts_in_hull.npy')


class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2,313,1,1)

net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1,313],2.606,dtype='float32')]


image = cv2.imread('./images/img6.jpg')
compare_image = cv2.imread('./images/img6.jpg')
scaled = image.astype("float32")/255.0
lab = cv2.cvtColor(scaled,cv2.COLOR_BGR2LAB)

print(compare_image.size)
print(image.size)
resized = cv2.resize(lab,(224,224))
L = cv2.split(resized)[0]
L -= 50


net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1,2,0))

ab = cv2.resize(ab, (image.shape[1],image.shape[0]))

L = cv2.split(lab)[0]
colorized = np.concatenate((L[:,:,np.newaxis], ab), axis=2)

colorized = cv2.cvtColor(colorized,cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized,0,1)

colorized = (255 * colorized).astype("uint8")


cv2.imshow("Original",image)
cv2.imshow("Colorized",colorized)
compare_images(compare_image ,colorized,"Orignal vs Colored")
cv2.waitKey(0)