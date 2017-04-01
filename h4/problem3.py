# CS 181, Spring 2017
# Homework 4: Clustering
# Name: Russell Kunes
# Email: russellkunes@college.harvard.edu

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class KMeans(object):
	# K is the K in KMeans
        def __init__(self, K):
		self.K = K


	#objective function for k means 
	def __objective(self, r, u, X):
		sm = 0
		for c in xrange(u.shape[0]):
			g = np.subtract(X[r == c],u[c])
			sm += sum(np.array([np.linalg.norm(ux) for ux in g]))
		return sm

	# X is a (N x 28 x 28) array where 28x28 is the dimensions of each of the N images.
	def fit(self, X):

		objective =[]
		K = self.K 
		X = np.array(X)
		self.X = X

		#initialize centroids
		u = np.zeros((K,28,28)) #make more general later

		#initialize assignment vectors
		r = np.zeros(X.shape[0]) 
		
		#randomly initialize the cluster centers
		for k in xrange(u.shape[0]):
			for i in xrange(u.shape[1]):
				for j in xrange(u.shape[2]):
					u[k,i,j] = np.random.uniform(0,200)

		stop_condition = False 

		while stop_condition == False:
			#copy 
			u_old = np.copy(u)
			#step 1:
			# loop through the N data points
			for i in xrange(X.shape[0]):
				x = X[i]
				diff = np.subtract(u,x)
				r[i] = np.argmin(np.array([np.linalg.norm(ux) for ux in diff]))

			#step 2:
			# set mu_k
			for j in xrange(u.shape[0]):
				u[j] = np.mean(X[r == j], axis = 0)

			objective.append(self.__objective(r,u,X))
			#check for convergence 

			if np.linalg.norm(u- u_old) < .000001:
				stop_condition = True


		self.objective = np.array(objective)
		self.u = u
		self.r = r

	# This should return the arrays for K images. Each image should represent the mean of each of the fitted clusters.
	def get_mean_images(self):
		return self.u 

	# This should return the arrays for D images from each cluster that are representative of the clusters.
	def get_representative_images(self, D):
		u = self.u
		r = self.r 
		K= self.K
		X = self.X
		closest  = []

		for k in xrange(K):
			newX = X[r==k]
			g = np.subtract(newX, u[k])
			dists = np.array([np.linalg.norm(ux) for ux in g])
			indices = dists.argsort()[:D]
			closest.append(newX[indices])
		return closest 

	# img_array should be a 2D (square) numpy array.
	# Note, you are welcome to change this function (including its arguments and return values) to suit your needs. 
	# However, we do ask that any images in your writeup be grayscale images, just as in this example.
	def create_image_from_array(self, img_array):
		plt.figure()
		plt.imshow(img_array, cmap='Greys_r')
		plt.show()
		return

# This line loads the images for you. Don't change it! 
pics = np.load("images.npy", allow_pickle=False)

# You are welcome to change anything below this line. This is just an example of how your code may look.
# That being said, keep in mind that you should not change the constructor for the KMeans class, 
# though you may add more public methods for things like the visualization if you want.
# Also, you must cluster all of the images in the provided dataset, so your code should be fast enough to do that.


KMeansClassifier = KMeans(K=5)
KMeansClassifier.fit(pics)
closest_images = KMeansClassifier.get_representative_images(1)
mean_images = KMeansClassifier.get_mean_images()
# KMeansClassifier.create_image_from_array(KMeansClassifier.get_representative_images(2)[0][0])
# KMeansClassifier.create_image_from_array(KMeansClassifier.get_representative_images(2)[0][1])
# KMeansClassifier.create_image_from_array(KMeansClassifier.get_mean_images()[0])
# plt.figure()
# plt.plot(range(len(KMeansClassifier.objective)),KMeansClassifier.objective)
# plt.show()

fig = plt.figure(figsize = (16,10))
plt.subplot(151)
plt.imshow(closest_images[0][0], cmap='Greys_r')
plt.subplot(152)
plt.imshow(closest_images[1][0], cmap='Greys_r')
plt.subplot(153)
plt.imshow(closest_images[2][0], cmap='Greys_r')
plt.subplot(154)
plt.imshow(closest_images[3][0], cmap='Greys_r')
plt.subplot(155)
plt.imshow(closest_images[4][0], cmap='Greys_r')
plt.tight_layout()
plt.savefig('representative_images2_k5.jpg')
plt.show()


fig = plt.figure(figsize = (16,10))
plt.subplot(151)
plt.imshow(mean_images[0], cmap='Greys_r')
plt.subplot(152)
plt.imshow(mean_images[1], cmap='Greys_r')
plt.subplot(153)
plt.imshow(mean_images[2], cmap='Greys_r')
plt.subplot(154)
plt.imshow(mean_images[3], cmap='Greys_r')
plt.subplot(155)
plt.imshow(mean_images[4], cmap='Greys_r')
plt.tight_layout()
plt.savefig('mean_images2_k5.jpg')
plt.show()


