# CS 181, Harvard University
# Spring 2016
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as c
import time
from Perceptron import Perceptron

# Implement this class
class KernelPerceptron(Perceptron):
	def __init__(self, numsamples):
		self.numsamples = numsamples

	# Implement this!
	def fit(self, X, Y):
		#learn S
		S =[] #list of indices ; should be a dictionary?
		alphas =[]
		X = np.array(X)

		#store the number of examples
		n = len(Y)
		numsamples = self.numsamples

		for k in xrange(numsamples):
			
			# pick a random example
			i = np.random.randint(0,n)
			y_t = Y[i]
			x_t = X[i,:]

			#compute y(x_t)
			b = 0 
			for j in range(len(S)):
				ind = S[j]
				al = alphas[j]
				x_j = X[ind,:] 
				kern = al* np.dot(x_t,x_j)
				b+= kern
			if y_t*b <= 0:
				S.append(i)
				alphas.append(y_t)
		
		self.S = S
		self.alphas = alphas
		self.X_train = X
		self.X = X
		self.Y = Y

	# Implement this!
	def predict(self, X):
		X= np.array(X)

		S = self.S
		alphas = self.alphas
		X_train = self.X_train

		predict = []

		for x_i in X:
			b= 0 
			for j in range(len(S)):
				ind = S[j]
				al = alphas[j]
				x_j = X_train[ind,:]
				kern = al*np.dot(x_i,x_j)
				b+= kern

			if b>=0:
				predict.append(1)
			if b<0:
				predict.append(-1)

		return np.array(predict)


# Implement this class
class BudgetKernelPerceptron(Perceptron):
	def __init__(self, beta, N, numsamples):
		self.beta = beta
		self.N = N
		self.numsamples = numsamples
		
	# Implement this!
	def fit(self, X, Y):
		#learn S
		S =[] 
		alphas =[]
		X = np.array(X)
		beta = self.beta
		N = self.N 

		#store the number of examples
		n = len(Y)
		numsamples = self.numsamples

		for k in xrange(numsamples):

			# pick a random example
			i = np.random.randint(0,n)
			y_t = Y[i]
			x_t = X[i,:]

			#compute y(x_t)
			b = 0 
			for j in range(len(S)):
				ind = S[j]
				al = alphas[j]
				x_j = X[ind,:] 
				kern = al* np.dot(x_t,x_j)
				b+= kern

			if y_t*b <= beta:
				S.append(i)
				alphas.append(y_t)

			if len(S) > N:

				to_max = []
				for p in range(len(S)):
					ind = S[p]
					al = alphas[p]
					x_j = X[ind,:]
					y_j = Y[ind]
					K = np.dot(x_j,x_j)
					
					b = 0
					for v in range(len(S)):
						ind_ = S[v]
						al_ = alphas[v]
						x_v = X[ind_,:]
						kern_ = al_*np.dot(x_j,x_v)
						b+= kern_
					to_max.append(y_j*(b - al * K))

				to_max = np.array(to_max) 
				arg_max = np.argmax(to_max)
				alphas.pop(arg_max)
				S.pop(arg_max)
	
		self.S = S
		self.alphas = alphas
		self.X_train = X
		self.X = X
		self.Y = Y	

	# Implement this!
	def predict(self, X):
		X= np.array(X)

		S = self.S
		alphas = self.alphas
		X_train = self.X_train

		predict = []

		for x_i in X:
			b= 0 
			for j in range(len(S)):
				ind = S[j]
				al = alphas[j]
				x_j = X_train[ind,:]
				kern = al*np.dot(x_i,x_j)
				b+= kern

			if b>=0:
				predict.append(1)
			if b<0:
				predict.append(-1)

		return np.array(predict)



# Do not change these three lines.
data = np.loadtxt("data.csv", delimiter=',')
X = data[:, :2]
Y = data[:, 2]

# These are the parameters for the models. Please play with these and note your observations about speed and successful hyperplane formation.
beta = 0.0001
N = 30
numsamples = 20000

kernel_file_name = 'k.png'
budget_kernel_file_name = 'bk.png'

# Don't change things below this in your final version. Note that you can use the parameters above to generate multiple graphs if you want to include them in your writeup.
k = KernelPerceptron(numsamples)
k_start = time.time()
k.fit(X,Y)
k_end = time.time()
#k.visualize(kernel_file_name, width=0, show_charts=True, save_fig=True, include_points=False)

bk = BudgetKernelPerceptron(beta, N, numsamples)
bk_start = time.time()
bk.fit(X, Y)
bk_end = time.time()
#bk.visualize(budget_kernel_file_name, width=0, show_charts=True, save_fig=True, include_points=False)


# # training time
bk_time = bk_end - bk_start
k_time = k_end - k_start
print "Beta: ", beta
print "N: ", N
print "Number of Random Samples: ", numsamples
print "Budget Kernel Perceptron Training Time: ", bk_time
print "Kernel Perceptron Training Time: ", k_time

#misclassification accuracy

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
svm = SVC(C = 1000000, kernel = 'linear')
svm_start = time.time()
svm.fit(X,Y)
svm_end = time.time()
val = np.loadtxt("val.csv", delimiter=',')
X_val = val[:, :2]
Y_val = val[:, 2]

yk_hat = k.predict(X_val)
ybk_hat = bk.predict(X_val)
svm_hat = svm.predict(X_val)

print "Kernel Perceptron training accuracy: ", accuracy_score(Y,k.predict(X))
print "Budget Kernel perceptron training acc: ", accuracy_score(Y,bk.predict(X))
print "Kernel Perceptron validation accuracy: ", accuracy_score(Y_val,yk_hat)
print "BK validation accuracy: ", accuracy_score(Y_val,ybk_hat)
print "Standard SVM validation accuracy: ", accuracy_score(Y_val,svm_hat)
print "Standard SVM, time", svm_end- svm_start
#number of support vectors
print "BK, Number of SV: ", len(bk.S)
print "K, Number of SV: ", len(k.S)

#number of random samples





