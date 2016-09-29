import numpy as np

class ridge:
	'''
		An elementary class for performing ridge regression. Extensibility to other forms of regression is probably not too difficult. 
	'''

	beta = None

	def train(self, X = None, Y = None, gamma = 0):
		to_invert = np.dot(X.T,X) + gamma * np.identity(X.shape[1])
		inverted = np.linalg.inv(to_invert)
		estimator = np.dot(inverted, X.T)
		self.beta = np.dot(estimator, Y)
		
	def predict(self, X = None):
		return np.dot(X, self.beta)
		