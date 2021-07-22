import numpy as np

dvalues = np.array([[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]])

weights = np.array([[0.2, 0.8, - 0.5, 1], [0.5, - 0.91, 0.26, - 0.5], [- 0.26, - 0.27, 0.17, 0.87]]).T

# sum weights of given input
# and multiply by the passed in gradient for this neuron


dinputs = np.dot(dvalues[0], weights.T)
print (dinputs)




