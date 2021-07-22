import numpy as np
from nnfs.datasets import spiral_data
from nnfs.datasets import vertical_data
import matplotlib.pyplot as plt

np.random.seed(0)

class Layer:

    def __init__(self, weights_count=0, node_count=0):

        self.weights_count = weights_count
        self.nodes_count = node_count
        self.weights = np.random.rand(self.weights_count, self.nodes_count)
        self.biases = np.zeros((1, self.nodes_count))
        self.output = []

        #print(self.weights)
        #print(self.biases)

    def forward(self, input):

        self.output = np.dot(input, self.weights) + self.biases


        return self.output


    def relu(self, input):
        #print(list(input))
        return np.maximum(0, input)

    def softMax(self, input):
        ext_values = np.exp(input - np.max(input, axis=1, keepdims=True))
        normalized_exp = ext_values/np.sum(ext_values, axis=1, keepdims=True)
        return normalized_exp





class Loss:

    #def calculate(self):



    def calculate_loss_crossEntropy(self, y_predict, y_target):

        cliped_y_predict = np.clip(y_predict, 1e-7, 1 - 1e-7)

        if len(y_target.shape)==1:
            output = cliped_y_predict[range(len(cliped_y_predict)), y_target]
        elif len(y_target.shape)==2:
            output = np.sum(cliped_y_predict * y_target, axis=1)



        neg_log_output = -np.log(output)
        return np.mean(neg_log_output)






def backward ( self , dvalues ):
    # Create uninitialized array
    self.dinputs = np.empty_like(dvalues)
    # Enumerate outputs and gradients
    for index, (single_output, single_dvalues) in \
            enumerate ( zip (self.output, dvalues)):
        # Flatten output array
        single_output = single_output.reshape( - 1 , 1 )
        # Calculate Jacobian matrix of the output and
        jacobian_matrix = np.diagflat(single_output) - \
        np.dot(single_output, single_output.T)
        # Calculate sample-wise gradient
        # and add it to the array of sample gradients
        self.dinputs[index] = np.dot(jacobian_matrix,
        single_dvalues)








x, y = spiral_data(100, 3)


layer1 = Layer(2, 3)
o1 = layer1.forward(x)
o1 = layer1.relu(o1)
#print(o1)
l2 = Layer(3, 3)
ab = l2.forward(o1)
ab = l2.softMax(ab)
#print(ab)

input = np.array([[1, 2, 3], [-2, -1, 0]])
target = np.array([[1, 0, 0], [1, 1, 0]])
softmax_output = layer1.softMax(ab)

loss = Loss()
lc = loss.calculate_loss_crossEntropy(softmax_output, y)
print(lc)


X, y = vertical_data( samples = 100 , classes = 3 )
plt.scatter(X[:, 0 ], X[:, 1 ], c = y, s = 40 , cmap = 'brg' )
plt.show()











