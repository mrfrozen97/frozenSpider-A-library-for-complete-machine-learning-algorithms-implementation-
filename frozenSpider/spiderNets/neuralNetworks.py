import numpy as np
from nnfs.datasets import spiral_data

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


        return self.relu(self.output)


    def relu(self, input):
        #print(list(input))
        return np.maximum(0, input)

    def softMax(self, input):
        ext_values = np.exp(input - np.max(input, axis=1, keepdims=True))
        normalized_exp = ext_values/np.sum(ext_values, axis=1, keepdims=True)
        return normalized_exp





class Loss:

    #def calculate(self):



    def calculate_loss(self, y_predict, y_target):

        cliped_y_predict = np.clip(y_predict, 1e-7, 1 - 1e-7)

        if len(y_target.shape)==1:
            output = cliped_y_predict[range(len(cliped_y_predict)), y_target]
        elif len(y_target.shape)==2:
            output = np.sum(cliped_y_predict * y_target, axis=1)



        neg_log_output = -np.log(output)
        return np.mean(neg_log_output)















x, y = spiral_data(100, 3)


layer1 = Layer(2, 3)
o1 = layer1.forward(x)
#print(o1)
l2 = Layer(3, 3)
ab = l2.forward(o1)
#print(ab)

input = np.array([[1, 2, 3], [-2, -1, 0]])
target = np.array([[1, 0, 0], [1, 1, 0]])
softmax_output = layer1.softMax(ab)

loss = Loss()
lc = loss.calculate_loss(softmax_output, y)
print(lc)




softmax_outputs = np.array([[ 0.7 , 0.1 , 0.2 ],
[ 0.1 , 0.5 , 0.4 ],
[ 0.02 , 0.9 , 0.08 ]])

class_targets = [ 0 , 1 , 1 ]
print (softmax_outputs[[ 0 , 1 , 2 ], class_targets])











