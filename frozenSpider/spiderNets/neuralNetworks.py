import numpy as np


class Neural_network:



    def __init__(self):

        self.layers = 2
        self.input_count = 0
        self.output_count = 0
        self.weights = []
        self.biases = []



class Layer:

    def __init__(self, node_count=0, ):

        self.nodes_count = node_count