import numpy as np

np.random.seed(0)

X = [[1,2,3,2.5],
     [2,3,1,0.5],
     [1,1,1,1]]

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases
    
layer1 = Layer_Dense(4,3)
layer2 = Layer_Dense(3,2)

layer1.forward(X)
layer2.forward(layer1.output)
print(layer2.output)
