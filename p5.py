from email import contentmanager
from email import contentmanager
import numpy as np

from datasets import spiral_data

X, y = spiral_data(100, 3)

np.random.seed(0)

# X = [[1,2,3,2.5],
#      [2,3,1,0.5],
#      [-1.5,2.7,3.3,-0.8]]

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases
    
class Activation_ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

class Activation_Softmax:
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = norm_values

class Loss:
    def calculate(self,output,y):
        sample_losses = self.forward(output,y)
        return np.mean(sample_losses)

class Loss_CategoricalCrossentropy(Loss):
    def forward(self,softmax_output,target_output):
        num_samples = len(softmax_output)
        # clip the values to prevent log(0) to avoid infinite value error 
        softmax_output = np.clip(softmax_output, 1e-7, 1 - 1e-7)
        
        if len(target_output.shape) == 1:
            correct_confidences = softmax_output[range(num_samples), target_output]

        elif len(target_output.shape) == 2:
            correct_confidences = np.sum(softmax_output*target_output,axis=1)
    
        log_likelihood = -np.log(correct_confidences)
        return log_likelihood
        

layer1 = Layer_Dense(2,5)
activation1 = Activation_ReLU()

layer1.forward(X)
activation1.forward(layer1.output)

layer2 = Layer_Dense(5,3)
activation2 = Activation_Softmax()

layer2.forward(activation1.output)
activation2.forward(layer2.output)

print(activation2.output)

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output,y)
print('Categorical Cross Entropy Loss is :', loss)