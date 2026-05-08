import numpy as np
inputs = [1,2,3,2.5]
weights = [[3.1,2.0,-0.5,1],
           [0.5,-0.91,0.26,-0.5],
           [-0.26,2.6,1.42,-0.12]]

biases = [2,3,0.5]

'''
layer_outputs = []
for neuron_weight, neuron_bias in zip(weights, biases):
    neuron_output = 0
    for input, weight in zip(inputs, neuron_weight):
        neuron_output += input * weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)
'''

output = np.dot(weights,inputs) + biases
print(output)




