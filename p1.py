import numpy as np

inputs = [[1,2,3,2.5],
          [2,3,1,0.5],
          [1,1,1,1]]

weights1 = [[3.1,2.0,-0.5,1],
           [0.5,-0.91,0.26,-0.5],
           [-0.26,2.6,1.42,-0.12]]

biases1 = [2,3,0.5]

weights2 = [[0.1,-0.14,0.3],
           [0.3,-0.7,1.6],
           [-1.2,1.5,0]]

biases2 = [-1,2,-0.5]

layer1_output = np.dot(inputs,np.array(weights1).T) +biases1
print(layer1_output)

layer2_output = np.dot(layer1_output,np.array(weights2).T) +biases2
print(layer2_output)
