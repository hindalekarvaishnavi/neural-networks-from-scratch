import math

layer_outputs = [4.8,1.2,-0.7,2.1]

E = math.e

# find exponents in neural networks remove the negative numbers to prevent overflow
exp_values = []

# exp_values = np.exp(layer_outputs)

for output in layer_outputs:
    exp_values.append(E ** output)

print(exp_values)

# normalize the values to make them probabilities
norm_base = sum(exp_values)

norm_values = []

# norm_values = np.exp(layer_outputs) / np.sum(np.exp(layer_outputs))

for value in exp_values:
    norm_values.append(value/norm_base)

print(norm_values)