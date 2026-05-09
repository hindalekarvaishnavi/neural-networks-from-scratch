import math

layer_outputs = [4.8,1.2,-0.7,2.1]

E = math.e

exp_values = []

for output in layer_outputs:
    exp_values.append(E ** output)

print(exp_values)

norm_base = sum(exp_values)

norm_values = []

for value in exp_values:
    norm_values.append(value/norm_base)

print(norm_values)