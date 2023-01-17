import numpy as np
import neurolab as nl
from random import randint
from dataCreator import create_data, create_sheet
from plots import train_plot, test_plot

n_fragments = 3
neurons = 26
dataset = create_sheet(create_data())["x_0"][100:]
dynamics = []
classes = ['A', 'B', 'C']
# Dynamic building
for i in range(dataset.size - 1):
    if dataset.iloc[i] >= dataset.iloc[i + 1]:
        dynamics.append(1)
    else:
        dynamics.append(-1)

target = [[] for i in range(n_fragments)]
# Match dynamic values by classes
count = 0
for value in dynamics:
    match count // neurons:
        case 0:
            target[0].append(value)
        case 1:
            target[1].append(value)
        case 2:
            target[2].append(value)
    count += 1

# Create and train network
target = np.asfarray(target)
net = nl.net.newhop(target)
output = net.sim(target)
print("Test on train samples:")
for i in range(n_fragments):
    print('Match with class', classes[i], '-', (output[i] == target[i]).all())
train_plot(target, output, neurons)

test_cluster = 2
test_target = [i for i in target[test_cluster]]
# Corruption
corrupt = randint(1, neurons - 1)
for i in range(corrupt):
    x = randint(0, neurons - 1)
    test_target[x] *= -1.0

print(corrupt, "Values was corrupted")

test_output = net.sim([test_target])


# Testing
def test():
    for i in range(n_fragments):
        if (test_output == output[i]).all():
            print('Test match with class', classes[i], ', correct one is', classes[test_cluster])
            if classes[i] != classes[test_cluster]:
                test_plot(test_target, output[i], neurons, classes[i], output[test_cluster], classes[test_cluster])
            else:
                test_plot(test_target, output[i], neurons, classes[i])
            return ''
    print('Test no have matches!')


test()
