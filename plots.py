import matplotlib.pyplot as plt
import numpy as np


def train_plot(input, output, neurons):
    plt.plot(output[0], label='Класс A')
    plt.plot([i for i in range(neurons, 2 * neurons)], output[1], label='Класс B')
    plt.plot([i for i in range(2 * neurons, 3 * neurons)], output[2], label='Класс C')
    plt.plot(input[0], label='Образ 1', linestyle='--')
    plt.plot([i for i in range(neurons, 2 * neurons)], input[1], label='Образ 2', linestyle='--')
    plt.plot([i for i in range(2 * neurons, 3 * neurons)], input[2], label='Образ 3', linestyle='--')
    plt.xticks(np.linspace(0, neurons * 3, 1))
    plt.axis([0, neurons * 3 + 20, -2, 2])
    plt.xlabel("Time")
    plt.ylabel("F(t)")
    plt.legend()
    plt.show()


def test_plot(test_target, output, neurons, cluster_actual, *args):
    plt.plot(output, label=f'Actual Class({cluster_actual})')
    plt.plot(test_target, label='Corrupted', linestyle='--')
    plt.xticks(np.linspace(0, neurons, 1))
    plt.axis([0, neurons + 10, -2, 2])
    plt.xlabel("Time")
    plt.ylabel("F(t)")
    plt.legend()
    plt.show()
    plt.close()

    if len(args) != 0:
        plt.plot(args[0], label=f'Expected Class({args[1]})')
        plt.plot(test_target, label='Corrupted', linestyle='--')
        plt.xticks(np.linspace(0, neurons, 1))
        plt.axis([0, neurons + 10, -2, 2])
        plt.xlabel("Time")
        plt.ylabel("F(t)")
        plt.legend()
        plt.show()
