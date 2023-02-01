from network import Network, TaskRewardNetwork
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd


data = {
    (0, 0, 0): [0, 0],
    (0, 0, 1): [0, 1],
    (0, 1, 0): [0, 1],
    (1, 0, 0): [0, 1],
    (0, 1, 1): [1, 0],
    (1, 1, 0): [1, 0],
    (1, 1, 1): [1, 1],
}

# data = {
#     (0, ): [1],
#     (1, ): [0],
# }

net = Network(n_hidden=4, n_outputs=len(list(data.values())[0]), starting_energy=100)
# net = TaskRewardNetwork(n_hidden=4, n_outputs=len(list(data.values())[0]), starting_energy=100)

h_input = []
h_hidden = []
h_output = []
h_energy = []
task_success = []

steps = 10000
e_schedule = np.linspace(0.5, 0.0, steps) ** 2

input = random.choice(list(data))
for step in range(steps):
    # if step % 5 == 0:
    input = random.choice(list(data))
    net.set_epsilon(e_schedule[step])

    output = net.act([input])
    if all(np.array(output) == np.array(data[input])):
        net.add_energy(net.n_neurons * 1.5)
        task_success.append(1)
    else:
        net.add_energy(0)
        task_success.append(0)

    h_input.append(input)
    h_hidden.append(net.hidden_state[:-1])
    h_output.append(output)
    h_energy.append(net.energy_pool)


f = plt.subplot(5, 1, 1)
plt.plot(pd.Series(task_success).rolling(50).mean())

plt.subplot(5, 1, 2, sharex=f)
plt.plot(h_energy)
zeros = np.arange(0,len(h_energy))[np.array(h_energy) == 0]
plt.scatter(zeros, np.zeros(len(zeros)), c='red', marker='x')
plt.grid()

plt.subplot(5, 1, 3, sharex=f)
lines = list(zip(*h_input))
for i in range(len(lines)):
    plt.plot(np.array(lines[i]) + i)

plt.subplot(5, 1, 4, sharex=f)
lines = list(zip(*h_hidden))
for i in range(len(lines)):
    plt.plot(np.array(lines[i]) + i)

plt.subplot(5, 1, 5, sharex=f)
lines = list(zip(*h_output))
for i in range(len(lines)):
    plt.plot(np.array(lines[i]) + i)

plt.show()