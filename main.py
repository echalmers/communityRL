from network import ActionPenaltyNetwork, TaskRewardNetwork, QLearningWrapper
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import pandas as pd
from multiprocessing import Pool


data = {
    (0, 0, 0, 0): [0, 0],
    (0, 0, 0, 1): [0, 1],
    (0, 0, 1, 0): [0, 1],
    (0, 1, 0, 0): [0, 1],
    (1, 0, 0, 0): [0, 1],
    (1, 1, 0, 0): [1, 0],
    (0, 1, 1, 0): [1, 0],
    (0, 0, 1, 1): [1, 0],
    (1, 0, 0, 1): [1, 0],
    (0, 1, 0, 1): [1, 0],
    (1, 0, 1, 0): [1, 0],
    (1, 1, 1, 0): [1, 1],
    (0, 1, 1, 1): [1, 1],
    (1, 0, 1, 1): [1, 1],
    (1, 1, 0, 1): [1, 1],
}

# data = {
#     (0, ): [1],
#     (1, ): [0],
# }

def run_experiment(net):
    print(f'running with {net.__class__.__name__}...')

    results = []
    steps = 10000
    e_schedule = np.linspace(0.5, 0, steps) ** 2

    input = random.choice(list(data))
    for step in range(steps):
        # if step % 5 == 0:
        input = random.choice(list(data))
        net.set_epsilon(e_schedule[step])

        output = net.act(input)
        if all(np.array(output) == np.array(data[input])):
            net.get_reward(100)
            task_success = 1
        else:
            net.get_reward(0)
            task_success = 0

        results.append({
            'net': net.__class__.__name__,
            'step': step,
            # 'input': input,
            # 'hidden': net.hidden_state,
            # 'output': output,
            'success': task_success
        })
    results = pd.DataFrame(results)
    results['success'] = results['success'].rolling(100).mean()
    return results


if __name__ == '__main__':

    nets = []
    nets += [ActionPenaltyNetwork(n_hidden=2, n_outputs=len(list(data.values())[0])) for _ in range(8)]
    nets += [TaskRewardNetwork(n_hidden=2, n_outputs=len(list(data.values())[0])) for _ in range(8)]
    nets += [QLearningWrapper(action_list=[tuple(v) for v in data.values()]) for _ in range(8)]

    pool = Pool(12)
    df = map(run_experiment, nets)
    pool.close()

    df = pd.concat(df, ignore_index=True)

    print('plotting...')
    sns.lineplot(df.iloc[::10, :], x='step', y='success', hue='net', errorbar='ci', n_boot=1)
    plt.show()

# f = plt.subplot(5, 1, 1)
# plt.plot(pd.Series(task_success).rolling(50).mean())
#
# plt.subplot(5, 1, 2, sharex=f)
# plt.plot(h_energy)
# zeros = np.arange(0,len(h_energy))[np.array(h_energy) == 0]
# plt.scatter(zeros, np.zeros(len(zeros)), c='red', marker='x')
# plt.grid()
#
# plt.subplot(5, 1, 3, sharex=f)
# lines = list(zip(*h_input))
# for i in range(len(lines)):
#     plt.plot(np.array(lines[i]) + i)
#
# plt.subplot(5, 1, 4, sharex=f)
# lines = list(zip(*h_hidden))
# for i in range(len(lines)):
#     plt.plot(np.array(lines[i]) + i)
#
# plt.subplot(5, 1, 5, sharex=f)
# lines = list(zip(*h_output))
# for i in range(len(lines)):
#     plt.plot(np.array(lines[i]) + i)
#
# plt.show()