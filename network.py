from learners import TabularLearner

import numpy as np


class Network:

    def __init__(self, n_hidden, n_outputs, starting_energy):
        self.hidden_layer = [TabularLearner([0, 1], default_value=100, alpha=0.1, gamma=0.95, epsilon=0.1)
                             for _ in range(n_hidden)]
        self.output_layer = [TabularLearner([0, 1], default_value=100, alpha=0.1, gamma=0.95, epsilon=0.1)
                             for _ in range(n_outputs)]
        self.n_neurons = n_hidden + n_outputs

        self.input_state = None
        self.hidden_state = None
        self.output_state = None

        self.energy_pool = starting_energy

    def add_energy(self, amount):
        self.energy_pool += amount

    def set_epsilon(self, e):
        for agent in self.hidden_layer:
            agent.epsilon = e
        for agent in self.output_layer:
            agent.epsilon = e

    @property
    def energy_state(self):
        if self.energy_pool <= 0:
            return 0
        elif self.energy_pool < 1.5 * self.n_neurons:
            return 1
        return 2

    def augment_reward(self, reward, agent):
        if self.energy_state == 0:
            return reward - 100
        return reward

    def act(self, input):

        # forward pass
        energy_state = self.energy_state
        new_input_state = np.append(input, energy_state)

        new_hidden_state = np.append(
            np.array([agent.select_action(new_input_state) for agent in self.hidden_layer]),
            energy_state
        )

        new_output_state = np.array([agent.select_action(new_hidden_state) for agent in self.output_layer])

        # energy cost
        self.energy_pool = max(0, self.energy_pool - (new_hidden_state.sum() + new_output_state.sum()))

        # learning process
        if self.input_state is not None:
            for i in range(len(self.hidden_layer)):
                self.hidden_layer[i].update(state=self.input_state, action=self.hidden_state[i], reward=self.augment_reward(self.hidden_state[i], self.hidden_layer[i]), new_state=new_input_state)
            for i in range(len(self.output_layer)):
                self.output_layer[i].update(state=self.hidden_state, action=self.output_state[i], reward=self.augment_reward(self.output_state[i], self.output_layer[i]), new_state=new_hidden_state)

        # reset for next call
        self.input_state = new_input_state
        self.hidden_state = new_hidden_state
        self.output_state = new_output_state

        return self.output_state


class TaskRewardNetwork(Network):

    reward = 0

    def add_energy(self, amount):
        self.reward = amount
        super().add_energy(amount)

    def augment_reward(self, reward, agent):
        return self.reward
