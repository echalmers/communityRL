from abc import ABC, abstractmethod

from learners import TabularLearner

import numpy as np


class Network:

    def __init__(self, n_hidden, n_outputs):
        self.hidden_layer = [TabularLearner([0, 1], default_value=100, alpha=0.1, gamma=0.95, epsilon=0.1)
                             for _ in range(n_hidden)]
        self.output_layer = [TabularLearner([0, 1], default_value=100, alpha=0.1, gamma=0.95, epsilon=0.1)
                             for _ in range(n_outputs)]
        self.n_neurons = n_hidden + n_outputs

        self.input_state = None
        self.hidden_state = None
        self.output_state = None

        self.reward = 0

    def get_reward(self, amount):
        self.reward = amount

    def set_epsilon(self, e):
        for agent in self.hidden_layer:
            agent.epsilon = e
        for agent in self.output_layer:
            agent.epsilon = e

    @abstractmethod
    def reward_function(self, task_reward, action):
        raise NotImplementedError()

    def act(self, input):

        # forward pass
        new_input_state = input

        new_hidden_state = np.array([agent.select_action(new_input_state) for agent in self.hidden_layer])

        new_output_state = np.array([agent.select_action(new_hidden_state) for agent in self.output_layer])

        # learning process
        if self.input_state is not None:
            for i in range(len(self.hidden_layer)):
                self.hidden_layer[i].update(state=self.input_state, action=self.hidden_state[i], reward=self.reward_function(self.reward, self.hidden_state[i]), new_state=new_input_state)
            for i in range(len(self.output_layer)):
                self.output_layer[i].update(state=self.hidden_state, action=self.output_state[i], reward=self.reward_function(self.reward, self.output_state[i]), new_state=new_hidden_state)

        # reset for next call
        self.input_state = new_input_state
        self.hidden_state = new_hidden_state
        self.output_state = new_output_state

        return self.output_state


class ActionPenaltyNetwork(Network):

    def reward_function(self, task_reward, action):
        return task_reward - action


class TaskRewardNetwork(Network):

    def reward_function(self, task_reward, action):
        return task_reward


class QLearningWrapper:

    def __init__(self, action_list):
        self.learner = TabularLearner(action_list=action_list, default_value=100, alpha=0.1, gamma=0.95, epsilon=0.1)
        self.reward = 0
        self.input_state = None
        self.action = None

    def act(self, input):

        if self.input_state is not None:
            self.learner.update(self.input_state, self.action, self.reward, input)

        self.action = self.learner.select_action(input)
        self.input_state = input

        return list(self.action)

    def get_reward(self, amount):
        self.reward = amount

    def set_epsilon(self, e):
        self.learner.epsilon = e
