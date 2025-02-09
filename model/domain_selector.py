import numpy as np
import random


class EpsilonGreedyStateSelector:
    def __init__(self, num_states, initial_epsilon=1.0, min_epsilon=0.1, decay_rate=0.9):
        self.num_states = num_states
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.value_fn = [[None for _ in range(num_states)] for i in range(num_states)]  # 状态价值函数表
        self.state_list = [[[] for _ in range(num_states)] for i in range(num_states)]

    def update_value_function(self, states, val_loss):
        state_num_list = [len(states[i]) - 1 for i in range(self.num_states)]
        for i, state_num in enumerate(state_num_list):
            if self.value_fn[i][state_num] is None or val_loss[i] > self.value_fn[i][state_num]:
                self.value_fn[i][state_num] = val_loss[i]
                self.state_list[i][state_num] = states[i]

    def get_next_state(self, model):
        next_state_num = [None] * self.num_states
        next_state = [None] * self.num_states
        flag = [False] * self.num_states
        for i in range(self.num_states):
            if  random.random() < self.epsilon:
                next_state_num[i] = random.randint(0, self.num_states - 1)
                next_state[i] = model.domain_distance(model.proto_emb, next_state_num[i], i)
            else:
                next_state_num[i] = np.nanargmax([v if v is not None else float('-inf') for v in self.value_fn[i]])
                next_state[i] = self.state_list[i][next_state_num[i]]
                
        return next_state

    def decay_temperature(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)

