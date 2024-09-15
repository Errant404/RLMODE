import numpy as np

# SoftMax函数
def softmax(Q_values, T):
    exp_values = np.exp(Q_values / T)
    return exp_values / np.sum(exp_values)


class Qlearning:
    def __init__(self,
                 n_states,
                 n_actions,
                 alpha=0.1,
                 gamma=0.99,
                 q_init=None,
                 **kwargs):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = np.zeros((n_states, n_actions)) if q_init is None else q_init
        self.discount_factor = 0.9
        self.learning_rate = 0.1
        self.T = 0.9

    # 选择动作
    def choose_action(self, state):
        probabilities = softmax(self.q_table[state], self.T)
        return np.random.choice(self.n_actions, p=probabilities)

    # 更新Q表
    def update_q_table(self, current_state, next_state, reward):
        action = self.choose_action(current_state)
        self.q_table[current_state, action] += self.learning_rate * (reward
                                                           + self.discount_factor * np.max(self.q_table[next_state])
                                                           - self.q_table[current_state, action])
