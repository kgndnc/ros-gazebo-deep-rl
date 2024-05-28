import random
import pickle
import math


class QLearning:
    def __init__(self, actions, alpha=0.1, gamma=0.9):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = self.load_q_table()

    def update_q_table(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in self.actions}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0 for a in self.actions}

        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())
        new_q = ((1.0 - self.alpha) * current_q) + \
            (self.alpha * (reward + (self.gamma * max_next_q)))
        self.q_table[state][action] = new_q
        print(f"Updated Q-table")

    def choose_action(self, state, epsilon=0.1):
        if random.uniform(0, 1) < epsilon or state not in self.q_table:
            print("choosing random action")
            return random.choice(self.actions)

        print(
            f"Choosing from Q-Table. {max(self.q_table[state], key=self.q_table[state].get)}")
        return max(self.q_table[state], key=self.q_table[state].get)

    def save_q_table(self):
        with open('Q-Table.pkl', 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self):
        read_dictionary = {}

        try:
            with open('Q-Table.pkl', 'rb') as f:
                read_dictionary = pickle.load(f)
            print("Loaded Q-Table")
        except:
            read_dictionary = {}
            print("Q-Table not found")

        return read_dictionary
