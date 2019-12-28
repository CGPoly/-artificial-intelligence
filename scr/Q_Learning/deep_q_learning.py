import numpy as np
import deep_neural_network as dnn
import random


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return samples


class Dqn:
    def __init__(self, nodes, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = dnn.NeuralNetwork(nodes=nodes, learning_rate=0.001, activation_func="sigmoid")
        self.memory = ReplayMemory(100000)
        self.last_state = np.zeros(nodes[0])
        self.last_action = 0
        self.last_reward = 0
        
    @staticmethod
    def __softmax(x):
        return np.asarray([float(i) / sum(x) for i in x])
        # return [np.exp(i)/sum([np.exp(j) for j in x]) for i in x]

    @staticmethod
    def __pick_one(probability):
        pro = [float(i) / sum(probability) for i in probability]
        index = 0
        r = np.random.rand()
        while r > 0:
            r = r - pro[index]
            index += 1
            pass
        index -= 1
        return index
    
    def save(self):
        pass
    
    def select_action(self, state):
        temperature = 100  # höhere Zahlen erhöhen die Wahrscheinlichkeit des, von der KI, gewählten Wertes (geben der KI mehr mach)
        print(self.model.query(state)[:, 0] * temperature)
        return self.__pick_one(self.__softmax(self.model.query(state)[:, 0] * temperature))
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        for i in range(len(batch_state)):
            next_output = self.model.query(batch_next_state[i])  # .detach().max(1)[0]
            target = (self.gamma * next_output + batch_reward[i])[:, 0]
            # target_list = np.zeros(self.model.nodes[-1]) + 0.01
            # target_list[target] = 0.99
            self.model.train(batch_state[i], target)
    
    def update(self, reward, new_signal):
        new_state = new_signal
        self.memory.push((self.last_state, new_state, self.last_action, self.last_reward))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def query(self, input_list):
        return self.model.query(input_list)
    
    def score(self):
        return sum(self.reward_window) / (len(self.reward_window) + 1.)
