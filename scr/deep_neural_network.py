import scipy.special as science_special
import numpy as np


class NeuralNetwork:
    def __init__(self, nodes: [list, np.ndarray], learning_rate: float, use_bias: bool = False, activation_func="sigmoid"):
        self.nodes = nodes
        self.use_bias = use_bias

        if use_bias:
            for i in range(len(self.nodes) - 1):
                self.nodes[i] += 1
                pass
            pass

        self.lr = learning_rate

        self.weight = []
        for i in range(len(self.nodes) - 1):
            self.weight.append(np.random.normal(0.0, pow(self.nodes[i], -0.5), (self.nodes[i+1], self.nodes[i])))
            pass

        self.activation_function = []

        if isinstance(activation_func, list):
            for i in range(len(self.nodes) - 1):
                self.activation_function.append(self.__give_activation_function(activation_func[i]))
                pass
            pass
        elif isinstance(activation_func, str):
            for i in range(len(self.nodes) - 1):
                self.activation_function.append(self.__give_activation_function(activation_func))
                pass
            pass
        elif callable(activation_func):
            for i in range(len(self.nodes) - 1):
                self.activation_function.append([activation_func])
                pass
            pass
        else:
            raise Exception('activation_func is no accepted type')
            pass
        pass

    def set_weight(self, weight: list):
        self.weight = weight

    def train(self, inputs_list, target_lists):
        if self.use_bias:
            inputs_list = np.append(inputs_list, 1)
            pass
        target = np.array(target_lists, ndmin=2).T
        
        layer = [np.array(inputs_list, ndmin=2).T]
        for i in range(len(self.nodes) - 1):
            layer.append(self.activation_function[i][0](np.dot(self.weight[i], layer[i])))
            pass
        
        error = [target - layer[- 1]]
        # error = [target - layer[len(layer) - 1]]
        for i in range(len(self.weight)-1, 0, -1):
            error.append(np.dot(self.weight[i].T, error[len(error)-1]))
            pass

        for i in range(len(self.weight)):
            self.weight[i] += self.lr * np.dot(error[len(error) - i - 1] * layer[i + 1] * (1.0 - layer[i + 1]), np.transpose(layer[i]))
            pass
        pass

    def query(self, inputs_list):
        if self.use_bias:
            inputs_list = np.append(inputs_list, 1)
            pass
        layer = np.array(inputs_list, ndmin=2).T
        for i in range(len(self.nodes) - 1):
            layer = self.activation_function[i][0](np.dot(self.weight[i], layer))
        return layer

    def back_query(self, label: int) -> np.ndarray:
        targets_list = np.zeros(self.nodes[len(self.nodes) - 1]) + 0.01
        targets_list[label] = 0.99
        # transpose the targets list to a vertical array
        final_outputs = np.array(targets_list, ndmin=2).T

        # calculate the signal into the final output layer
        final_inputs = self.activation_function[len(self.activation_function)-1][1](final_outputs)

        layer = final_inputs
        for i in range(len(self.weight) - 1, -1, -1):
            layer = np.dot(self.weight[i].T, layer)
            layer -= np.min(layer)
            layer /= np.max(layer)
            layer *= 0.98
            layer += 0.01
            if not i == 0:
                layer = self.activation_function[i][1](layer)
        if self.use_bias:
            layer = np.delete(layer, self.nodes[0]-1)
            pass
        return layer

    @staticmethod
    def __give_activation_function(activation_func: str):
        if activation_func == "sigmoid":
            activation_functions = [lambda x: science_special.expit(x), lambda x: science_special.logit(x)]
            pass
        elif activation_func == "ReLu":
            def relu(x):
                try:
                    return np.asarray([[max([0, j]) for j in i] for i in x])
                except TypeError:
                    return x if x > 0 else 0
            activation_functions = [lambda x: relu(x), lambda x: x]
        elif activation_func == "tanh":
            activation_functions = [lambda x: np.tanh(x), lambda x: np.arctanh(x)]
            pass
        elif activation_func == "gaussian":
            activation_functions = [lambda x: np.exp(-x**2), lambda x: x]
            pass
        elif activation_func == "sin":
            activation_functions = [lambda x: np.sin(x), lambda x: np.tan(x)]
            pass
        elif activation_func == "atan":
            activation_functions = [lambda x: np.arctan(x), lambda x: np.arcsin(x)]
            pass
        elif activation_func == "":
            activation_functions = [lambda x: x, lambda x: -x]
            pass
        else:
            raise Exception("No known activation function")
        return activation_functions
    pass
