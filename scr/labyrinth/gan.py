import numpy as np


class GAN:
    @staticmethod
    def __sigmoid(x):
        return 1 / (1 + np.e ** -x)

    def __init__(self, input_nodes, output_nodes, learning_rate):
        self.i_nodes = input_nodes
        self.o_nodes = output_nodes

        self.lr = learning_rate

        self.wio = np.random.normal(0.0, pow(self.o_nodes, -0.5), (self.o_nodes, self.i_nodes))

        self.activation_function = lambda x: GAN.__sigmoid(x)
        pass

    def set_weight(self, weight_io):
        self.wio = weight_io

    def train(self, inputs_list, loss):
        inputs = np.array(inputs_list, ndmin=2).T
        final = self.activation_function(np.dot(self.wio, inputs))

        # derivative_array = [np.mean(loss * inputs[:, i]) for i in range(len(inputs[0]))]
        # self.wio += self.lr * np.asarray([derivative_array[i] for i in range(len(derivative_array))])
        self.wio += self.lr * np.dot((loss * final * (1.0 - final)), np.transpose(inputs))
        pass

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        final = self.activation_function(np.dot(self.wio, inputs))
        return final
    
    
class DeepHGAN:
    @staticmethod
    def __sigmoid(x):
        return 1 / (1 + np.e ** -x)
    
    def __init__(self, nodes: [list, np.ndarray], learning_rate: float, use_bias: bool = False, activation_func="sigmoid"):
        self.nodes = np.asarray(nodes)
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
    
    @staticmethod
    def __difference_quotient(p1: tuple, p2: tuple):
        return (p2[1]-p1[1])/(p2[0]-p1[0])
    
    def train(self, loss_function, step=0.0001):
        inputs_list = np.random.rand(self.nodes[0])
        # if self.use_bias:
        #     inputs_list = np.append(inputs_list, 1)
        #     pass
        loss = loss_function(self.query(inputs_list))
        for i in range(len(self.weight)):
            for x in range(self.weight[i].shape[0]):
                for y in range(self.weight[i].shape[1]):
                    self.weight[i][x, y] += step
                    gradient = self.__difference_quotient((self.weight[i][x, y]-step, loss), (self.weight[i][x, y], loss_function(self.query(inputs_list))))
                    self.weight[i][x, y] += -step + (self.lr * gradient)
                    loss = loss_function(self.query(inputs_list))
                    pass
                print(x)
                pass
            pass
        return loss
        
        # layer = [np.array(inputs_list, ndmin=2).T]
        # for i in range(len(self.nodes) - 1):
        #     layer.append(self.activation_function[i](np.dot(self.weight[i], layer[i])))
        #     pass
        #
        # error = [loss]
        # for i in range(len(self.weight)-1, 0, -1):
        #     error.append(np.dot(self.weight[i].T, error[-1]))
        #     pass
        #
        # for i in range(len(self.weight)):
        #     # print(i)
        #     if i == 0:
        #
        #         pass
        #     else:
        #         self.weight[i] -= self.lr * np.dot(error[- i] * layer[i + 1] * (1.0 - layer[i + 1]), layer[i])  # np.transpose(layer[i]))
        #     pass
        # pass

    def query(self, inputs_list=None):
        if inputs_list is None:
            inputs_list = np.random.rand(self.nodes[0])
            pass
        # if self.use_bias:
        #     inputs_list = np.append(inputs_list, 1)
        #     pass
        layer = np.array(inputs_list, ndmin=2).T
        for i in range(len(self.nodes) - 1):
            layer = self.activation_function[i](np.dot(self.weight[i], layer))
        return layer

    @staticmethod
    def __give_activation_function(activation_func: str):
        if activation_func == "sigmoid":
            return lambda x: DeepHGAN.__sigmoid(x)
            pass
        if activation_func == "tanh":
            return lambda x: np.tanh(x)
            pass
        if activation_func == "gaussian":
            return lambda x: np.exp(-x**2)
            pass
        if activation_func == "sin":
            return lambda x: np.sin(x)
            pass
        if activation_func == "atan":
            return lambda x: np.arctan(x)
            pass
        if activation_func == "":
            return lambda x: x
            pass
        raise Exception("No known activation function")
    pass


class classicGAN:
    @staticmethod
    def __sigmoid(x):
        return 1 / (1 + np.e ** -x)
    
    def __init__(self, nodes: [list, np.ndarray], learning_rate: float, use_bias: bool = False, activation_func="sigmoid"):
        self.nodes = np.asarray(nodes)
        self.use_bias = use_bias
        
        if use_bias:
            for i in range(len(self.nodes) - 1):
                self.nodes[i] += 1
                pass
            pass
        
        self.lr = learning_rate
        
        self.weight = []
        for i in range(len(self.nodes) - 1):
            self.weight.append(np.random.normal(0.0, pow(self.nodes[i], -0.5), (self.nodes[i + 1], self.nodes[i])))
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
    
    @staticmethod
    def __difference_quotient(p1: tuple, p2: tuple):
        return (p2[1] - p1[1]) / (p2[0] - p1[0])
    
    def train(self, loss):
        inputs_list = np.random.rand(self.nodes[0])
        layer = [np.array(inputs_list, ndmin=2).T]
        for i in range(len(self.nodes) - 1):
            layer.append(self.activation_function[i](np.dot(self.weight[i], layer[i])))
            pass

        error = [loss]
        for i in range(len(self.weight)-1, 0, -1):
            error.append(np.dot(self.weight[i].T, error[-1]))
            pass

        for i in range(len(self.weight)):
            # print(i)
            if i == 0:

                pass
            else:
                self.weight[i] -= self.lr * np.dot(error[- i] * layer[i + 1] * (1.0 - layer[i + 1]), layer[i])  # np.transpose(layer[i]))
            pass
        pass
    
    def query(self, inputs_list=None):
        if inputs_list is None:
            inputs_list = np.random.rand(self.nodes[0])
            pass
        layer = np.array(inputs_list, ndmin=2).T
        for i in range(len(self.nodes) - 1):
            layer = self.activation_function[i](np.dot(self.weight[i], layer))
        return layer
    
    @staticmethod
    def __give_activation_function(activation_func: str):
        if activation_func == "sigmoid":
            return lambda x: classicGAN.__sigmoid(x)
            pass
        if activation_func == "tanh":
            return lambda x: np.tanh(x)
            pass
        if activation_func == "gaussian":
            return lambda x: np.exp(-x ** 2)
            pass
        if activation_func == "sin":
            return lambda x: np.sin(x)
            pass
        if activation_func == "atan":
            return lambda x: np.arctan(x)
            pass
        if activation_func == "":
            return lambda x: x
            pass
        raise Exception("No known activation function")
    
    pass
