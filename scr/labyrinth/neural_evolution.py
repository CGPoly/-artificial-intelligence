import numpy as np
import copy


class Population:
    def __init__(self, size, nodes, use_bias: bool = False, activation_func="sigmoid"):
        self.population = []
        self.NODES = nodes
        self.USE_BIAS = use_bias
        self.ACTIVATION_FUNC = activation_func
        for i in range(size):
            self.population.append(NeuralEvolution(self.NODES, self.USE_BIAS, self.ACTIVATION_FUNC))
            pass
        pass
    
    @staticmethod
    def __pick_one(arr, probability):
        pro = [float(i) / sum(probability) for i in probability]
        index = 0
        r = np.random.rand()
        while r > 0:
            r = r - pro[index]
            index += 1
            pass
        index -= 1
        return arr[index]
    
    def train(self, fitness_function, mutation_chance: float):
        fitness = []
        # print("starting search")
        inputs_list = np.random.rand(self.NODES[0])
        for i in self.population:
            fitness.append(fitness_function(i.query(inputs_list)))
            pass
        old_population = self.population.copy()
        for i in range(len(self.population)):
            self.population[i] = self.__pick_one(old_population, fitness)
            self.population[i].mutate(mutation_chance)
            pass
        return max(fitness)
    
    def query(self, fitness_function):
        results = []
        fitness = []
        inputs_list = np.random.rand(self.NODES[0])
        for i in self.population:
            results.append(i.query(inputs_list))
            fitness.append(fitness_function(results[-1]))
            pass
        return results[fitness.index(max(fitness))]  # , fitness
    
    def query_fast(self):
        return self.population[0].query()
        pass
    pass


class NeuralEvolution:
    @staticmethod
    def __sigmoid(x):
        return 1 / (1 + np.e ** -x)

    def __init__(self, nodes: [list, np.ndarray], use_bias: bool = False, activation_func="sigmoid"):
        self.nodes = np.asarray(nodes)
        self.use_bias = use_bias

        if use_bias:
            for i in range(len(self.nodes) - 1):
                self.nodes[i] += 1
                pass
            pass

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
            raise Exception('activation_func is in no accepted type')
            pass
        pass

    def set_weight(self, weight: list):
        self.weight = weight

    def mutate(self, mutation_rate):
        # print()
        # print("all weights", self.weight)
        for i in range(len(self.weight)):
            self.weight[i] = np.array([[self.weight[i][x, y] + np.random.normal(0.0, 0.1)
                                        if np.random.rand() < mutation_rate else self.weight[i][x, y]
                                        for y in range(self.weight[i].shape[1])] for x in range(self.weight[i].shape[0])], np.ndarray)
            # self.weight[i] = np.array(map(
            #     lambda x: x+np.random.normal(0.0, pow(self.nodes[i], -0.5)) if np.random.rand() < mutation_rate
            #     else x, self.weight[i]), np.ndarray)
            # print()
            # print("i for weights", i)
            # print()
            # print("weights[i]", self.weight[i])
            pass
        # print(), print(), print(), print(), print()
        pass
    
    def copy(self):
        return copy.deepcopy(self)
    
    def query(self, inputs_list):  # =None):
        # if inputs_list is None:
        #     inputs_list = np.random.rand(self.nodes[0])
        #     pass
        layer = np.array(inputs_list, ndmin=2).T
        for i in range(len(self.nodes) - 1):
            layer = self.activation_function[i](np.dot(self.weight[i], layer))
        return layer

    @staticmethod
    def __give_activation_function(activation_func: str):
        if activation_func == "sigmoid":
            return lambda x: NeuralEvolution.__sigmoid(x)
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

# class Population:
#     def __init__(self, nodes, size, use_bias: bool = False, activation_func="sigmoid"):
#         self.population = []
#         for i in range(size):
#             self.population.append(NeuralEvolution(nodes, use_bias, activation_func))
#             pass
#         pass
#
#     def train(self, loss_function, mutation_chance: float):
#         loss = []
#         print("starting search")
#         inputs_list = np.random.rand(self.nodes[0])
#         for i in self.population:
#             loss.append(1-loss_function(i.query(inputs_list)))
#             pass
#         print("finished search")
#         loss = [float(i)/max(loss) for i in loss]
#         # loss.sort(reverse=True)
#         killed = []
#         for i in range(len(self.population)):
#             if loss[i] > np.random.rand():
#                 killed.append(i)
#                 pass
#             pass
#         print("starting mutation")
#         for i in range(len(self.population)):
#             if any([j == i for j in killed]):
#                 new_net = None
#                 while new_net is None:
#                     new_net = self.population[np.random.randint(0, len(self.population))]
#                     pass
#                 self.population[i] = copy.copy(new_net)
#                 pass
#             if loss[i] != max(loss):
#                 self.population[i].mutate(mutation_chance)
#             pass
#
#         print("finished mutation")
#         print()
#         return max(loss)
#
#     def query(self):
#         return self.population[0].query()
#     pass
#
#
# class NeuralEvolution:
#     @staticmethod
#     def __sigmoid(x):
#         return 1 / (1 + np.e ** -x)
#
#     def __init__(self, nodes: [list, np.ndarray], use_bias: bool = False, activation_func="sigmoid"):
#         self.nodes = np.asarray(nodes)
#         self.use_bias = use_bias
#
#         if use_bias:
#             for i in range(len(self.nodes) - 1):
#                 self.nodes[i] += 1
#                 pass
#             pass
#
#         self.weight = []
#         for i in range(len(self.nodes) - 1):
#             self.weight.append(np.random.normal(0.0, pow(self.nodes[i], -0.5), (self.nodes[i + 1], self.nodes[i])))
#             pass
#
#         self.activation_function = []
#
#         if isinstance(activation_func, list):
#             for i in range(len(self.nodes) - 1):
#                 self.activation_function.append(self.__give_activation_function(activation_func[i]))
#                 pass
#             pass
#         elif isinstance(activation_func, str):
#             for i in range(len(self.nodes) - 1):
#                 self.activation_function.append(self.__give_activation_function(activation_func))
#                 pass
#             pass
#         elif callable(activation_func):
#             for i in range(len(self.nodes) - 1):
#                 self.activation_function.append([activation_func])
#                 pass
#             pass
#         else:
#             raise Exception('activation_func is in no accepted type')
#             pass
#         pass
#
#     def set_weight(self, weight: list):
#         self.weight = weight
#
#     def mutate(self, mutation_chance):
#         for i in range(len(self.weight)):
#             self.weight[i] = np.add(self.weight[i], np.asarray([[0 if mutation_chance < np.random.rand()
#             else np.random.normal(0.0, pow(self.nodes[i], -0.5))
#             for y in range(self.weight[i].shape[1])] for x in range(self.weight[i].shape[0])]))
#
#             pass
#         pass
#
#     def query(self, inputs_list=None):
#         if inputs_list is None:
#             inputs_list = np.random.rand(self.nodes[0])
#             pass
#         layer = np.array(inputs_list, ndmin=2).T
#         for i in range(len(self.nodes) - 1):
#             layer = self.activation_function[i](np.dot(self.weight[i], layer))
#         return layer
#
#     @staticmethod
#     def __give_activation_function(activation_func: str):
#         if activation_func == "sigmoid":
#             return lambda x: NeuralEvolution.__sigmoid(x)
#             pass
#         if activation_func == "tanh":
#             return lambda x: np.tanh(x)
#             pass
#         if activation_func == "gaussian":
#             return lambda x: np.exp(-x ** 2)
#             pass
#         if activation_func == "sin":
#             return lambda x: np.sin(x)
#             pass
#         if activation_func == "atan":
#             return lambda x: np.arctan(x)
#             pass
#         if activation_func == "":
#             return lambda x: x
#             pass
#         raise Exception("No known activation function")
#
#     pass
