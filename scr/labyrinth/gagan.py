import numpy as np
import copy
import os


class Population:
    def __init__(self, population_size, filters: tuple, input_depth=4, batch_size=1):
        self.filters = filters
        self.population = []
        self.input_depth = input_depth
        self.batch_size = batch_size
        for i in range(population_size):
            self.population.append(GAGAN(self.input_depth, self.filters))
    
    def save_population(self, file, generation, fitness):
        new_path = "populations/" + file
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        np.save(new_path + "/info", np.array([len(self.population), self.input_depth, generation]))
        np.save(new_path + "/filters", np.array(self.filters))
        np.save(new_path + "/fitness", np.array(fitness))
        for i in range(len(self.population)):
            self.population[i].save_weight(new_path + "/" + str(i))
    
    def load(self, file):
        new_path = "populations/" + file
        meta = np.load(new_path + "/info.npy")
        self.input_depth = meta[1]
        if len(self.population) != meta[0]:
            self.filters = np.load(new_path + "/filters.npy")
            self.population = []
            for i in range(meta[0]):
                self.population.append(GAGAN(self.input_depth, self.filters))
        for i in range(meta[0]):
            self.population[i].load_weight(new_path + "/" + str(i) + ".npy")
        return meta[2], list(np.load(new_path + "/fitness.npy"))
    
    @staticmethod
    def __pick_one(arr, probability):
        pro = [float(i) / sum(probability) for i in probability]
        index = 0
        r = np.random.rand()
        while r > 0:
            r = r - pro[index]
            index += 1
        index -= 1
        return copy.deepcopy(arr[index])
    
    @staticmethod
    def similarity(result1: np.ndarray, result2: np.ndarray):
        score = 0
        for x in range(result1.shape[0]):
            for y in range(result1.shape[1]):
                if result2[x, y] == result1[x, y]:
                    score += 1
        return score/result1.size
    
    def train(self, input_size: tuple, evaluator, translator, mutation_chance: float, inputs_list: np.ndarray = None):
        if inputs_list is None:
            inputs_list = np.random.rand(self.batch_size, input_size[0], input_size[1], self.input_depth)
        elif len(inputs_list.shape) == 3:
            inputs_list = np.random.rand(self.batch_size, input_size[0], input_size[1], self.input_depth)
        inputs_similar = np.random.rand(self.batch_size, input_size[0], input_size[1], self.input_depth)
        fitness = []
        for i in range(len(self.population)):
            f_tmp = []
            for j in range(self.batch_size):
                # out = translator(self.population[i].query_expand(inputs_list))
                out = translator(self.population[i].query_convolution(inputs_list[j]))
                # fitness.append(evaluator(out) + (0 if keep_input else 0.5 * (1 - self.similarity(out, translator(self.population[i].query_expand(inputs_similar))))))
                similarity = self.similarity(out, translator(self.population[i].query_convolution(inputs_similar[j])))
                f_tmp.append(evaluator(out) - (0 if similarity < 0.85 else 0.65 * similarity))
            fitness.append(np.mean(f_tmp))
        old_population = copy.deepcopy(self.population)
        self.population[0] = copy.deepcopy(old_population[fitness.index(max(fitness))])
        for i in range(1, len(self.population)):
            self.population[i] = self.__pick_one(old_population, fitness)
            self.population[i].mutate(mutation_chance)
        return max(fitness), fitness.index(max(fitness))
    
    def query_batch(self, input_size: tuple, fitness_function, inputs_list: np.ndarray = None):
        if inputs_list is None:
            inputs_list = np.random.rand(self.batch_size, input_size[0], input_size[1], self.input_depth)
        results = []
        fitness = []
        for i in self.population:
            for j in range(self.batch_size):
                # results.append(i.query_expand(inputs_list))
                results.append(i.query_convolution(inputs_list[j]))
                fitness.append(fitness_function(results[-1]))
        return results[fitness.index(max(fitness))]
    
    def query(self, input_size: tuple, fitness_function, inputs_list: np.ndarray = None):
        if inputs_list is None:
            inputs_list = np.random.rand(input_size[0], input_size[1], self.input_depth)
        results = []
        fitness = []
        for i in self.population:
            # results.append(i.query_expand(inputs_list))
            results.append(i.query_convolution(inputs_list))
            fitness.append(fitness_function(results[-1]))
        return results[fitness.index(max(fitness))]
    
    def query_fast(self, input_size: tuple):
        # return self.population[0].query_expand(np.random.rand(input_size[0], input_size[1], self.input_depth))
        return self.population[0].query_convolution(np.random.rand(input_size[0], input_size[1], self.input_depth))
    pass


class GAGAN:
    @staticmethod
    def ReLU(x):
        if isinstance(x, list):
            return [x if i > 0 else 0 for i in x]
        if isinstance(x, np.ndarray):
            if len(x.shape) == 1:
                return np.asarray([x if i > 0 else 0 for i in x], np.ndarray)
            if len(x.shape) == 2:
                return np.asarray([[x if i > 0 else 0 for i in j] for j in x], np.ndarray)
            if len(x.shape) == 3:
                return np.asarray([[[x if i > 0 else 0 for i in j] for j in k] for k in x], np.ndarray)
            raise Exception("too many dimensions")
        return x if x > 0 else 0
    
    @staticmethod
    def sigmoid(x):
        return 1/(1 + np.exp(-x))
    
    def __init__(self, input_depth: int, filters: tuple):
        """filters: [layer][x_length, y_length, z_length, stride]"""
        """weights: [layer][filter, x, y, depth]"""
        self.weight = []
        """
        # weights for expanding
        for i in range(len(filters)):
            self.weight.append([])
            if i == 0:
                for j in range(filters[i][2]):
                    self.weight[i].append(np.random.normal(0.0, 0.02, (filters[i][0], filters[i][1], input_depth)))
            else:
                for j in range(filters[i][2]):
                    self.weight[i].append(np.random.normal(0.0, 0.02, (filters[i][0], filters[i][1], filters[i-1][2])))
        """
        # weights for convolutions
        self.stride = []
        for i in range(len(filters)):
            if i != 0:
                self.weight.append(np.random.normal(0.0, 0.2, (filters[i][2], filters[i][0], filters[i][1], filters[i - 1][2])))
            else:
                self.weight.append(np.random.normal(0.0, 0.2, (filters[i][2], filters[i][0], filters[i][1], input_depth)))
            try:
                self.stride.append(filters[i][3])
            except IndexError:
                self.stride.append(0)
        self.activation_function = lambda x: np.tanh(x)

    def save_weight(self, file):
        np.save(file, self.weight)
    
    def load_weight(self, file):
        self.weight = np.load(file, allow_pickle=True)
        
    def query_expand(self, input_: np.ndarray):
        if input_.shape[2] != self.weight[0][0].shape[2]:
            raise ValueError('input has a false depth! Depth ' + str(input_.shape[2]) + ' instead of ' + str(self.weight[0][0].shape[2]))
        for i in range(len(self.weight)):
            input_ = self.multiply(input_, i).copy()
        return input_
    
    def multiply(self, input_: [np.ndarray, list], i: int):
        __output = np.ones((input_.shape[0] * self.weight[i][0].shape[0],
                            input_.shape[1] * self.weight[i][0].shape[1],
                            len(self.weight[i])))
        w_x_len = self.weight[i][0].shape[0]
        w_y_len = self.weight[i][0].shape[1]
        for x in range(input_.shape[0]):
            for y in range(input_.shape[1]):
                # __output[x * w_x_len:(x * w_x_len) + w_x_len, y * w_y_len:(y * w_y_len) + w_y_len] = np.dot(self.weight[i], input_[x, y]).T
                for filter_ in range(len(self.weight[i])):
                    __output[x * w_x_len:(x * w_x_len) + w_x_len, y * w_y_len:(y * w_y_len) + w_y_len, filter_] = np.dot(self.weight[i][filter_], input_[x, y])
        return __output

    def query_convolution(self, __input: np.ndarray) -> np.ndarray:
        # print(__input.shape)
        for i in range(len(self.weight)):
            stride = self.stride[i]
            __input = np.asarray([[__input[(x - stride) // (stride + 1), (y - stride) // (stride + 1), :] if x % (stride + 1) == stride and y % (
                        stride + 1) == stride else np.zeros(__input.shape[2])
                                   for y in range((stride * (__input.shape[1] + 1) + __input.shape[1] + 1) - 1)]
                                  for x in range((stride * (__input.shape[0] + 1) + __input.shape[0] + 1) - 1)])
            __output = np.zeros((__input.shape[0] - (self.weight[i].shape[1] - 1), __input.shape[1] - (self.weight[i].shape[2] - 1), self.weight[i].shape[0]))
            for x in range(__output.shape[0]):
                for y in range(__output.shape[1]):
                    __output[x, y, :] = self.__convolve(__input[x:x + self.weight[i].shape[1], y:y + self.weight[i].shape[2], :], i)
            # print(__output.shape)
            __input = __output.copy()
        # print("finished")
        return __input

    def __convolve(self, __input: np.ndarray, i: int):
        return np.transpose(__input * self.weight[i], (tuple(np.append(range(1, len(self.weight[i].shape)), 0)))).sum(tuple(range(len(self.weight[i].shape) - 1)))
    
    def mutate(self, mutation_rate):
        for i in range(len(self.weight)):
            for j in range(len(self.weight[i])):
                self.weight[i][j] = np.array([[self.weight[i][j][x, y] + np.random.normal(0.0, 0.1)
                                               if np.random.rand() < mutation_rate else self.weight[i][j][x, y]
                                               for y in range(self.weight[i][j].shape[1])]
                                              for x in range(self.weight[i][j].shape[0])], np.ndarray)
