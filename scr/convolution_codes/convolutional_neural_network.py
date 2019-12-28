import numpy as np
import deep_neural_network as dnn


class CNN:
    class __Block:
        class __Convolution:
            def __init__(self, convolution: np.ndarray = None, average: bool = True, shrink: bool = False):
                if convolution is None:
                    self.only_pool = True
                    pass
                else:
                    self.only_pool = False
                    self.shrink = shrink
                    self.convolution = convolution
                    self.average = average
                    self.shape = convolution.shape
                    pass
                pass

            def query_image(self, input_image: np.ndarray, round_num: int = -1) -> np.ndarray:
                if self.only_pool:
                    raise Exception("You had no convolution defined, so you could only use the pool method")
                    pass
                output_array = np.ndarray(input_image.shape)
                for x in range(input_image.shape[0]):
                    for y in range(input_image.shape[1]):
                        for z in range(input_image.shape[2]):
                            output_array[x, y, z] = self.__query_segment(input_image, [x, y, z], round_num)
                        pass
                    pass
                if self.shrink:
                    output_array = output_array[int(((self.shape[0] + 1) / 2) - 1):output_array.shape[0] - int(((self.shape[0] + 1) / 2) - 1),
                                                int(((self.shape[1] + 1) / 2) - 1):output_array.shape[1] - int(((self.shape[1] + 1) / 2) - 1)]
                print(output_array)
                print(np.sum(output_array, axis=2))
                return np.sum(output_array, axis=2)
                pass

            def __query_segment(self, input_image: np.ndarray, index: list, round_num: int = -1) -> float:
                output_array = np.zeros(self.shape)
                convolution = np.copy(self.convolution)
                for x in range(convolution.shape[0]):
                    for y in range(convolution.shape[1]):
                        for z in range(convolution.shape[2]):
                            try:
                                x_index = x + index[0] - int(((convolution.shape[0] + 1) / 2) - 1)
                                y_index = y + index[1] - int(((convolution.shape[1] + 1) / 2) - 1)
                                z_index = z + index[2] - int(((convolution.shape[2] + 1) / 2) - 1)
                                if x_index < 0 or y_index < 0 or z_index < 0:
                                    convolution[x, y, z] = 0
                                    pass
                                else:
                                    output_array[x, y, z] = convolution[x, y, z] * input_image[x_index, y_index, z_index]
                                    pass
                                pass
                            except IndexError:
                                convolution[x, y, z] = 0
                                pass
                            pass
                        pass
                    pass
                if round_num == -1:
                    if self.average:
                        if not sum(sum(sum(convolution))) == 0:
                            return sum(sum(sum(output_array))) / sum(sum(sum(convolution)))
                        raise ValueError
                    return sum(sum(sum(output_array)))
                if self.average:
                    if not sum(sum(sum(convolution))) == 0:
                        return round(sum((sum(sum(output_array))) / sum(sum(sum(convolution)))) * round_num) / round_num
                    raise ValueError
                return round((sum(sum(sum(output_array)))) * round_num) / round_num

            @staticmethod
            def pool(input_image: np.ndarray, pool_size: int):
                output_array = np.ndarray((int(input_image.shape[0] / pool_size), int(input_image.shape[1] / pool_size), input_image.shape[2]))  # (data.shape[0], data.shape[1]))
                for x in range(output_array.shape[0]):
                    for y in range(output_array.shape[1]):
                        convolution = []
                        convolution_index = [[]]
                        del convolution_index[0]
                        for x_s in range(-int(((pool_size - 1) / 2)), int((pool_size - 1) / 2) + 1, 1):
                            for y_s in range(-int((pool_size - 1) / 2), int((pool_size - 1) / 2) + 1, 1):
                                try:
                                    convolution.append(sum(input_image[x * pool_size + x_s][y * pool_size + y_s]))
                                    pass
                                except IndexError:
                                    convolution.append(-float("inf"))
                                    pass
                                convolution_index.append([x+x_s, y+y_s])
                                pass
                            pass
                        index = convolution_index[convolution.index(max(convolution))]
                        output_array[x, y] = input_image[index[0], index[1]]
                        pass
                    pass
                return output_array
            pass

        def __init__(self, convolutions: np.ndarray, pool_size: int, activation_function: str = "ReLu"):
            self.convolutions = [self.__Convolution(convolution=i, average=False, shrink=True) for i in convolutions]
            self.pool = self.__Convolution()
            self.pool_size = pool_size
            self.activation_function = self.__give_activation_function(activation_function)
            pass

        def query(self, input_image: np.ndarray):
            output_images = np.asarray([i.query_image(input_image) for i in self.convolutions])
            output_image = np.asarray([[[z for z in y] for y in x] for x in output_images], np.ndarray)
            # output_image = np.ndarray((output_images.shape[1], output_images.shape[2], output_images.shape[0]))
            # for x in output_image:
            #     for y in x:
            #         for z in y:
            #             output_image[y, z, x] = z
            #             pass
            #         pass
            #     pass
            # output_image = self.convolutions.query_image(input_image)
            for x in range(output_image.shape[0]):
                for y in range(output_image.shape[1]):
                    for z in range(output_image.shape[3]):
                        output_image[x, y, z] = self.activation_function(output_image[x, y, z])
                        pass
                    pass
                pass
            return self.pool.pool(input_image, self.pool_size)
            pass

        @staticmethod
        def __give_activation_function(activation_func: str):
            if activation_func == "sigmoid":
                return lambda x: 1/(1 + np.exp(-x))
            if activation_func == "ReLu":
                def __ReLU(x: float):
                    if x < 0:
                        return 0
                    return x
                return lambda x: __ReLU(x)
            if activation_func == "tanh":
                return lambda x: np.tanh(x)
            if activation_func == "gaussian":
                return lambda x: np.exp(-x ** 2)
            if activation_func == "":
                return lambda x: x
            raise ValueError("unknown input")
        pass

    def __init__(self, blocks: [list, np.ndarray], nodes: [list, np.ndarray], learning_rate: float = 0.1):
        # self.blocks = [self.__Block(np.random.normal(loc=0, size=i[0]), i[1]) for i in blocks]
        self.blocks = []
        for i in blocks:
            pool = i[1]

            self.blocks.append(self.__Block(np.random.normal(loc=0, size=i[0]), pool))
            pass
        self.fully_connected_layer = dnn.NeuralNetwork(nodes, learning_rate)
        pass

    def query(self, input_image: np.ndarray) -> [np.ndarray]:
        output_img = input_image
        for i in self.blocks:
            output_img = i.query(output_img)
            pass
        return self.fully_connected_layer.query(output_img.flatten()), output_img
    pass
