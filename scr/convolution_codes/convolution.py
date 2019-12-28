import numpy as np


class Convolution:
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
                output_array[x][y] = self.__query_segment(input_image, [x, y], round_num)
                pass
            pass
        if self.shrink:
            return output_array[int(((self.shape[0] + 1) / 2) - 1):output_array.shape[0]-int(((self.shape[0] + 1) / 2) - 1),
                                int(((self.shape[1] + 1) / 2) - 1):output_array.shape[1]-int(((self.shape[1] + 1) / 2) - 1)]
        return output_array
        pass

    def __query_segment(self, input_image: np.ndarray, index: list, round_num: int = -1) -> float:
        output_array = np.zeros(self.shape)
        convolution = np.copy(self.convolution)
        for x in range(convolution.shape[0]):
            for y in range(convolution.shape[1]):
                try:
                    x_index = x + index[0] - int(((convolution.shape[0] + 1) / 2) - 1)
                    y_index = y + index[1] - int(((convolution.shape[1] + 1) / 2) - 1)
                    if x_index < 0 or y_index < 0:
                        convolution[x][y] = 0
                        pass
                    else:
                        output_array[x][y] = convolution[x][y] * input_image[x_index][y_index]
                        pass
                    pass
                except IndexError:
                    convolution[x][y] = 0
                    pass
                pass
            pass
        if round_num == -1:
            if self.average:
                if not sum(sum(convolution)) == 0:
                    return sum(sum(output_array)) / sum(sum(convolution))
                raise ValueError
            return sum(sum(output_array))
        if self.average:
            if not sum(sum(convolution)) == 0:
                return round((sum(sum(output_array)) / sum(sum(convolution))) * round_num) / round_num
            raise ValueError
        return round((sum(sum(output_array))) * round_num) / round_num

    @staticmethod
    def pool(input_image: np.ndarray, pool_size: int):
        output_array = np.ndarray((int(input_image.shape[0] / pool_size), int(input_image.shape[1] / pool_size)))  # (data.shape[0], data.shape[1]))
        for x in range(output_array.shape[0]):
            for y in range(output_array.shape[1]):
                convolution = []
                for x_s in range(-int(((pool_size - 1) / 2)), int((pool_size - 1) / 2) + 1, 1):
                    for y_s in range(-int((pool_size - 1) / 2), int((pool_size - 1) / 2) + 1, 1):
                        try:
                            convolution.append(input_image[x * pool_size + x_s][y * pool_size + y_s])
                            pass
                        except IndexError:
                            convolution.append(-float("inf"))
                            pass
                        pass
                    pass
                output_array[x][y] = max(convolution)
                pass
            pass
        return output_array
    pass
