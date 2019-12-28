class KNN:
    def __init__(self):
        self.training_data = [[]]
        pass

    def train(self, data_x, data_y):
        for i in range(len(data_x)):
            self.training_data[i].append(data_x[i][0])
            self.training_data[i].append(data_y[i][0])
            self.training_data.append([])
            pass
        del self.training_data[len(self.training_data) - 1]
        pass

    def query_multiple(self, data, neighbors: int, multi_dimensional: bool = False):
        output = []
        for i in data:
            output.append(self.query(i, neighbors, multi_dimensional))
            pass
        return output
        pass

    @staticmethod
    def __distance(point1: list, point2: list) -> float:
        return sum([(point1[i] - point2[i]) ** 2 for i in range(len(point1))]) ** 0.5

    def query(self, x, neighbors: int, multi_dimensional: bool = False) -> float:
        error = []
        exactly = False
        neighbors_y = []
        for k in range(neighbors):
            if not exactly:
                for i in range(len(self.training_data)):
                    if not any([self.training_data[i][0] == n for n in neighbors_y]):
                        if multi_dimensional:
                            input_num = self.training_data[i][0:len(self.training_data[i]) - 1]
                            error.append(self.__distance(input_num, x))
                            pass
                        else:
                            input_num = self.training_data[i][0]
                            error.append((input_num - x) ** 2)
                            pass
                        pass
                    pass
                min_error_index = error.index(min(error))
                if min_error_index == 0:
                    exactly = True
                    pass
                neighbors_y.append(self.training_data[min_error_index][1])
                pass
            pass
        if exactly:
            return neighbors_y[len(neighbors_y) - 1]
        return sum(neighbors_y) / neighbors

    @staticmethod
    def calc_error_difference(predict_data, output_data):
        return sum([((output_data[i][0] - predict_data[i]) ** 2) for i in range(len(output_data))])
    pass
