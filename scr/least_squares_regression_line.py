import numpy as np


class LeastSquaresRegression:
    def __init__(self):
        self.m = np.random.rand()
        self.b = np.random.rand()
        pass

    def query(self, input_data):
        return input_data * self.m + self.b

    def train(self, input_data, output_data):
        n = len(input_data)

        mean_x = np.mean(input_data)
        mean_y = np.mean(output_data)

        sum_for_x = sum([(input_data[i] - mean_x) ** 2 for i in range(n)])
        sdx = np.sqrt((1 / (n - 1)) * sum_for_x)

        sum_for_y = sum([(output_data[i] - mean_y) ** 2 for i in range(n)])
        sdy = np.sqrt((1 / (n - 1)) * sum_for_y)

        xi_star = [(input_data[i] - mean_x) / sdx for i in range(n)]
        yi_star = [(output_data[i] - mean_y) / sdy for i in range(n)]

        sum_for_r = sum([xi_star[i] * yi_star[i] for i in range(n)])
        r = sum_for_r/n

        self.m = (r*sdy)/sdx
        self.b = mean_y-((r*sdy*mean_x)/sdx)
        pass

    def calc_error(self, input_data, output_data) -> float:
        return sum([(output_data[i] - self.query(input_data[i])) ** 2 for i in range(len(input_data))])[0]

    @staticmethod
    def calc_error_difference(predict_data, output_data):
        return sum((output_data - predict_data) ** 2)[0]
    pass
