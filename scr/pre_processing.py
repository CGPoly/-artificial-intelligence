import numpy as np
from sklearn import preprocessing


def turn_data_into_bool(input_data: np.array, threshold: float = 0.5) -> np.array:
    # returns the Data as array with only 1 and 0
    return preprocessing.Binarizer(threshold=threshold).transform(input_data)


def mean_removal(input_data: np.array) -> np.array:
    # removes the mean of an feature vector
    return preprocessing.scale(input_data)


def scaling(input_data: np.array) -> np.array:
    # scales the Data so that all values are between 1 and 0
    scaled_data = preprocessing.MinMaxScaler(feature_range=(0, 1))
    return scaled_data.fit_transform(input_data)


def normalize_L1(input_data: np.array) -> np.array:
    # scales the Data so that all Values in one row added = 1
    return preprocessing.normalize(input_data, norm="l1")


def normalize_L2(input_data: np.array) -> np.array:
    # scales the Data so that the square of all Values in one row added = 1
    return preprocessing.normalize(input_data, norm='l2')


class LabelEncoding:
    def train(self, input_labels: list):
        self.encoder.fit(input_labels)
        pass

    def encode(self, input_data: list) -> list:
        return self.encoder.transform(input_data)

    def decode(self, input_data: list) -> list:
        return self.encoder.inverse_transform(input_data)

    def __init__(self, input_labels: list):
        self.encoder = preprocessing.LabelEncoder()
        self.train(input_labels)
        pass
