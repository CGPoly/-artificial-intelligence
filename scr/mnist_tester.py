from deep_neural_network import *
import scipy.ndimage.interpolation as science_ndi
import numpy as np
import matplotlib.pyplot as plt
# import convolution_codes.cnn_stolen as ccccs


class mnist_test:
    @staticmethod
    def plot_input_str(input_string: str):
        image_array = np.asfarray(input_string.split(',')[1:]).reshape((28, 28))
        plt.imshow(image_array, cmap='Greys', interpolation='None')
        plt.show()
        pass

    @staticmethod
    def plot_input(input_to_plot: np.ndarray):
        image_array = input_to_plot.reshape((28, 28))
        plt.imshow(image_array, cmap='Greys', interpolation='None')
        plt.show()
        pass

    @staticmethod
    def convert_to_input(input_string: str) -> np.ndarray:
        return (np.asfarray(input_string.split(',')[1:]) / 255.0 * 0.99) + 0.01

    @staticmethod
    def convert_to_output(input_string: str, o_nodes: int) -> np.ndarray:
        targets = np.zeros(o_nodes) + 0.01
        targets[int(input_string.split(',')[0])] = 0.99
        return targets

    @staticmethod
    def translate_output(output_list: list) -> int:
        biggest = [-1, -1]
        for i in range(0, 9):
            if output_list[i] > biggest[0]:
                biggest = [output_list[i], i]
        return biggest[1]

    def test_neural_network(self, test_list: list) -> [float]:
        scoreboard = []
        # scoreboard_cnn = []
        # cnn = ccccs.LiteOCR("convolution_codes/alpha_weights.pkl")

        for record in test_list:
            all_values = record.split(',')
            correct_label = int(all_values[0])
            inputs = self.convert_to_input(record)
            outputs = self.n.query(inputs)
            # outputs_cnn = cnn.predict(inputs)
            label = np.argmax(outputs)
            if label == correct_label:
                scoreboard.append(1)
            else:
                scoreboard.append(0)
                pass
            """try:
                # label_cnn = int(outputs_cnn)
                if label_cnn == correct_label:
                    scoreboard.append(1)
                else:
                    scoreboard.append(0)
            except ValueError:
                scoreboard.append(0)
            pass"""
        print()
        scoreboard_array = np.asarray(scoreboard)
        # scoreboard_array_cnn = np.asarray(scoreboard_cnn)
        return scoreboard_array.sum() / scoreboard_array.size  # , scoreboard_array_cnn.sum() / scoreboard_array_cnn.size

    def train_neural_network(self, train_list: list, rotate: bool = False):
        for record in train_list:
            inputs = self.convert_to_input(record)
            targets = self.convert_to_output(record, 10)
            self.n.train(inputs, targets)
            if rotate:
                # rotations
                for i in range(0, 6, 3):
                    inputs_plus_i_img = science_ndi.rotate(inputs.reshape(28, 28), i, cval=0.01, order=1, reshape=False)
                    self.n.train(inputs_plus_i_img.reshape(784), targets)
                    inputs_minus_i_img = science_ndi.rotate(inputs.reshape(28, 28), -i, cval=0.01, order=1, reshape=False)
                    self.n.train(inputs_minus_i_img.reshape(784), targets)
                    pass
                pass
            pass
        for i in range(len(self.n.weight) - 1):
            np.save(self.file_name + str(i), self.n.weight[i])
            np.save(self.file_name + "out", self.n.weight[len(self.n.weight) - 1])
        pass

    def set_weights(self):
        length = len(self.n.nodes) - 1
        weights = []
        for i in range(length - 1):
            weights.append(np.load(self.file_name + str(i) + ".npy"))
        weights.append(np.load(self.file_name + "out" + ".npy"))
        self.n.set_weight(weights)
        pass

    def __init__(self):
        test_data_file = open("mnist_data_set/mnist_test.csv", 'r')
        test_data_list = test_data_file.readlines()
        test_data_file.close()

        self.file_name = "weights/weight"

        new_training = True
        epoch = 1
        nodes = [784, 16, 16, 10]
        self.n = NeuralNetwork(nodes, 0.01, False, "sigmoid")

        if new_training is None:
            self.set_weights()
            training_data_file = open("mnist_data_set/mnist_train.csv", 'r')
            training_data_list = training_data_file.readlines()
            training_data_file.close()
            for i in range(epoch):
                self.train_neural_network(training_data_list)
                pass
            pass
        elif new_training:
            training_data_file = open("mnist_data_set/mnist_train.csv", 'r')
            training_data_list = training_data_file.readlines()
            training_data_file.close()
            for i in range(epoch):
                self.train_neural_network(training_data_list)
                pass
            pass
        else:
            self.set_weights()
            pass

        print("performance =", self.test_neural_network(test_data_list))
        for i in range(10):
            image_array = self.n.back_query(i).reshape((28, 28))
            plt.title(str(i))
            plt.imshow(image_array, cmap='Greys', interpolation='None')
            plt.show()
            pass
        pass
    pass


if __name__ == "__main__":
    mnist_test()
