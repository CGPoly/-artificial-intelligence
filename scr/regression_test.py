import matplotlib.pyplot as plt
import numpy as np
import deep_neural_network
import least_squares_regression_line
import k_nearest_neighbor

x_data = [[]]
y_data = [[]]


def function(input_x):
    return np.sin(input_x*np.pi)  # np.exp(-input_x*10)  # abs(np.sin(input_x*2*np.pi))  # 2*x + 2


for i in range(10000):
    x_data[i].append(np.abs(np.random.rand()))
    y_data[i].append(function(x_data[i][0]))
    x_data.append([])
    y_data.append([])
    pass
x_data.remove([])
y_data.remove([])

training_length = int(0.6 * len(x_data))
testing_length = len(x_data) - training_length

X_train, y_train = x_data[:training_length], y_data[:training_length]
X_test, y_test = x_data[training_length:], y_data[training_length:]

n = deep_neural_network.NeuralNetwork([1, 10, 10, 1], 1, True, "sigmoid")
for record in range(len(X_train)):
    x = [X_train[record]]
    y = [y_train[record]]
    n.train(x, y)
    pass

print()
n_predicts = []
for record in range(len(X_test)):
    x = [X_test[record]]
    y = n.query(x)
    n_predicts.append(y[0])
    pass
n_predicts_array = np.asarray(n_predicts)

lssl = least_squares_regression_line.LeastSquaresRegression()
lssl.train(X_train, y_train)
lssl_predicts = lssl.query(X_test)
lssl_predicts_array = np.asarray(lssl_predicts)

print("Performance own NN mean = " + str(lssl.calc_error_difference(n_predicts_array, y_test)/len(y_test)))
print("Performance own NN sum = " + str(lssl.calc_error_difference(n_predicts_array, y_test)))
print()

print("Performance own LSSL mean = " + str(lssl.calc_error_difference(lssl_predicts_array, y_test)/len(y_test)))
print("Performance own LSSL sum = " + str(lssl.calc_error_difference(lssl_predicts_array, y_test)))
print()

knn = k_nearest_neighbor.KNN()
knn.train(X_train, y_train)
knn_predicts = knn.query_multiple(X_test, 1)
knn_predicts_array = np.asarray(knn_predicts)

print("Performance own KNN mean = " + str(knn.calc_error_difference(knn_predicts_array, y_test) / len(y_test)))
print("Performance own KNN sum = " + str(knn.calc_error_difference(knn_predicts_array, y_test)))
print()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X_train, y_train, color='red')
ax.scatter(X_test, lssl_predicts, color='green')
ax.scatter(X_test, n_predicts, color='blue')
ax.scatter(X_test, knn_predicts, color='purple')
plt.show()
