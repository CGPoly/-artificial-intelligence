import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.image as pim
import convolution_codes.convolution as ccc


class AverageBlur:
    def __init__(self, size: tuple):
        x_size = (size[0] * 2) - 1
        y_size = (size[1] * 2) - 1
        self.convolution = ccc.Convolution(np.ones((x_size, y_size)))
        pass

    def blur_image(self, input_image: np.ndarray) -> np.ndarray:
        return self.convolution.query_image(input_image)
    pass


class GaussianBlur:
    class __Gaussian:
        def __init__(self, peak_height: float = 4, width: float = 1, position: float = 0):
            if not width == 0:
                self.a = peak_height
                self.b = position
                self.c = width
                pass
            else:
                raise ValueError("width can't be 0 (zero)")
            pass

        def query(self, x: float) -> float:
            return self.a * np.exp(- (((x - self.b) ** 2) / (2 * (self.c ** 2))))
        pass

    def __init__(self, size: tuple, gaussian_peak_height: float = 1, gaussian_width: float = 1):
        x_size = (size[0] * 2) - 1
        y_size = (size[1] * 2) - 1
        gaussian = self.__Gaussian(gaussian_peak_height, gaussian_width, 0)
        convolution_array = np.zeros((x_size, y_size))
        for x in range(x_size):
            for y in range(y_size):
                convolution_array[x][y] = gaussian.query(self.__distance([x - (x_size - 1) / 2, y - ((y_size - 1) / 2)], [0, 0]))
                pass
            pass
        self.convolution = ccc.Convolution(convolution_array)
        pass

    @staticmethod
    def __distance(point1: list, point2: list) -> float:
        return sum([(point1[i] - point2[i]) ** 2 for i in range(len(point1))]) ** 0.5

    def blur_image(self, input_image: np.ndarray) -> np.ndarray:
        return self.convolution.query_image(input_image)
    pass


class SobelEdge:
    class __Gaussian:
        def __init__(self, peak_height: float = 4, width: float = 1, position: float = 0):
            if not width == 0:
                self.a = peak_height
                self.b = position
                self.c = width
                pass
            else:
                raise ValueError("width can't be 0 (zero)")
            pass

        def query(self, x: float) -> float:
            return self.a * np.exp(- (((x - self.b) ** 2) / (2 * (self.c ** 2))))
        pass

    def __init__(self, size: tuple, gaussian_peak_height: float = 1, gaussian_width: float = 1):
        x_size = (size[0] * 2) - 1
        y_size = (size[1] * 2) - 1
        gaussian = self.__Gaussian(gaussian_peak_height, gaussian_width, 0)
        convolution_y_array = np.zeros((x_size, y_size))
        convolution_x_array = np.zeros((x_size, y_size))
        for x in range(x_size):
            for y in range(y_size):
                if not y == (y_size - 1) / 2:
                    if y < (y_size - 1) / 2:
                        convolution_y_array[x][y] = - gaussian.query(self.__distance([x - (x_size - 1) / 2, y - ((y_size - 1) / 2)], [0, 0]))
                        pass
                    else:
                        convolution_y_array[x][y] = gaussian.query(self.__distance([x - (x_size - 1) / 2, y - ((y_size - 1) / 2)], [0, 0]))
                        pass
                    pass
                if not x == (x_size - 1)/2:
                    if x < (x_size - 1) / 2:
                        convolution_x_array[x][y] = - gaussian.query(self.__distance([x - (x_size - 1) / 2, y - ((y_size - 1) / 2)], [0, 0]))
                        pass
                    else:
                        convolution_x_array[x][y] = gaussian.query(self.__distance([x - (x_size - 1) / 2, y - ((y_size - 1) / 2)], [0, 0]))
                        pass
                    pass
                pass
            pass
        self.convolution_size = (x_size, y_size)
        self.convolution_x = ccc.Convolution(convolution_x_array, average=False, shrink=True)
        self.convolution_y = ccc.Convolution(convolution_y_array, average=False, shrink=True)
        pass

    @staticmethod
    def __distance(point1: list, point2: list) -> float:
        return sum([(point1[i] - point2[i]) ** 2 for i in range(len(point1))]) ** 0.5

    def detect_edge_x(self, input_image: np.ndarray) -> np.ndarray:
        return self.convolution_x.query_image(input_image, round_num=10000)

    def detect_edge_y(self, input_image: np.ndarray) -> np.ndarray:
        return self.convolution_y.query_image(input_image, round_num=10000)

    def detect_edge_magnitude(self, input_image: np.ndarray) -> np.ndarray:
        edge_x = self.detect_edge_x(input_image)
        edge_y = self.detect_edge_y(input_image)
        edge = np.ndarray(edge_x.shape)
        for x in range(edge_x.shape[0]):
            for y in range(edge_x.shape[1]):
                edge[x, y] = np.sqrt(edge_x[x, y] ** 2 + edge_y[x, y] ** 2)
                pass
            pass
        return edge

    def detect_edge_angle(self, input_image: np.ndarray) -> np.ndarray:
        edge_x = self.detect_edge_x(input_image)
        edge_y = self.detect_edge_y(input_image)
        edge = np.ndarray(edge_x.shape)
        for x in range(edge_x.shape[0]):
            for y in range(edge_x.shape[1]):
                edge[x, y] = np.arctan(np.nan_to_num(edge_y[x, y] / edge_x[x, y]))
                pass
            pass
        return edge
    pass


# class Pool:
#     @staticmethod
#     def pooling(data: np.ndarray, pool_size: int):
#         output_array = np.ndarray((int(data.shape[0] / pool_size), int(data.shape[1] / pool_size)))  # (data.shape[0], data.shape[1]))
#         for x in range(output_array.shape[0]):
#             for y in range(output_array.shape[1]):
#                 convolution = []
#                 for x_s in range(-int(((pool_size - 1) / 2)), int((pool_size - 1) / 2) + 1, 1):
#                     for y_s in range(-int((pool_size - 1) / 2), int((pool_size - 1) / 2) + 1, 1):
#                         try:
#                             convolution.append(data[x * pool_size + x_s][y * pool_size + y_s])
#                             pass
#                         except IndexError:
#                             convolution.append(-float("inf"))
#                             pass
#                         pass
#                     pass
#                 output_array[x][y] = max(convolution)
#                 pass
#             pass
#         return output_array
#     pass


class CollapseColors:
    @staticmethod
    def collapse_colors(data: np.ndarray) -> np.ndarray:
        output = np.ndarray((data.shape[0], data.shape[1]))
        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                output[x][y] = data[x][y][0]
                pass
            pass
        return output
    pass


a = AverageBlur((10, 10))
g = GaussianBlur((10, 10), 4, 0.5)
pool = ccc.Convolution()
sobel = SobelEdge((2, 2), 4, 1)
c_d = CollapseColors()
image = np.asarray(c_d.collapse_colors(np.asarray(plt.imread("test_img_1.png"), np.ndarray)))
# [[50, 50, 50, 50, 100, 100, 100, 100], [50, 50, 50, 50, 100, 100, 100, 100], [50, 50, 50, 50, 100, 100, 100, 100], [50, 50, 50, 50, 100, 100, 100, 100]], np.ndarray)
# c_d.collapse_colors(np.asarray(plt.imread("test_img_1.png"), np.ndarray))
# np.random.random((30, 30))
plt.show()
pool_img = pool.pool(image, 10)
blur_average = a.blur_image(pool_img)
blur_gaussian = g.blur_image(pool_img)
sobel_x = sobel.detect_edge_x(blur_gaussian)
sobel_y = sobel.detect_edge_y(blur_gaussian)
sobel_mag = sobel.detect_edge_magnitude(blur_gaussian)
sobel_angle = sobel.detect_edge_angle(blur_gaussian)

image_array = np.asfarray(image)
pool_array = np.asarray(pool_img)
blur_average_array = np.asfarray(blur_average)
blur_gaussian_array = np.asfarray(blur_gaussian)
sobel_x_array = np.asfarray(sobel_x)
sobel_y_array = np.asfarray(sobel_y)
sobel_mag_array = np.asfarray(sobel_mag)
sobel_angle_array = np.asfarray(sobel_angle)

f, ax_arr = plt.subplots(4, 2)
ax_arr[0, 0].imshow(image_array, cmap='Greys', interpolation='None')
ax_arr[0, 1].imshow(pool_array, cmap='Greys', interpolation='None')
ax_arr[1, 0].imshow(blur_average_array, cmap='Greys', interpolation='None')
ax_arr[1, 1].imshow(blur_gaussian_array, cmap='Greys', interpolation='None')
ax_arr[2, 0].imshow(sobel_x_array, cmap='Greys', interpolation='None')
ax_arr[2, 1].imshow(sobel_y_array, cmap='Greys', interpolation='None')
ax_arr[3, 0].imshow(sobel_mag_array, cmap='Greys', interpolation='None')
ax_arr[3, 1].imshow(sobel_angle_array, cmap='Greys', interpolation='None')
plt.show()

# pim.imsave('orig.png', image_array, cmap='Greys')
# pim.imsave('average.png', blur_average_array, cmap='Greys')
# pim.imsave('gaussian.png', blur_gaussian_array, cmap='Greys')
