import numpy as np
import sys

sys.setrecursionlimit(4000)


class Search:
    @staticmethod
    def start_search(graph):
        p = []
        n = []
        search_area = [(0, 1), (-1, 0), (1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        for i in range(len(search_area)):
            for j in range(len(search_area)-1):
                for k in range(len(search_area)-2):
                    graph_tmp = np.copy(graph)
                    n.append(Search.__search_from(graph_tmp, search_area))
                    p.append(graph_tmp)
                    a, b = search_area[3], search_area[2]
                    search_area[3], search_area[2] = b, a
                search_tmp = search_area[0:]
                search_area = search_area[:0] + search_tmp[-1:] + search_tmp[:-1]
            search_area = search_area[-1:] + search_area[:-1]
        p = np.asarray(p, np.ndarray)
        path_length = [float('inf')]*len(p)
        for i in range(len(p)):
            path_length[i] = np.count_nonzero(p[i] == 0)
        memory = np.asarray([[None if any(i is None for i in p[:, x, y]) else float('inf') if all(i == float('inf') for i in p[:, x, y])
                              else 0 if p[path_length.index(min(path_length)), x, y] == 0 else -1 if all(i == -1 for i in p[:, x, y]) else 1
                              for y in range(p[0].shape[1])] for x in range(p[0].shape[0])], np.ndarray)
        return any(n), memory
    
    @staticmethod
    def __search_from(graph, search_area, actual_node=None):
        if actual_node is None:
            actual_node = (graph.shape[0] - 1, 0)
        if graph[actual_node] is None:
            return False
        if graph[actual_node] != -1:
            return False
        if actual_node == (0, graph.shape[1] - 1):
            graph[actual_node] = 0
            return True
        graph[actual_node] = 1
        found = []
        for i in search_area:
            tmp_node = tuple(map(lambda x, y: x + y, actual_node, i))
            if -1 < tmp_node[0] < graph.shape[0] and -1 < tmp_node[1] < graph.shape[1]:
                found_tmp = Search.__search_from(graph, search_area, tmp_node)
                found.append(found_tmp)
        if any(found):
            graph[actual_node] = 0
        else:
            graph[actual_node] = float('inf')
        return any(found)


class Interpretation:
    @staticmethod
    def color_ways(labyrinth):
        graph = np.asarray([[None if x == 0 else -1 for x in y] for y in labyrinth], np.ndarray)
        n, memory = Search.start_search(graph)
        colors = np.ndarray((memory.shape[0], memory.shape[1], 3))
        for x in range(memory.shape[0]):
            for y in range(memory.shape[1]):
                if memory[x, y] == -1:
                    colors[x, y, :] = [1, 1, 1]
                elif memory[x, y] is None:
                    colors[x, y, :] = [0, 0, 0]
                elif memory[x, y] == float("inf"):
                    colors[x, y, :] = [1, 0, 0]
                else:
                    colors[x, y, :] = [0, 1 - memory[x, y], memory[x, y]]
        return colors

    @staticmethod
    def color_ways_solved(labyrinth_solved):
        colors = np.ndarray((labyrinth_solved.shape[0], labyrinth_solved.shape[1], 3))
        for x in range(labyrinth_solved.shape[0]):
            for y in range(labyrinth_solved.shape[1]):
                if labyrinth_solved[x, y] == 0:
                    colors[x, y, :] = [1, 1, 1]
                elif labyrinth_solved[x, y] == 1:
                    colors[x, y, :] = [0, 0, 0]
                elif labyrinth_solved[x, y] == float("inf"):
                    colors[x, y, :] = [1, 0, 0]
                else:
                    colors[x, y, :] = [0, 1 - labyrinth_solved[x, y], labyrinth_solved[x, y]]
        return colors
    
    @staticmethod
    def evaluate(labyrinth) -> float:
        graph = np.asarray([[None if x == 0 else -1 for x in y] for y in labyrinth], np.ndarray)
        n, memory = Search.start_search(graph)
        path_size = np.count_nonzero(memory == -1) + np.count_nonzero(memory == float('inf')) + np.count_nonzero(memory == 0) + np.count_nonzero(memory == 1)
        if not n:
            return (np.count_nonzero(memory == float('inf'))/path_size)/5
        return (((1-np.count_nonzero(memory == -1)/path_size) + (1-np.count_nonzero(memory == 1)/path_size)
                + (np.count_nonzero(memory == float('inf'))/path_size) + (np.count_nonzero(memory == 0)/path_size)))/3
