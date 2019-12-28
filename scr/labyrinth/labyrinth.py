import numpy as np  # Mathematik
import copy  # kopieren ohne Überschreibungen
import os  # erstellen von Dateipfaden
from tkinter import *  # erstellen von GUIs
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # erstellen von Diagrammen in GUIs
from matplotlib.figure import Figure  # erstellen von Diagrammen in GUIs
import matplotlib.image as image_saver  # speichern von Bildern
import sys  # einstellen des Rekursionslimits
import datetime  # akktuelles Datum und Uhrzeit


class Population:  # der Genetische Algorithmus
    def __init__(self, population_size, filters: tuple, input_depth=4, batch_size=1):  # Konstruktor
        # initialieren von Objektvariabelen
        self.filters = filters
        self.population = []
        self.input_depth = input_depth
        self.batch_size = batch_size
        # die 0. Generation wird mit Netzwerken gefüllt
        for i in range(population_size):
            self.population.append(Generator(self.input_depth, self.filters))
    
    def save_population(self, file, generation, fitness):  # speichern der aktuellen Generation
        # erstellen des Dateipfades
        new_path = "populations/" + file
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        # speichern der Metavariabelen
        np.save(new_path + "/info", np.array([len(self.population), self.input_depth, generation]))
        np.save(new_path + "/filters", np.array(self.filters))
        np.save(new_path + "/fitness", np.array(fitness))
        for i in range(len(self.population)):  # speichern der einzellen Netzwerke
            self.population[i].save_weight(new_path + "/" + str(i))
    
    def load(self, file):  # laden einer gespeicherten Generation
        # laden des Dateipfades und speichern der Metavariablen
        new_path = "populations/" + file
        meta = np.load(new_path + "/info.npy")
        self.input_depth = meta[1]
        if len(self.population) != meta[0]:  # wenn die vorherige Population nicht gleich groß war wie die jetzige
            # anpassen der Populationsparameter
            self.filters = np.load(new_path + "/filters.npy")
            self.population = []
            for i in range(meta[0]):
                self.population.append(Generator(self.input_depth, self.filters))
        for i in range(meta[0]):  # einfügen der neuen Generation
            self.population[i].load_weight(new_path + "/" + str(i) + ".npy")
        return meta[2], list(np.load(new_path + "/fitness.npy"))  # weitergabe von informationen an die GUI
    
    @staticmethod
    def __pick_one(arr, probability):  # wählt aus einem Iterierbaren Objekt ein zufälliges aus (abhängig von mitgegebenen Wahrscheinlichkeiten)
        pro = [float(i) / sum(probability) for i in probability]  # skaliert die Wahrscheinlichkeit so, dass sie addiert 1 ergeben
        index = 0
        r = np.random.rand()  # erzeugt eine Zufallszahl zwischen 0 und 1
        # zieht die mitgegebenen Wahrscheinlichkeiten von der Zufallszahl ab, bis 0 erreicht wurde
        while r > 0:
            r = r - pro[index]
            index += 1
        index -= 1  # geht einen Schritt zurück um das richtige Obkjekt auszuwählen
        return copy.deepcopy(arr[index])  # gibt das ausgewählte Objekt zurück
    
    @staticmethod
    def similarity(result1: np.ndarray, result2: np.ndarray):  # gibt bei zwei gleichgroßen binären Matrizen die Ähnlichkeit an
        score = 0
        # iteriert horizontal und vertikal durch die Matrizen
        for x in range(result1.shape[0]):
            for y in range(result1.shape[1]):
                # wenn ein Punkt in beiden Matrizen gleich ist wird dem Score ein  Punkt hinzugefügt
                if result2[x, y] == result1[x, y]:
                    score += 1
        return score / result1.size  # die Ähnlichkeit in Prozent wird zurückgegeben
    
    def train(self, input_size: tuple, evaluator, translator, mutation_chance: float, inputs_list: np.ndarray = None):  # trainiert die KI
        if inputs_list is None:  # erzeugt eine Eingangsliste wenn keine vorhanden ist
            inputs_list = np.random.rand(self.batch_size, input_size[0], input_size[1], self.input_depth)
        elif len(inputs_list.shape) == 3:  # erzeugt eine neue Eingangsliste wenn die Vorhandene nicht im Batch-Format ist
            inputs_list = np.random.rand(self.batch_size, input_size[0], input_size[1], self.input_depth)
        inputs_similar = np.random.rand(self.batch_size, input_size[0], input_size[1], self.input_depth)  # erzeugt eine Eingangsliste für das Vergleichslabyrinth
        # iteriert durch die Population und nimmt von allen Netzwerken die Fitness auf
        fitness = []
        for i in range(len(self.population)):
            # iteriert durch den Batch und nimmt von allen Eingangsdaten die Fitness auf
            f_tmp = []
            for j in range(self.batch_size):
                out = translator(self.population[i].query_convolution(inputs_list[j]))  # lässt ein Labyrinth erzeugen
                similarity = self.similarity(out, translator(self.population[i].query_convolution(inputs_similar[j])))  # berchnet die Ähnlichkeit zwischen zwei Labyrinthen
                f_tmp.append(evaluator(out) - (0 if similarity < 0.85 else 0.65 * similarity))  # berschnet die die Fitness des aktuellen Labyrinths
            fitness.append(np.mean(f_tmp))  # speichert den Durchschnitt des Batches
        old_population = copy.deepcopy(self.population)  # copiert die aktuelle Generation
        self.population[0] = copy.deepcopy(old_population[fitness.index(max(fitness))])  # übernimmt den besten der letzten Generation ohne mutation
        for i in range(1, len(self.population)):  # ersetzt den Rest der Population und mutiert diesen
            self.population[i] = self.__pick_one(old_population, fitness)
            self.population[i].mutate(mutation_chance)
        return max(fitness), fitness.index(max(fitness))  # gibt die höchste Fitness und den Index der höchsten Fitness an die GUI weiter
    
    def query_batch(self, input_size: tuple, fitness_function, inputs_list: np.ndarray = None):  # erzeugt Ergebnisse für einen gesamten Batch
        if inputs_list is None:  # erzeugt eine Eingangsliste wenn keine vorhanden ist
            inputs_list = np.random.rand(self.batch_size, input_size[0], input_size[1], self.input_depth)
        results = []
        fitness = []
        # iteriert durch die gesamte Population (ForEach-Schleife) und durch den Batch
        for i in self.population:
            for j in range(self.batch_size):
                results.append(i.query_convolution(inputs_list[j]))  # fügt das Ergebnislabyrinth hinzu
                fitness.append(fitness_function(results[-1]))  # fügt die Fitness des Labyrinths am Ende der Liste hinzu
        return results[fitness.index(max(fitness))]  # gibt das Labyrinth mit der besten Wertung zurück
    
    def query(self, input_size: tuple, fitness_function, inputs_list: np.ndarray = None):  # erzeugt Ergebnisse für die gesamte Population (vgl. query_batch)
        if inputs_list is None:
            inputs_list = np.random.rand(input_size[0], input_size[1], self.input_depth)
        results = []
        fitness = []
        for i in self.population:
            results.append(i.query_convolution(inputs_list))
            fitness.append(fitness_function(results[-1]))
        return results[fitness.index(max(fitness))]
    
    def query_fast(self, input_size: tuple):  # erzeugt Ergebnisse für das erste Netzwerk
        return self.population[0].query_convolution(np.random.rand(input_size[0], input_size[1], self.input_depth))


class Generator:  # Das Generator Netzwerk
    @staticmethod
    def ReLU(x):  # die rectified linear unit Aktivierungsfunktion
        if isinstance(x, list):  # wenn x eine Liste ist
            return [x if i > 0 else 0 for i in x]
        if isinstance(x, np.ndarray):  # wenn x ein numpy ndarray ist
            if len(x.shape) == 1:  # wenn x die Dimension 1 hat
                return np.asarray([x if i > 0 else 0 for i in x], np.ndarray)
            if len(x.shape) == 2:  # wenn x die Dimension 2 hat
                return np.asarray([[x if i > 0 else 0 for i in j] for j in x], np.ndarray)
            if len(x.shape) == 3:  # wenn x die Dimension 3 hat
                return np.asarray([[[x if i > 0 else 0 for i in j] for j in k] for k in x], np.ndarray)
            raise Exception("too many dimensions")
        return x if x > 0 else 0  # gib x zurück wenn x größer als 0 ist. Sonst gib 0 zurück
    
    @staticmethod
    def sigmoid(x):  # die Sigmoid-Aktivierungsfunktion
        return 1 / (1 + np.exp(-x))  # 1 durch 1 + e hoch minus x
    
    def __init__(self, input_depth: int, filters: tuple, expand: bool = False):  # der Konstruktor
        # Beschreibung der Ansteuerung von den ndarrays filters (convolution) und weights (""" """ ist ein mehrzeiliger String, wird hier aber zum kommentieren genutzt)
        """filters: [layer][x_length, y_length, z_length, stride]"""
        """weights: [layer][filter, x, y, depth]"""
        self.weight = []
        if expand:  # weights for expanding (eine andere Art von Generatoren die ich genutzt habe)
            for i in range(len(filters)):
                self.weight.append([])
                if i == 0:
                    for j in range(filters[i][2]):
                        self.weight[i].append(np.random.normal(0.0, 0.02, (filters[i][0], filters[i][1], input_depth)))
                else:
                    for j in range(filters[i][2]):
                        self.weight[i].append(np.random.normal(0.0, 0.02, (filters[i][0], filters[i][1], filters[i-1][2])))
        else:  # weights for convolutions (die Art von Generatoren die gerade genutzt wird)
            self.stride = []
            for i in range(len(filters)):  # iteriert durch alle Layer
                if i != 0:  # wenn es gerade nicht die erste Ebene ist, dann füge an self.weight eine normalverteilte Matrix (my = 0, sigma = 0.2)
                    # und der Form z_length * x_length * y_length * z_length_von_der_vorherigen_Matrix an
                    self.weight.append(np.random.normal(0.0, 0.2, (filters[i][2], filters[i][0], filters[i][1], filters[i - 1][2])))
                else:  # sonst tu das selbe nur nimm die Form _length * x_length * y_length * input_depth
                    self.weight.append(np.random.normal(0.0, 0.2, (filters[i][2], filters[i][0], filters[i][1], input_depth)))
                try:  # versuche
                    self.stride.append(filters[i][3])  # an self.stride den vierten Wert der Filter anzuhängen
                except IndexError:  # wenn es dabei einen IndexError gibt
                    self.stride.append(0)  # dann hänge eine 0 an self.stride an
        self.activation_function = lambda x: np.tanh(x)  # wählt den Tangens hyperbolicus als Aktivierungsfunktion aus
    
    def save_weight(self, file):  # speichert die Weights an die angegebene Stelle
        np.save(file, self.weight)
    
    def load_weight(self, file):  # lädt die Weights von der angegebenen Stelle
        self.weight = np.load(file, allow_pickle=True)
    
    def query_expand(self, input_: np.ndarray):  # die Queryfunktion für expandierende Genratornetzwerke (veraltet)
        if input_.shape[2] != self.weight[0][0].shape[2]:  # wenn die tiefe des Inputs nicht stimmt gib einen Fehler aus
            raise ValueError('input has a false depth! Depth ' + str(input_.shape[2]) + ' instead of ' + str(self.weight[0][0].shape[2]))
        for i in range(len(self.weight)):  # für alle Ebenen
            input_ = self.multiply(input_, i).copy()  # ersetzte die Eingangsdaten gegen das Ergebniss der Multiply-Funktion
        return input_  # gib das Ergebniss zurück
    
    def multiply(self, input_: [np.ndarray, list], i: int):  # die Multiply-Funktion
        # erstelle eine Matrix voller Einsen mit vollgender Größe
        __output = np.ones((input_.shape[0] * self.weight[i][0].shape[0],
                            input_.shape[1] * self.weight[i][0].shape[1],
                            len(self.weight[i])))
        # speichere die x und y Länge um später schneller drauf zu greifen zu können
        w_x_len = self.weight[i][0].shape[0]
        w_y_len = self.weight[i][0].shape[1]
        # iteriere durch alle Pixel des Inputs und durch die tiefe der Filter
        for x in range(input_.shape[0]):
            for y in range(input_.shape[1]):
                for filter_ in range(len(self.weight[i])):
                    # ersetze die Matrix von x*länge_x bis x*länge_x + länge_x und y*länge_y bis y*länge_y + länge_y in der Ebene filter_
                    # gegen das Skalarprodukt von den Weights an der Stelle i, fiter_ mal denn input an der Stelle x, y
                    __output[x * w_x_len:(x * w_x_len) + w_x_len, y * w_y_len:(y * w_y_len) + w_y_len, filter_] = np.dot(self.weight[i][filter_], input_[x, y])
        return __output  # gib das Ergebnis zurück
    
    def query_convolution(self, __input: np.ndarray) -> np.ndarray:  # die Queryfunktion für convolutionäre Genratornetzwerke
        for i in range(len(self.weight)):  # iteriert durch alle Ebenen
            stride = self.stride[i]  # speichere den aktuellen Stride um schneller darauf zugreiden zu können
            __input = np.asarray([[__input[(x - stride) // (stride + 1), (y - stride) // (stride + 1), :]  # setzt den Schritt ein
                                   # wenn x bzw. y zum modulo des (Strides + 1) gleich dem Stride sind sonst wird diese Stelle in input mit nullen gefüllt
                                   if x % (stride + 1) == stride and y % (stride + 1) == stride else np.zeros(__input.shape[2])
                                   for y in range((stride * (__input.shape[1] + 1) + __input.shape[1] + 1) - 1)]  # iteriert y von 0 zu <Parameter von range>
                                  for x in range((stride * (__input.shape[0] + 1) + __input.shape[0] + 1) - 1)])  # iteriert x von 0 zu <Parameter von range>
            # initialisiert die Variable __output als 3D Matrix voller nullen
            __output = np.zeros((__input.shape[0] - (self.weight[i].shape[1] - 1), __input.shape[1] - (self.weight[i].shape[2] - 1), self.weight[i].shape[0]))
            # iteriert die Variablen x, y über die x-, y-Achse von __output
            for x in range(__output.shape[0]):
                for y in range(__output.shape[1]):
                    # ersetzt die aktuelle Stelle von __output gegen das Ergebnis von __convolve von i und __input an der Stelle x bzw. y bis x + weight_x_len, y + weight_y_len
                    __output[x, y, :] = self.__convolve(__input[x:x + self.weight[i].shape[1], y:y + self.weight[i].shape[2], :], i)
            __input = __output.copy()  # ersetzt die Variable __input gegen eine copy von __output
        return __input  # gib das Ergebnis zurück
    
    def __convolve(self, __input: np.ndarray, i: int):  # Die Funktion zum ausführen von einzelnen convolutionen
        # Zuerst werden __input und die Weights im i'ten Layer Elementweise Multipliziert
        # die np.transpose Funktion strukturiert das entstandene ndarray so um, dass die vormals erste Achse die letzte wird (x->w, y->x, z->y, w->z)
        # Zuletzt addiert die Sum-Funktion alle Achsen, bis auf die neue w-Achse auf. Danach wird ein Array mit der länge der w-Achse zurückgegeben.
        return np.transpose(__input * self.weight[i], (tuple(np.append(range(1, len(self.weight[i].shape)), 0)))).sum(tuple(range(len(self.weight[i].shape) - 1)))
    
    def mutate(self, mutation_rate):  # Die Funktion zum Mutieren der einzelnen Netzwerke
        for i in range(len(self.weight)):  # iteriert über alle Layer
            for j in range(len(self.weight[i])):  # iteriert über alle Filter
                # addiert die aktuelle Stelle in den Weights mit einer normalverteilten Zufallsgröße, wenn eine zufällig erzeugte Zahl kleiner als die Mutationsrate ist.
                self.weight[i][j] = np.array([[self.weight[i][j][x, y] + np.random.normal(0.0, 0.1)
                                               if np.random.rand() < mutation_rate else self.weight[i][j][x, y]
                                               for y in range(self.weight[i][j].shape[1])]
                                              for x in range(self.weight[i][j].shape[0])], np.ndarray)


class Search:  # die Klasse mit dem Labyrinth-Löse-Algorithmus
    @staticmethod
    def start_search(graph):  # Die Funktion zum Lösen eines Labyrinths
        # Die Startvariablen werden initialisiert
        p = []
        n = []
        # search_area gibt an in welche Richtungen alles gesucht wird. Hier ist das Gerade und Schräg in alle Richtungen in einser Schritten
        search_area = [(0, 1), (-1, 0), (1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        # tauscht die Reihenfolge von search_area bis jede mögliche Reihenfolge einmal existierte
        for i in range(len(search_area)):
            for j in range(len(search_area) - 1):
                for k in range(len(search_area) - 2):
                    graph_tmp = np.copy(graph)  # kopiert den Graphen, so dass der Ursprungsgraph nicht überschrieben wird
                    n.append(Search.__search_from(graph_tmp, search_area))  # hängt das Ergebnis der Suche an n an
                    p.append(graph_tmp)  # hängt den Ergebnissgraph von der Suche an p an
                    # beginnt mit dem Tauschen der Suchgegenden
                    a, b = search_area[3], search_area[2]
                    search_area[3], search_area[2] = b, a
                search_tmp = search_area[0:]
                search_area = search_area[:0] + search_tmp[-1:] + search_tmp[:-1]  # (Listen addition = append)
            search_area = search_area[-1:] + search_area[:-1]
        p = np.asarray(p, np.ndarray)  # macht p zu einem ndarray
        path_length = [float('inf')] * len(p)  # erstellt eine Liste mit der Länge von p, die mit unendlichen gefüllt ist
        for i in range(len(p)):  # iteriert über p
            path_length[i] = np.count_nonzero(p[i] == 0)  # fügt die Anzahl der Nullen vom aktuellen Labyrinth in path_length ein
        # setzt dort ein None (Wand) ein, wo eines der Labyrinthe eine Wand hat, dort eine unendlich (Sackgasse) wo alle Labyrinthe keinen Weg gefunden haben,
        # dort eine 0 (kürzester Weg) wo das Labyrinth mit der kürzeseten Lösung eine 0 hat, dort eine -1 (nicht erreichter Weg) wo alle Labyrinthe eine -1 haben
        # und dort eine 1 (alternativer Weg) wo keines davon gilt. Das wird über alle x, y des Labyrinthes iteriert und danach zum ndarray gemacht
        memory = np.asarray([[None if any(i is None for i in p[:, x, y]) else float('inf') if all(i == float('inf') for i in p[:, x, y])
                              else 0 if p[path_length.index(min(path_length)), x, y] == 0 else -1 if all(i == -1 for i in p[:, x, y]) else 1
                              for y in range(p[0].shape[1])] for x in range(p[0].shape[0])], np.ndarray)
        return any(n), memory  # gibt zurück ob das Labyrinth lösbar ist und wie die Lösung aussieht
    
    @staticmethod
    def __search_from(graph, search_area, actual_node=None):  # der Rekursive Teil des Suchalgorithmuses
        if actual_node is None:  # falls keine aktuelle Node existiert wird mit der unten Links gestartet
            actual_node = (graph.shape[0] - 1, 0)
        if graph[actual_node] is None:  # wenn die aktuelle Node in einer Mauer drinnen ist gib Falsch zurück
            return False
        if graph[actual_node] != -1:  # wenn die aktuelle Node kein unbenutzter Weg ist fig Falsch zurück
            return False
        if actual_node == (0, graph.shape[1] - 1):  # wenn die aktuelle Node die obere linke Ecke ist setzte den graphen dort 0 (Weg zum Ziel) und gib True zurück
            graph[actual_node] = 0
            return True
        graph[actual_node] = 1  # setzte die aktuelle Node gleich 1 (benutzter Weg)
        found = []  # erzeuge eine neue Liste
        for i in search_area:  # iteriere durch die searchareas
            tmp_node = tuple(map(lambda x, y: x + y, actual_node, i))  # erzeugt den Tuple tmp_node, die die aktulle Node + die aktuelle Search_area ist
            if -1 < tmp_node[0] < graph.shape[0] and -1 < tmp_node[1] < graph.shape[1]:  # wenn die tmp_node noch innérhalb des Labyrinthes ist, dann
                # hänge an found das Ergebniss des Rekursiven Ausrufs an
                found_tmp = Search.__search_from(graph, search_area, tmp_node)
                found.append(found_tmp)
        if any(found):  # wenn einer der gefundenen Wege zum Ziel geführt hat, dann setzte die aktuelle Stelle gleicb 0 (Weg zum Ziel)
            graph[actual_node] = 0
        else:  # wenn nicht dann setzte die aktuelle Position gleich unendlich (Sackgasse)
            graph[actual_node] = float('inf')
        return any(found)  # gib zurück ob einer der Wege ins Ziel geführt hat (Durch Pointer wird graph auch zurückgegeben)


class Interpretation:  # Die Klasse zum Interpretieren der Labyrinthe
    @staticmethod
    def color_ways(labyrinth):  # Die Methode, die die Labyrinthe in Farbe übersetzt
        graph = np.asarray([[None if x == 0 else -1 for x in y] for y in labyrinth], np.ndarray)  # übersetzt ein 1, 0 Labyrinth in ein None, -1 Labyrinth
        n, memory = Search.start_search(graph)  # lässt das Labyrinth lösen
        return Interpretation.color_ways_solved(memory)  # übersetzt es in Farbe
    
    @staticmethod
    def color_ways_solved(memory):  # Die Methode, die ein schon gelöstes Labyrinth in Farbe übersetzt
        colors = np.ndarray((memory.shape[0], memory.shape[1], 3))  # erzeugt eine Matrix mit der selben x und y Länge wie das Labyrinth aber einer Tiefe von 3(rgb)
        # iteriert durch das Labyrinth
        for x in range(memory.shape[0]):
            for y in range(memory.shape[1]):
                if memory[x, y] == -1:  # wenn die aktuelle Stelle gleich -1 (nicht erreicht) ist gib ihr die Farbe Weiß
                    colors[x, y, :] = [1, 1, 1]
                elif memory[x, y] is None:  # wenn die akktuelle Stelle nichts (Mauer) ist gib ihr die Farbe Schwarz
                    colors[x, y, :] = [0, 0, 0]
                elif memory[x, y] == float("inf"):  # wenn die akktuelle Stelle gleich unendlich (Sackgasse) ist gib ihr die Rot
                    colors[x, y, :] = [1, 0, 0]
                else:  # Ansonsten gib ihr einen Blaugrün Ton (1 (alternativer Weg) ist dann Blau und 0 (kürzester Weg) ist dann Grün)
                    colors[x, y, :] = [0, 1 - memory[x, y], memory[x, y]]
        return colors  # gib das gefärbte Labyrinth zurück
    
    @staticmethod
    def evaluate(labyrinth) -> float:  # bewertet das Labyrinth mit einer Gleitkommazahl
        graph = np.asarray([[None if x == 0 else -1 for x in y] for y in labyrinth], np.ndarray)  # übersetzt ein 1, 0 Labyrinth in ein None, -1 Labyrinth
        n, memory = Search.start_search(graph)  # lässt das Labyrinth lösen
        # zählt wieviele Stellen es gibt, die keine Mauer haben
        path_size = np.count_nonzero(memory == -1) + np.count_nonzero(memory == float('inf')) + np.count_nonzero(memory == 0) + np.count_nonzero(memory == 1)
        if not n:  # wenn das Labyrinth nicht gelöst werden konnte
            return (np.count_nonzero(memory == float('inf')) / path_size) / 5  # teile die Anzahl der Sackgassen durch path_size und durch 5. Gib das Ergebnis zurück
        # gib Punkte für alle Pixel, die eine Sackgasse, der kürzeste Weg, kein unerreichter Pixel, oder kein alternativer Weg sind
        # Teile Das Ergebniss durch path_size und anschließend durch 3 und gib das zurück
        return (((1 - np.count_nonzero(memory == -1) / path_size) + (1 - np.count_nonzero(memory == 1) / path_size)
                 + (np.count_nonzero(memory == float('inf')) / path_size) + (np.count_nonzero(memory == 0) / path_size))) / 3


class GUI:  # Die Klasse mit der Grafischen Benutzer Oberfläche
    def __init__(self):  # der Konstruktor
        sys.setrecursionlimit(16000)  # erhöht das Rekursionslimit um den Löseallgorithmus zu ermöglichen
        self.input_depth = 100  # setzt die Tiefe der ersten Ebene
        """self.generator = Population(10, ((4, 4, 256), (2, 2, 128), (2, 2, 64), (2, 2, 2)), self.input_depth) """  # alternative Eingaben für expandierenden Generator
        # Erzeugt eine neue Population mit 50 Netzwerken, die eine Eingangstiefe von 100 über drei Schichten in eine Matrix mit der Tiefe 2 umskalieren
        self.generator = Population(50, ((6, 6, 256, 4), (2, 2, 64, 1), (3, 3, 2, 2)), input_depth=self.input_depth, batch_size=100)  # Eingaben für convolutioneren Generator
        self.is_training = False  # erzeugt eine Variable um zu überprüfenob gerade trainiert wird
        self.epochs = 2  # Die Epoche pro Trainingseinheit
        self.fitness_total = []  # eine Liste mit der Maximalen Fitness einer jeden Generation
        self.generation = 0  # aktuelle Generation startet bei 0
        self.best_index = 0  # die, die das linke Labyrinth erzeugt hat
        self.input = np.random.rand(1, 1, self.input_depth)  # input für den Fall von globalen Inputs
        
        self.plots = Tk()  # Das Fenster mit den Ergebnissen
        self.settings = Tk()  # das Fenster mit den Knöpfen
        
        self.l_fitness = Label(master=self.plots, text="Index = 0; fitness = 0")  # Der Text unter den Bildern
        self.l_fitness.pack(side='bottom')  # wird unter den am unteren Teil des Ergebnisfensters platziert
        
        # Erzeugt das Diagramm mit den Fitnesswerten
        fig_f = Figure()
        self.ax_f = fig_f.add_subplot(111)
        self.fitness_c = FigureCanvasTkAgg(fig_f, master=self.plots)
        self.fitness_c.get_tk_widget().pack()

        # Erzeugt das Diagramm, in dem das linke Labyrinth angezeigt wird
        fig_r1 = Figure()
        self.ax_r1 = fig_r1.add_subplot(111)
        self.result_c1 = FigureCanvasTkAgg(fig_r1, master=self.plots)
        self.result_c1.get_tk_widget().pack(side='left')

        # Erzeugt das Diagramm, in dem das rechte Labyrinth angezeigt wird
        fig_r2 = Figure()
        self.ax_r2 = fig_r2.add_subplot(111)
        self.result_c2 = FigureCanvasTkAgg(fig_r2, master=self.plots)
        self.result_c2.get_tk_widget().pack(side='right')
        
        self.same_inputs = BooleanVar(self.settings, value=False)  # erzeugt eine boolsche Variable, welche die Inputs über die Anzahl der Epochen pro Training gleich hält
        self.global_inputs = BooleanVar(self.settings, value=False)  # erzeugt eine boolsche Variable, welche die Inputs immer gleich hält
        Checkbutton(self.settings, text="keep the inputs", variable=self.same_inputs).pack()  # erzeugt eine Checkbox um same_inputs zu steuern
        Checkbutton(self.settings, text="keep the global inputs", variable=self.global_inputs).pack()  # erzeugt eine Checkbox um global_inputs zu steuern
        Button(self.settings, text="Start/Stop the training", command=self.toggle_training).pack()  # erzeugt einen Knopf um das Training zu starten und zu stoppen
        Button(self.settings, text="train once", command=self.train_once).pack()  # erzeugt einen Knopf um die KI einmal um die Anzahl der Epochen zu trainieren
        self.l_status = Label(self.settings, text="not training", fg='red')  # erzeugt einen Text, der sagt, was die KI gerade macht
        self.l_status.pack()  # fügt den Text in die GUI ein
        Button(self.settings, text="query", command=self.query).pack()  # erzeugt einen Knopf, der die Population zwei neue Ergebnisse erstellen lässt
        Button(self.settings, text="query batch", command=self.query_batch).pack()  # erzeugt einen Knopf, der die Population über einen Batch zwei neue Ergebnisse erstellen lässt
        Button(self.settings, text="fast query", command=self.query_fast).pack()  # erzeugt einen Knopf, der das erste Netzwerk der Population zwei neue Labyrinthe erstellen lässt
        Button(self.settings, text="query global input", command=self.query_global).pack()  # erzeugt einen Knopf, der die KI ein Labyrinth mit den globalen inputs erstellen lässt
        Label(self.settings, text="epochs:").pack()  # erstellt die Überschrift über der Epochentextbox
        self.t_epochs = Entry(self.settings)  # erstellt eine Textbox
        self.t_epochs.insert(END, '2')  # hängt am Ende der textbox eine 2 an = füllt die leere Textbox mit 2
        self.t_epochs.pack()  # fügt die Textbox in die GUI ein
        self.l_generation = Label(self.settings, text="Generation: 0", fg='red')  # fügt den Text ein indem die Anzahl der bisherigen Generationen steht
        self.l_generation.pack()  # fügt den Text zur GUI hinzu
        Label(self.settings, text="save / load file:").pack()  # erstellt die Überschrift für die Textbox, in der der Speicher- und Ladeort  der KI steht
        self.t_save_load = Entry(self.settings)  # erstellt die oben erwähnte Textbox
        self.t_save_load.pack()  # fügt die Textbox der GUI hinzu
        Button(self.settings, text="save", command=self.save_population).pack()  # erstellt den Knopf zum speichern der aktuellen Population
        Button(self.settings, text="load", command=self.load_population).pack()  # erstellt den Knopf zum Laden einer gespeicherten Population
        Button(self.settings, text="save left labyrinth", command=lambda: self.save_labyrinth("l")).pack()  # erstellt den Knopf zum speichern des linken Labyrinths
        Button(self.settings, text="save right labyrinth", command=lambda: self.save_labyrinth("r")).pack()  # erstellt den Knopf zum speichern des rechten Labyrinths
        
        self.open = True  # initialiseiert open und deklariert es als Wahr
        self.plots.protocol("WM_DELETE_WINDOW", self.on_close)  # wenn settings geschlossen wird wird die Methode on_close aufgerufen
        self.settings.protocol("WM_DELETE_WINDOW", self.on_close)  # wenn settings geschlossen wird wird die Methode on_close aufgerufen
        
        self.right = None  # initialisiert das rechte Labyrinth
        self.left = None  # initialisiert das linke Labyrinth
        
        self.run()  # startet die run-Funktion und erzeugt damit die ersten beiden Labyrinthe
    
    def on_close(self):  # Diese Methode wird gerufen, wenn die GUI geschlossen wird
        self.is_training = False  # beendet das training
        self.open = False  # beendet die Schleife in run
    
    def save_population(self):  # speichert die aktuelle Population
        self.l_status.config(text="saving", fg='orange')  # zeigt das Speichern in der GUI an
        self.settings.update()  # sorgt dafür, dass das Speichern angezeigt wird
        self.generator.save_population(self.t_save_load.get(), self.generation, self.fitness_total.copy())  # gib den Speicherbefehl an die Population weiter
    
    def load_population(self):  # lädt eine gespeicherte Population
        self.l_status.config(text="loading", fg='orange')  # zeigt das Laden an
        self.settings.update()  # sorgt dafür, dass das Laden angezeigt wird
        out = self.generator.load(self.t_save_load.get())  # gibt den Ladebefehl an die Population weiter
        self.generation = out[0]  # übernimmt die neue aktuelle Generation
        self.fitness_total = out[1].copy()  # übernimmt die neue Fitness
        self.plot_fitness()  # zeigt die neue Fitness im Diagramm an
        self.query()  # erzeugt zwei neue Labyrinthe
    
    def toggle_training(self):  # wechselt zwischen nicht trainieren und trainieren
        self.is_training = not self.is_training  # invertiert den Wert von is_training
        self.l_status.config(text="ending training", fg='orange')  # zeigt das Enden des Trainings an (wird überschrieben, wenn das Training gerade erst begonnen hat)
        self.settings.update()  # sorgt dafür, dass das Ende desTrainings angezeigt wird
    
    def train_once(self, long_time: bool = False):  # trainiert die KI einmal
        if long_time:  # wenn das Training ein Langzeittraining ist (togggle training) wird das angezeigt
            self.l_status.config(text="training", fg='green')
        else:  # wenn das Training ein Kurzzeittraining ist (train once) wird das angezeigt
            self.l_status.config(text="ending training", fg='orange')
        __input = np.random.rand(1, 1, self.input_depth)  # erzeugt einen neuen Input mit der Dimension 1*1*input_depth
        for j in range(self.epochs):  # Es wird solange trainiert wie eingestellt wurde
            self.settings.update()  # das Einstellungsfenster wird aktualisiert, damit die GUI ansteuerbar bleibt
            if self.same_inputs or self.global_inputs:  # wenn der input gleich bleiben soll
                # trainiert die KI mit dem input und sichert die ergebene Fitness und den ergebenen Index. lambda-Funktionen sind eine Art einzeiler Funktionen zu erstellen
                fitness, index = self.generator.train((1, 1), lambda x: Interpretation.evaluate(x), lambda x: train.translate_gan(x), 0.001, __input)
                self.fitness_total.append(fitness)  # fügt die erreichte Fitness in die Fitnessliste ein
            else:
                # trainiert die KI und sichert die ergebene Fitness und den ergebenen Index. lambda-Funktionen sind eine Art einzeiler Funktionen zu erstellen
                fitness, index = self.generator.train((1, 1), lambda x: Interpretation.evaluate(x), lambda x: train.translate_gan(x), 0.001)
                self.fitness_total.append(fitness)  # fügt die erreichte Fitness in die Fitnessliste ein
            self.best_index = index  # ersetzt die Klassenvariable best_index gegen den sich ergebenen Index
            self.generation += 1  # erhöht die aktuelle Generation um 1
            self.l_generation.config(text="Generation = " + str(self.generation), fg='blue')  # zeigt die aktuelle Generation an
            self.settings.update()  # das Einstellungsfenster wird aktualisiert, damit die GUI ansteuerbar bleibt und damit die neue Generation angezeigt wird
        self.query()  # erzeugt neue Labyrinthe mit der neuen Generation
        self.plot_fitness()  # zeigt die neue Fitnesskurve an
    
    def query_global(self):  # Die Methode zum erstellen von Labyrinthen mit globalem input
        self.query(self.input)  # ruft die query methode mit dem globalen input auf
    
    def query(self, inputs_list: np.ndarray = None):  # erstellt Labyrinthe und zeigt sie auf der GUI
        self.l_status.config(text="querying", fg='orange')  # zeigt an, dass gerade neue Labyrinthe erzeugt werden
        self.settings.update()  # sorgt dafür, dass der aktuelle Status angezeigt wird
        if inputs_list is None:  # wenn kein input gegeben wurde erzeugt die Methode zwei Labyrinthe mit einem zufälligem input
            labyrinth1 = train.get_gan_results(self.generator)
            labyrinth2 = train.get_gan_results(self.generator)
        else:  # wenn ein input gegeben wurde erzeugt die KI ein Labyrinth mit eben jenem input und eines mit einem zufälligem input
            labyrinth1 = train.get_gan_results(self.generator, inputs_list)
            labyrinth2 = train.get_gan_results(self.generator)
        # lässt die neuen Labyrinthe anzeigen, gibt auch den Score des ersten Labyrinthes mit
        self.plot_result((Interpretation.color_ways(labyrinth1), Interpretation.color_ways(labyrinth2)), Interpretation.evaluate(labyrinth1))
    
    def query_batch(self, inputs_list: np.ndarray = None):  # erstellt einen Batch von Labyrinthen und zeigt das beste auf der GUI an
        self.l_status.config(text="querying", fg='orange')  # zeigt an, dass gerade neue Labyrinthe erzeugt werden
        self.settings.update()  # sorgt dafür, dass der aktuelle Status angezeigt wird
        if inputs_list is None:  # wenn kein input gegeben wurde erzeugt die Methode zwei Labyrinthe mit einem zufälligem input
            labyrinth1 = train.get_gan_results_batch(self.generator)
            labyrinth2 = train.get_gan_results_batch(self.generator)
        else:  # wenn ein input gegeben wurde erzeugt die KI ein Labyrinth mit eben jenem input und eines mit einem zufälligem input
            labyrinth1 = train.get_gan_results_batch(self.generator, inputs_list)
            labyrinth2 = train.get_gan_results_batch(self.generator)
        # lässt die neuen Labyrinthe anzeigen, gibt auch den Score des ersten Labyrinthes mit
        self.plot_result((Interpretation.color_ways(labyrinth1), Interpretation.color_ways(labyrinth2)), Interpretation.evaluate(labyrinth1))
    
    def query_fast(self):  # erstellt ein Labyrinth vom ersten Netzwerk der aktuellen Generation
        self.l_status.config(text="querying", fg='orange')  # zeigt an, dass gerade neue Labyrinthe erzeugt werden
        self.settings.update()  # sorgt dafür, dass der aktuelle Status angezeigt wird
        # estellt zwei Labyrinthe mit zufälligen inputs über die schnelle Methode
        labyrinth1 = train.get_gan_results_fast(self.generator)
        labyrinth2 = train.get_gan_results_fast(self.generator)
        # lässt die neuen Labyrinthe anzeigen, gibt auch den Score des ersten Labyrinthes mit
        self.plot_result((Interpretation.color_ways(labyrinth1), Interpretation.color_ways(labyrinth2)), Interpretation.evaluate(labyrinth1))
    
    def run(self):  # die mainloop
        # erzeugt zwei neue Labyrinthe und zeigt sie in der GUI zusammmen mit der Fitness an
        labyrinth1 = train.get_gan_results(self.generator)
        labyrinth2 = train.get_gan_results(self.generator)
        self.plot_fitness()
        self.plot_result((Interpretation.color_ways(labyrinth1), Interpretation.color_ways(labyrinth2)), Interpretation.evaluate(labyrinth2))
        self.l_fitness.config(text="Index = " + str(self.best_index) + "; fitness = " + str(Interpretation.evaluate(labyrinth1)))
        while self.open:  # startet eine Schleife, die erst mit dem Schleißen des Programms enden wird
            # aktualisiert beide Fenster, damit die GUI ansteuerbar bleibt
            self.plots.update()
            self.settings.update()
            try:  # versucht die eingestellte Epoche als int zu bekommen
                self.epochs = int(self.t_epochs.get())
            except ValueError:  # wenn das nicht Funktioniert (kein int in der Textbox steht) wird das Training beendet
                self.is_training = False
            self.l_status.config(text="not training", fg='red')  # zeigt an, dass nicht Trainiert wird (wird überschrieben falls falsch)
            self.l_generation.config(text="Generation = " + str(self.generation), fg='blue')  # zeigt die aktuelle Generation an
            if self.is_training:  # wenn gerade Trainiert wird rufe die trainierfunktion mit einem Langzeittraining auf
                self.train_once(True)
    
    def plot_fitness(self):  # zeigt das Fitnessdiagramm auf der GUI an
        self.ax_f.plot(self.fitness_total, color='blue')  # plottet die Fitness
        self.fitness_c.draw()  # updatet den Canvas
        self.plots.update()  # updatet das Plotfenster
    
    def plot_result(self, result: [np.ndarray, tuple, list], fitness: float):  # plottet die Labyrinthe
        # teilt das result in die beiden Labyrinthe auf
        self.left = result[0]
        self.right = result[1]
        # übersetzt beide Labyrinthe und zeigt sie dann an
        self.ax_r1.imshow(train.translate_for_user(result[0]), origin='upper')
        self.ax_r2.imshow(train.translate_for_user(result[1]), origin='upper')
        # aktuallisiert den Text unter den Bildern
        self.l_fitness.config(text="Index = " + str(self.best_index) + "; fitness = " + str(fitness))
        # sorgt dafür, dass alles angezeigt wird
        self.result_c1.draw()
        self.result_c2.draw()
        self.plots.update()
    
    def save_labyrinth(self, position: str):  # speichert die Labyrinthe
        if position == "r":  # wenn das rechte Labyrinth gespeichert werden soll, speichere das gelöste und das ungelöste Labyrinth als PNG-Bilder mit der aktuellen Zeit als Namen
            image_saver.imsave("images/" + str(datetime.datetime.now()).replace(":", "-") + "-solved.png", train.translate_for_user(self.right))
            image_saver.imsave("images/" + str(datetime.datetime.now()).replace(":", "-") + "-blank.png", train.translate_for_user_bw(self.right), cmap='Greys_r')
        else:  # wenn das linke Labyrinth gespeichert werden soll, speichere das gelöste und das ungelöste Labyrinth als PNG-Bilder mit der aktuellen Zeit als Namen
            image_saver.imsave("images/" + str(datetime.datetime.now()).replace(":", "-") + "-solved.png", train.translate_for_user(self.left))
            image_saver.imsave("images/" + str(datetime.datetime.now()).replace(":", "-") + "-blank.png", train.translate_for_user_bw(self.left), cmap='Greys_r')


class train:  # Diese Klasse hat die Funktionen für das Training und für die Übersetztung der Labyrinthe
    @staticmethod
    def get_gan_results(__generator: Population, inputs_list: np.ndarray = None):  # gibt das binäre Labyrinth, dass aus dem Generator entstanden ist zurück
        if inputs_list is None:  # wenn kein input gegeben wurde
            # erzeugt die Ergebnisse des Generators von zufälligen Werten und übersetzt sie mit translate_gan
            return train.translate_gan(__generator.query((1, 1), lambda x: Interpretation.evaluate(train.translate_gan(x))))
        # erzeugt die Ergebnisse des Generators von den Eingangswerten und übersetzt sie mit translate_gan
        return train.translate_gan(__generator.query((1, 1), lambda x: Interpretation.evaluate(train.translate_gan(x)), inputs_list))
    
    @staticmethod
    def get_gan_results_fast(__generator: Population):  # gibt das binäre Labyrinth, dass aus dem ersten Generator entstanden ist zurück
        return train.translate_gan(__generator.query_fast((1, 1)))  # erzeugt die Ergebnisse des ersten Generators von zufälligen Werten und übersetzt sie mit translate_gan
    
    @staticmethod
    def get_gan_results_batch(__generator: Population, inputs_list: np.ndarray = None):  # gibt das binäre Labyrinth, dass aus dem Generator entstanden ist zurück
        if inputs_list is None:  # wenn kein input gegeben wurde
            # erzeugt die Ergebnisse des Generators von einem zufälligen Batch und übersetzt sie mit translate_gan
            return train.translate_gan(__generator.query_batch((1, 1), lambda x: Interpretation.evaluate(train.translate_gan(x))))
        # erzeugt die Ergebnisse des Generators von dem gegebenen Batch und übersetzt sie mit translate_gan
        return train.translate_gan(__generator.query_batch((1, 1), lambda x: Interpretation.evaluate(train.translate_gan(x)), inputs_list))
    
    @staticmethod
    def translate_for_user_bw(__labyrinth: np.ndarray):  # nimmt das übersetzte Labyrinth und entfernt die Lösung
        # übersetzt das Labyrinth mit translate_for_user und gibt dort eine 1 (keine Mauer) aus, wo mindestens einer der rgb-Werte == 1 ist, überall anders ist eine 0 (Mauer)
        return np.asarray(train.translate_for_user(__labyrinth)).any(2)
    
    @staticmethod
    def translate_for_user(__labyrinth: np.ndarray):  # übersetzt das Labyrinth so, dass die Mauern nur ein 1/3 so dick sind und die diagonalen Wege bessser nachvollziebar sind
        result = np.ones((__labyrinth.shape[0] * 3, __labyrinth.shape[1] * 3, 3))  # erzeugt eine Matrix mit der dreifachen Seiten Länge
        # iteriert x, y durch diese neue Matrix
        for x in range(__labyrinth.shape[0] * 3):
            for y in range(__labyrinth.shape[1] * 3):
                # wenn keiner der Farben an der Stelle der Teilung ohne Rest von (x, y durch 3) == 1 ist und die aktuelle Stelle mit dem Rest von 1 Teilbar durch drei ist, dann
                if not any(__labyrinth[x // 3, y // 3]) and (x % 3 == 1 and y % 3 == 1):
                    result[x, y, :] = [0, 0, 0]  # füge an dieser Stelle eine Wand ein
                else:  # ansonsten
                    # Füge an dieser Stelle die Werte der Teilung ohne Rest von (x, y durch 3) von dem Labyrinth ein, außer diese Stelle hat keine 1, dann füge die Farbe Weiß ein
                    result[x, y, :] = __labyrinth[x // 3, y // 3] if any(__labyrinth[x // 3, y // 3]) else [1, 1, 1]
        result = result[1:-1, 1:-1, :]  # entferne die äußersten Pixel
        tmp = result.copy()  # copiere das bisherige Ergebnis, um überschreibungen zu verhindern
        # iteriert x, y über die Länge, Breite des Ergebnisses
        # füllt nun die zwischen den Mauerpixeln entstandenen Lücken
        for x in range(tmp.shape[0]):
            for y in range(tmp.shape[1]):
                if all(tmp[x, y, :]):  # wenn der aktuelle Pixel weiß ist
                    # wenn (x, y) + (+2, +1, -1, -2) noch inerhalb der Grenzen von tmp ist und ein Teil (s.u.) beschriebenen Punkte schwarz
                    # ist oder ein anderer Teil (s.u.) nicht schwarz ist, dann
                    if (x + 2 < tmp.shape[0] and -1 < x - 1 and not (any(tmp[x + 2, y]) or any(tmp[x - 1, y]))) or \
                            (y + 2 < tmp.shape[1] and -1 < y - 1 and not (any(tmp[x, y + 2]) or any(tmp[x, y - 1]))) or \
                            (-1 < x - 2 and x + 1 < tmp.shape[0] and not (any(tmp[x - 2, y]) or any(tmp[x + 1, y]))) or \
                            (-1 < y - 2 and y + 1 < tmp.shape[1] and not (any(tmp[x, y - 2]) or any(tmp[x, y + 1]))):
                        result[x, y, :] = [0, 0, 0]  # mach den aktuellen Pixel schwarz
        # iteriert x, y über die Länge, Breite des Ergebnisses
        for x in range(tmp.shape[0]):
            for y in range(tmp.shape[1]):
                if all(tmp[x, y, :]):  # wenn der aktuelle Pixel weiß ist
                    # wenn (x, y) + (+1, -1) noch inerhalb der Grenzen von tmp ist und ein Teil (s.u.) beschriebenen Punkte schwarz
                    # ist oder ein anderer Teil (s.u.) nicht schwarz ist, dann
                    if (-1 < x - 1 and -1 < y - 1 and not (any(tmp[x - 1, y]) or any(tmp[x, y - 1]))) or \
                            (-1 < x - 1 and y + 1 < tmp.shape[1] and not (any(tmp[x - 1, y]) or any(tmp[x, y + 1]))) or \
                            (x + 1 < tmp.shape[0] and -1 < y - 1 and not (any(tmp[x + 1, y]) or any(tmp[x, y - 1]))) or \
                            (x + 1 < tmp.shape[0] and y + 1 < tmp.shape[1] and not (any(tmp[x + 1, y]) or any(tmp[x, y + 1]))):
                        result[x, y, :] = [0, 0, 0]  # mach den aktuellen Pixel schwarz
        return result  # gib das Ergebnis zurück
    
    @staticmethod
    def translate_gan(__labyrinth: np.ndarray):  # Diese Methode bringt das Ergabnis des Generators in ein binäres Labyrinth
        # wenn der erste Output größer ist, als der zweite wird keine Mauer gesetzt
        __labyrinth = np.asarray([[int(__labyrinth[x, y, 0] > __labyrinth[x, y, 1]) for y in range(__labyrinth.shape[1])] for x in range(__labyrinth.shape[0])], np.ndarray)
        __labyrinth[-1, 0] = 1  # verhindert eine Mauer in der unteren linken Ecke
        __labyrinth[0, -1] = 1  # verhindert eine Mauer in der oberen rechten Ecke
        return __labyrinth  # gibt das resultierende Labyrinth zurück


if __name__ == "__main__":  # Dieser Teil startet das Programm (if-Abfrage verhindert, dass das Programm ausversehen startet z.B. beim importieren)
    GUI()  # erzeugt ein Objekt der Klasse GUI
