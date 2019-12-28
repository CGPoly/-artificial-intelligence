from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.image as image_saver
import numpy as np
import sys
import datetime
# import matplotlib.pyplot as plt

import labyrinth.labyrinth_solver as labyrinth_solver
import labyrinth.gagan as gagan


class GUI:
    def __init__(self):
        sys.setrecursionlimit(16000)
        self.input_depth = 100
        # self.generator = gagan.Population(10, ((4, 4, 256), (2, 2, 128), (2, 2, 64), (2, 2, 2)), self.input_depth)  # for expanding
        self.generator = gagan.Population(50, ((6, 6, 256, 4), (2, 2, 64, 1), (3, 3, 2, 2)), input_depth=self.input_depth, batch_size=100)  # for convolution
        self.is_training = False
        self.epochs = 2
        self.fitness_total = []
        self.generation = 0
        self.best_index = 0
        self.input = np.random.rand(1, 1, self.input_depth)
        
        self.plots = Tk()
        self.settings = Tk()

        self.l_fitness = Label(master=self.plots, text="Index = 0; fitness = 0")
        self.l_fitness.pack(side='bottom')
        
        fig_f = Figure()
        self.ax_f = fig_f.add_subplot(111)
        # self.fitness, = self.ax_f.plot([], linestyle='-')
        self.fitness_c = FigureCanvasTkAgg(fig_f, master=self.plots)
        self.fitness_c.get_tk_widget().pack()

        fig_r1 = Figure()
        self.ax_r1 = fig_r1.add_subplot(111)
        # self.result = self.ax_r1.imshow([[[0, 0, 0]]])
        self.result_c1 = FigureCanvasTkAgg(fig_r1, master=self.plots)
        self.result_c1.get_tk_widget().pack(side='left')
        
        fig_r2 = Figure()
        self.ax_r2 = fig_r2.add_subplot(111)
        # self.result2 = self.ax_r2.imshow([[[0, 0, 0]]])
        self.result_c2 = FigureCanvasTkAgg(fig_r2, master=self.plots)
        self.result_c2.get_tk_widget().pack(side='right')

        self.same_inputs = BooleanVar(self.settings, value=False)
        self.global_inputs = BooleanVar(self.settings, value=False)
        Checkbutton(self.settings, text="behold the inputs", variable=self.same_inputs).pack()
        Checkbutton(self.settings, text="behold the global inputs", variable=self.global_inputs).pack()
        Button(self.settings, text="Start/Stop the training", command=self.toggle_training).pack()
        Button(self.settings, text="train once", command=self.train_once).pack()
        self.l_status = Label(self.settings, text="not training", fg='red')
        self.l_status.pack()
        Button(self.settings, text="query", command=self.query).pack()
        Button(self.settings, text="query batch", command=self.query_batch).pack()
        Button(self.settings, text="fast query", command=self.query_fast).pack()
        Button(self.settings, text="query global input", command=self.query_global).pack()
        Label(self.settings, text="epochs:").pack()
        self.t_epochs = Entry(self.settings)
        self.t_epochs.insert(END, '2')
        self.t_epochs.pack()
        self.l_generation = Label(self.settings, text="Generation: 0", fg='red')
        self.l_generation.pack()
        Label(self.settings, text="save / load file:").pack()
        self.t_save_load = Entry(self.settings)
        self.t_save_load.pack()
        Button(self.settings, text="save", command=self.save_population).pack()
        Button(self.settings, text="load", command=self.load_population).pack()
        Button(self.settings, text="save left labyrinth", command=lambda: self.save_labyrinth("l")).pack()
        Button(self.settings, text="save right labyrinth", command=lambda: self.save_labyrinth("r")).pack()
        
        self.open = True
        self.plots.protocol("WM_DELETE_WINDOW", self.on_close)
        self.settings.protocol("WM_DELETE_WINDOW", self.on_close)
        self.run()
        
        self.right = None
        self.left = None
    
    def query_global(self):
        self.query(self.input)
        pass
    
    def on_close(self):
        self.is_training = False
        self.open = False
    
    def save_population(self):
        self.l_status.config(text="saving", fg='orange')
        self.settings.update()
        self.generator.save_population(self.t_save_load.get(), self.generation, self.fitness_total.copy())
    
    def load_population(self):
        self.l_status.config(text="loading", fg='orange')
        self.settings.update()
        out = self.generator.load(self.t_save_load.get())
        self.generation = out[0]
        self.fitness_total = out[1].copy()
        self.plot_fitness()
        self.query()
    
    def toggle_training(self):
        self.is_training = not self.is_training
        self.l_status.config(text="ending training", fg='orange')
        self.settings.update()
    
    def train_once(self, long_time: bool = False):
        if long_time:
            self.l_status.config(text="training", fg='green')
        else:
            self.l_status.config(text="ending training", fg='orange')
        # if self.global_inputs:
        #     __input = self.input
        #     pass
        # else:
        __input = np.random.rand(1, 1, self.input_depth)
        for j in range(self.epochs):
            self.settings.update()
            if self.same_inputs or self.global_inputs:
                fitness, index = self.generator.train((1, 1), lambda x: labyrinth_solver.Interpretation.evaluate(x), lambda x: train.translate_gan(x), 0.001, __input)
                self.fitness_total.append(fitness)
            else:
                fitness, index = self.generator.train((1, 1), lambda x: labyrinth_solver.Interpretation.evaluate(x), lambda x: train.translate_gan(x), 0.001)
                self.fitness_total.append(fitness)
            self.best_index = index
            self.generation += 1
            self.l_generation.config(text="Generation = " + str(self.generation), fg='blue')
            self.settings.update()
        self.query()
        self.plot_fitness()
    
    def query(self, inputs_list: np.ndarray = None):
        if self.global_inputs:
            __input = self.input
        self.l_status.config(text="querying", fg='orange')
        self.settings.update()
        if inputs_list is None:
            labyrinth1 = train.get_gan_results(self.generator)
            labyrinth2 = train.get_gan_results(self.generator)
        else:
            labyrinth1 = train.get_gan_results(self.generator, inputs_list)
            labyrinth2 = train.get_gan_results(self.generator)
        self.plot_result((labyrinth_solver.Interpretation.color_ways(labyrinth1), labyrinth_solver.Interpretation.color_ways(labyrinth2)),
                         labyrinth_solver.Interpretation.evaluate(labyrinth1))
    
    def query_batch(self, inputs_list: np.ndarray = None):
        self.l_status.config(text="querying", fg='orange')
        self.settings.update()
        if inputs_list is None:
            labyrinth1 = train.get_gan_results_batch(self.generator)
            labyrinth2 = train.get_gan_results_batch(self.generator)
        else:
            labyrinth1 = train.get_gan_results_batch(self.generator, inputs_list)
            labyrinth2 = train.get_gan_results_batch(self.generator)
        self.plot_result((labyrinth_solver.Interpretation.color_ways(labyrinth1), labyrinth_solver.Interpretation.color_ways(labyrinth2)),
                         labyrinth_solver.Interpretation.evaluate(labyrinth1))
    
    def query_fast(self):
        self.l_status.config(text="querying", fg='orange')
        self.settings.update()
        labyrinth1 = train.get_gan_results_fast(self.generator)
        labyrinth2 = train.get_gan_results_fast(self.generator)
        self.plot_result((labyrinth_solver.Interpretation.color_ways(labyrinth1), labyrinth_solver.Interpretation.color_ways(labyrinth2)),
                         labyrinth_solver.Interpretation.evaluate(labyrinth1))
    
    def run(self):
        labyrinth1 = train.get_gan_results(self.generator)
        labyrinth2 = train.get_gan_results(self.generator)
        self.plot_fitness()
        self.plot_result((labyrinth_solver.Interpretation.color_ways(labyrinth1), labyrinth_solver.Interpretation.color_ways(labyrinth2)),
                         labyrinth_solver.Interpretation.evaluate(labyrinth2))
        self.l_fitness.config(text="Index = " + str(self.best_index) + "; fitness = " + str(labyrinth_solver.Interpretation.evaluate(labyrinth1)))
        while self.open:
            self.plots.update()
            self.settings.update()
            try:
                self.epochs = int(self.t_epochs.get())
            except ValueError:
                self.is_training = False
            self.l_status.config(text="not training", fg='red')
            self.l_generation.config(text="Generation = " + str(self.generation), fg='blue')
            if self.is_training:
                self.train_once(True)
    
    def plot_fitness(self):
        self.ax_f.plot(self.fitness_total, color='blue')
        self.fitness_c.draw()
        self.plots.update()
    
    def plot_result(self, result: [np.ndarray, tuple], fitness: float):
        self.left = result[0]
        self.right = result[1]
        self.ax_r1.imshow(train.translate_for_user(result[0]), origin='upper')
        self.ax_r2.imshow(train.translate_for_user(result[1]), origin='upper')
        self.l_fitness.config(text="Index = " + str(self.best_index) + "; fitness = " + str(fitness))
        self.result_c1.draw()
        self.result_c2.draw()
        self.plots.update()
    
    def save_labyrinth(self, position: str):
        if position == "r":
            image_saver.imsave("images/"+str(datetime.datetime.now()).replace(":", "-")+"-solved.png", train.translate_for_user(self.right))
            image_saver.imsave("images/"+str(datetime.datetime.now()).replace(":", "-")+"-blank.png", train.translate_for_user_bw(self.right), cmap='Greys_r')
        else:
            image_saver.imsave("images/"+str(datetime.datetime.now()).replace(":", "-")+"-solved.png", train.translate_for_user(self.left))
            image_saver.imsave("images/"+str(datetime.datetime.now()).replace(":", "-")+"-blank.png", train.translate_for_user_bw(self.left), cmap='Greys_r')


class train:
    @staticmethod
    def get_gan_results(__generator: gagan.Population, inputs_list: np.ndarray = None):
        if inputs_list is None:
            return train.translate_gan(__generator.query((1, 1), lambda x: labyrinth_solver.Interpretation.evaluate(train.translate_gan(x))))
        return train.translate_gan(__generator.query((1, 1), lambda x: labyrinth_solver.Interpretation.evaluate(train.translate_gan(x)), inputs_list))

    @staticmethod
    def get_gan_results_fast(__generator: gagan.Population):
        return train.translate_gan(__generator.query_fast((1, 1)), True)

    @staticmethod
    def get_gan_results_batch(__generator: gagan.Population, inputs_list: np.ndarray = None):
        if inputs_list is None:
            return train.translate_gan(__generator.query_batch((1, 1), lambda x: labyrinth_solver.Interpretation.evaluate(train.translate_gan(x))))
        return train.translate_gan(__generator.query_batch((1, 1), lambda x: labyrinth_solver.Interpretation.evaluate(train.translate_gan(x)), inputs_list))

    @staticmethod
    def translate_for_user_bw(__labyrinth: np.ndarray):
        return np.asarray(train.translate_for_user(__labyrinth)).any(2)
    
    @staticmethod
    def translate_for_user(__labyrinth: np.ndarray):
        result = np.ones((__labyrinth.shape[0]*3, __labyrinth.shape[1]*3, 3))
        for x in range(__labyrinth.shape[0]*3):
            for y in range(__labyrinth.shape[1]*3):
                if not any(__labyrinth[x//3, y//3]) and (x % 3 == 1 and y % 3 == 1):
                    result[x, y, :] = [0, 0, 0]
                else:
                    result[x, y, :] = __labyrinth[x//3, y//3] if any(__labyrinth[x//3, y//3]) else [1, 1, 1]
        result = result[1:-1, 1:-1, :]
        tmp = result.copy()
        for x in range(tmp.shape[0]):
            for y in range(tmp.shape[1]):
                if all(tmp[x, y, :]):
                    if (x + 2 < tmp.shape[0] and -1 < x - 1 and not (any(tmp[x + 2, y]) or any(tmp[x - 1, y]))) or \
                       (y + 2 < tmp.shape[1] and -1 < y - 1 and not (any(tmp[x, y + 2]) or any(tmp[x, y - 1]))) or \
                       (-1 < x - 2 and x + 1 < tmp.shape[0] and not (any(tmp[x - 2, y]) or any(tmp[x + 1, y]))) or \
                       (-1 < y - 2 and y + 1 < tmp.shape[1] and not (any(tmp[x, y - 2]) or any(tmp[x, y + 1]))):
                        result[x, y, :] = [0, 0, 0]
        for x in range(tmp.shape[0]):
            for y in range(tmp.shape[1]):
                if all(tmp[x, y, :]):
                    if (-1 < x - 1 and -1 < y - 1 and not (any(tmp[x - 1, y]) or any(tmp[x, y - 1]))) or \
                       (-1 < x - 1 and y + 1 < tmp.shape[1] and not (any(tmp[x - 1, y]) or any(tmp[x, y + 1]))) or \
                       (x + 1 < tmp.shape[0] and -1 < y - 1 and not (any(tmp[x + 1, y]) or any(tmp[x, y - 1]))) or \
                       (x + 1 < tmp.shape[0] and y + 1 < tmp.shape[1] and not (any(tmp[x + 1, y]) or any(tmp[x, y + 1]))):
                        result[x, y, :] = [0, 0, 0]
        return result
    
    @staticmethod
    def translate_gan(__labyrinth: np.ndarray, grey=False):
        # if grey:
        #     plt.imsave("images/"+str(datetime.datetime.now()).replace(":", "-")+"-grey0.png", __labyrinth[:, :, 0], cmap="Greys")
        #     plt.imsave("images/"+str(datetime.datetime.now()).replace(":", "-")+"-grey1.png", __labyrinth[:, :, 1], cmap="Greys")
        __labyrinth = np.asarray([[int(__labyrinth[x, y, 0] > __labyrinth[x, y, 1]) for y in range(__labyrinth.shape[1])] for x in range(__labyrinth.shape[0])], np.ndarray)
        __labyrinth[-1, 0] = 1
        __labyrinth[0, -1] = 1
        return __labyrinth
