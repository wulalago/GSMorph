import os

import numpy as np
import matplotlib.pyplot as plt


class Plotter(object):
    """
    Record and plot the loss and metric curve.
    ----------------------------
    Parameters:
        send_path: [string] path to save the plot figure and log files
    ----------------------------
    """
    def __init__(self, send_path):
        self.send_path = send_path
        self.buffer = dict()

    def update(self, logs):
        for key in logs.keys():
            if key not in self.buffer.keys():
                self.buffer[key] = []
            self.buffer[key].append(logs[key])

    def send(self):
        for key in self.buffer.keys():
            plt.figure()
            plt.plot(self.buffer[key])
            # plt.yscale('log')
            plt.title(key)
            plt.xlabel("iteration")
            plt.savefig(os.path.join(self.send_path, key+".png"))
            plt.close()

    def save(self):
        np.save(os.path.join(self.send_path, 'log.npy'), self.buffer)


class Recorder(object):
    """
    record the metric and return the statistic results
    """
    def __init__(self):
        self.data = dict()
        self.keys = []

    def update(self, item):
        for key in item.keys():
            if key not in self.keys:
                self.keys.append(key)
                self.data[key] = []

            self.data[key].append(item[key])

    def reset(self, keys=None):
        if keys is None:
            keys = self.data.keys()
        for key in keys:
            self.data[key] = []

    def call(self, key, return_std=False):
        arr = np.array(self.data[key])
        if return_std:
            return np.mean(arr), np.std(arr)
        else:
            return np.mean(arr)

    def average(self):
        average_dict = {}
        for key in self.keys:
            average_dict[key] = np.mean(np.array(self.data[key]))

        return average_dict

    def stddev(self):
        stddev_dict = {}
        for key in self.keys:
            stddev_dict[key] = np.std(np.array(self.data[key]))

        return stddev_dict

    def info(self, std=False):
        avg_dict = self.average()
        std_dict = self.stddev()
        print_info = []

        for key in self.keys:
            if std:
                print_info += [f'{key}: {avg_dict[key]:.5f}-{std_dict[key]:.5f} ']
            else:
                print_info += [f'{key}: {avg_dict[key]:.5f} ']
        print_info = ''.join(print_info)
        return print_info
