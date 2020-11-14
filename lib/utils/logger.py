# logger & progress bar

from __future__ import absolute_import
import os
import os.path as osp
import sys
import time
import torch.nn as nn
import torch.nn.init as init

# __all__ = ['Logger', "progress_bar"]

'''
usage
# logger = Logger('pid.txt', title='mnist')
# logger.set_names(['LearningRate', 'TrainLoss', 'ValidLoss', 'TrainAcc', 'ValidAcc'])
'''

class Logger(object):
    '''Save training process to log file.'''
    def __init__(self, fpath, title=None, resume=False):
        self.file = None
        self.resume = resume
        self.title = '' if title is None else title
        if fpath is not None:
            if resume:
                self.file = open(fpath, 'r')
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')
            else:    # build a file
                self.file = open(fpath, 'w')

    def set_names(self, names):    # names for every line
        if self.resume:
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('  ')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()

    def set_keys(self, keys, n_class):    # names for every line
        if self.resume:
            pass
        names = ['TestId', 'Binary']
        for ii_class in range(1, n_class):
            names.append('Class%d'%ii_class)
        self.set_names(names)
    

    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        print(numbers)
        for index, num in enumerate(numbers):
            if(type(num) is int):
                self.file.write("%03d"%num)
            elif(type(num) is str):
                self.file.write("%s"%num)
            else:
                self.file.write("%.4f "%num)
            self.file.write('  ')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def write(self, content, add_extra_empty_line=False):
        self.file.write(content)
        print(content)
        self.file.write('\n')
        if add_extra_empty_line:
            self.file.write('\n')

    def close(self):
        if self.file is not None:
            self.file.close()



