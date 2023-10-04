import os
import numpy as np

path = '/srv/local2/ksande25/NS_data/TVQA/cache/'

class Cache(object):
    def __init__(self, path=path):
        self.path = path
        if os.path.isfile(path + 'cache.npy'):
            self.data = np.load(path + 'cache.npy', allow_pickle=True).item()
        else:
            self.data = {}
    def add(self, clip, prompt, ans):
        if not clip in self.data.keys():
            self.data[clip] = []
        self.data[clip].append([prompt, ans])
    def save(self):
        n = len(os.listdir(path))
        np.save(path + 'cache_' + str(n) + '.npy', self.data)
        np.save(path + 'cache.npy', self.data)