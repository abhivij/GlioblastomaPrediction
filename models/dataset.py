# code below obtained by slight modification from https://github.com/VafaeeLab/UGVC1056_DeepLearning/blob/master/TEP/dataset.py

import torch.utils.data as data
import numpy as np
import torch


class Dataset(data.Dataset):

    def __init__(self, path):

        with open(path, 'r') as f:
            string = f.readline()

        self.tags = string.strip().split(sep=',')
        self.tags.pop(0)

        data = np.genfromtxt(path, delimiter=',')
        data = data[1:]
        data = np.delete(data, 0, 1)

        self.tags = np.array(self.tags, dtype=np.string_)

        s = np.arange(len(self.tags))

        self.data = data.swapaxes(0, 1)
        self.data = self.data[:,None,:]

        np.random.shuffle(s)

        self.data = self.data[s]
        self.tags = self.tags[s]

        self.labels = {}
        i = 0
        for tag in self.tags:
            if tag not in self.labels:
                self.labels[tag] = i
                i += 1

        self.num_classes = len(self.labels)
        self.num_features = len(self.data[0][0])

        print("Data tensor shape : ", self.data.shape)
        print("Num Features : %2d" % self.num_features)
        print("Num Classes : %2d" % self.num_classes)


    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]).float(), \
               torch.from_numpy(np.array(self.labels[self.tags[index]])).long()


    def __len__(self):
        return len(self.data)