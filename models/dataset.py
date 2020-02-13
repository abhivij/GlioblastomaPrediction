# code below obtained by slight modification from https://github.com/VafaeeLab/UGVC1056_DeepLearning/blob/master/TEP/dataset.py

import torch.utils.data as data
import numpy as np
import torch


class Dataset(data.Dataset):

    def __init__(self, path, print_details = False):

        self.tag_to_label = {b'Cancer':1, b'NonCancer':0}

        with open(path, 'r') as f:
            string = f.readline()

        self.tags = string.replace('"', '').strip().split(sep=',')
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

        self.num_features = len(self.data[0][0])

        if print_details:
            print("Data tensor shape : ", self.data.shape)
            print("Num Features : %2d" % self.num_features)


    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]).float(), \
               torch.from_numpy(np.array(self.tag_to_label[self.tags[index]])).long()


    def __len__(self):
        return len(self.data)