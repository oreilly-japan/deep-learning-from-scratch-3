from operator import itemgetter
import math
pil_available = True
try:
    from PIL import Image
except:
    pil_available = False
import random
import numpy as np
from dezero import cuda


# =============================================================================
# Dataset
# =============================================================================
class Dataset:
    def __init__(self, train=True, transforms=None, target_transforms=None):
        self.train = train
        self.transforms = transforms
        self.target_transforms = target_transforms
        if self.transforms is None:
            self.transforms = lambda x: x
        if self.target_transforms is None:
            self.target_transforms = lambda x: x

        self.data = None
        self.label = None

        self.prepare()

    def __getitem__(self, index):
        assert np.isscalar(index)
        if self.label is None:
            return self.transforms(self.data[index]), None
        else:
            return self.transforms(self.data[index]),\
                   self.target_transforms(self.label[index])

    def __len__(self):
        return len(self.data)

    def prepare(self):
        pass


class TupleDataset:
    """Dataset of tuples from multiple equal-length datasets.
    """
    def __init__(self, *datasets):
        self._datasets = datasets
        self._length = len(datasets[0])

    def __getitem__(self, index):
        batches = [dataset[index] for dataset in self._datasets]
        if isinstance(index, slice):
            L = len(batches[0])
            return [tuple([batch[i] for batch in batches]) for i in range(L)]
        else:
            return tuple(batches)

    def __len__(self):
        return self._length


# =============================================================================
# DataLoader
# =============================================================================
class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, gpu=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(dataset)
        self.max_iter = math.ceil(self.data_size / batch_size)
        self.gpu = gpu

        self.reset()

    def reset(self):
        self.iteration = 0

        if self.shuffle:
            self.index = np.random.permutation(len(self.dataset))
        else:
            self.index = np.arange(len(self.dataset))

    def __iter__(self):
        return self

    def _get_batch(self):
        i, batch_size = self.iteration, self.batch_size
        batch_index = self.index[i * batch_size:(i + 1) * batch_size]
        batch = [self.dataset[i] for i in batch_index]
        return batch

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        batch = self._get_batch()
        xp = cuda.cupy if self.gpu else np
        x = xp.array([example[0] for example in batch])
        t = xp.array([example[1] for example in batch])

        self.iteration += 1
        return x, t

    def next(self):
        return self.__next__()

    def to_cpu(self):
        self.gpu = False

    def to_gpu(self):
        self.gpu = True


class SeqDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, gpu=False):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=False,
                         gpu=gpu)

    def _get_batch(self):
        jump = self.data_size // self.batch_size
        batch_index = [(i * jump + self.iteration) % self.data_size for i in
                   range(self.batch_size)]
        batch = [self.dataset[i] for i in batch_index]
        return batch