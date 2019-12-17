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


class Dataset:
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


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


class DatasetLoader:
    def __init__(self, dataset, batch_size, shuffle=True, preprocess=None,
                 gpu=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(dataset)
        self.max_iter = math.ceil(self.data_size / batch_size)
        self.preprocess = preprocess
        self.gpu = gpu

        self.reset()

    def reset(self):
        self.iteration = 0
        if self.shuffle:
            random.shuffle(self.dataset)

    def __iter__(self):
        return self

    def _get_batch(self):
        i = self.iteration % self.max_iter
        start_idx = i * self.batch_size
        end_idx = (i + 1) * self.batch_size
        batch = self.dataset[start_idx:end_idx]
        return batch

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        batch = self._get_batch()

        xp = cuda.cupy if self.gpu else np
        if self.preprocess is None:
            x = xp.array([example[0] for example in batch])
        else:
            x = xp.array([self.preprocess(example[0]) for example in batch])
        t = xp.array([example[1] for example in batch])

        self.iteration += 1

        return x, t

    def next(self):
        return self.__next__()

    def to_cpu(self):
        self.gpu = False

    def to_gpu(self):
        self.gpu = True


class SeqDataLoader(DatasetLoader):
    def __init__(self, dataset, batch_size, preprocess=None, gpu=False):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=False,
                         preprocess=preprocess, gpu=gpu)

    def _get_batch(self):
        jump = self.data_size // self.batch_size
        offsets = [(i * jump + self.iteration) % self.data_size for i in
                   range(self.batch_size)]
        batch = itemgetter(*offsets)(self.dataset)
        return batch


# =============================================================================
# Preprocess function
# =============================================================================
def preprocess_vgg(image, size=(224, 224), dtype=np.float32):
    """VGGで使用する画像に対して前処理を施しndarrayへと変換する
    VGGのpre-trainedモデルでは、下記の前処理を行う
    - 224x224サイズへのリサイズ
    - BGR順にデータを
    - すべての画素から固定値を差し引く
    - 軸の順番を入れ替える

    Parameters
    ----------
    image : PIL.Image or numpy.ndarray
        入力画像がndarrayの場合は、その形状は(height, width)、
        (hegith, width, channels) もしくは (channels, hegith, width)のいずれか
        （そのチャンネルの並びはRGB）
    size : None or (int, int)
        リサイズする画像サイズ。Noneの場合はリサイズしない
    dtype : numpy.dtype
        変換後のデータ型
    Returns
    -------
    image : numpy.ndarray
        前処理を行ったndarray
    """

    if not pil_available:
        raise ImportError('PIL cannot be loaded. Install Pillow!')

    if isinstance(image, np.ndarray):
        if image.ndim == 3:
            if image.shape[0] == 1:
                image = image[0, :, :]
            elif image.shape[0] == 3:
                image = image.transpose((1, 2, 0))
        image = Image.fromarray(image.astype(np.uint8))
    image = image.convert('RGB')

    if size:
        image = image.resize(size)
    image = np.asarray(image, dtype=dtype)
    image = image[:, :, ::-1]
    image -= np.array([103.939, 116.779, 123.68], dtype=dtype)
    image = image.transpose((2, 0, 1))
    return image