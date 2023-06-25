
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import numpy as np
from scipy import io as scio
from torch.utils.data import Dataset
import os
import random
import numpy as np
from scipy import io as scio
from torch.utils.data import Dataset

class dataloader(Dataset):
    def __init__(self, path, number=256, transform=None, is_train=True):
        """
        Args:
            label_dir: path of true labels of samples
            visual_dir: path of visual features
            audio_dir: path of audio features
            tract_dir: path of trajectory features
            number: numbers of selected samples
            transform: transform function
        """
        self.path = path
        self.number = number
        self.transform = transform#数据预处理
        self.is_train = is_train
        if self.is_train:
            self.visual = scio.loadmat(os.path.join(self.path, 'scene_train.mat'))['train']
            self.audio = scio.loadmat(os.path.join(self.path, 'mfcc_train.mat'))['train']
            self.tract = scio.loadmat(os.path.join(self.path, 'tra_train.mat'))['train']
            self.label = scio.loadmat(os.path.join(self.path, 'Y_train.mat'))['train']
        else:
            self.visual = scio.loadmat(os.path.join(self.path, 'scene_test.mat'))['test']
            self.audio = scio.loadmat(os.path.join(self.path, 'mfcc_test.mat'))['test']
            self.tract = scio.loadmat(os.path.join(self.path, 'tra_test.mat'))['test']
            self.label = scio.loadmat(os.path.join(self.path, 'Y_test.mat'))['test']

        if self.number is not None:
            slices, cnt = [], 0
            self._visual = np.zeros(shape=(number, self.visual.shape[1]))
            self._audio = np.zeros(shape=(number, self.audio.shape[1]))
            self._tract = np.zeros(shape=(number, self.tract.shape[1]))
            self._label = np.zeros(shape=(number, self.label.shape[1]))

            while cnt < number:
                s = random.randint(0, self.label.shape[0])#随机生成一个数在此范围
                if s not in slices:
                    slices.append(s)
                    self._visual[cnt], self._audio[cnt] = self.visual[s], self.audio[s]
                    self._tract[cnt], self._label[cnt] = self.tract[s], self.label[s]

                cnt += 1
        else:
            self._visual = self.visual
            self._audio = self.audio
            self._tract = self.tract
            self._label = self.label

    def __getitem__(self, idx):
        if self.transform is not None:
            self.v = self.transform(self._visual[idx])
            self.a = self.transform(self._audio[idx])
            self.t = self.transform(self._tract[idx])
        else:
            self.v = self._visual[idx]
            self.a = self._audio[idx]
            self.t = self._tract[idx]
        self.l = self._label[idx]
        return np.array(self.v, dtype=np.float32), np.array(self.a, dtype=np.float32), np.array(self.t, dtype=np.float32), np.array(self.l, dtype=np.float32)

    def __len__(self):
        return len(self._label)


# class dataloader(Dataset):
#     def __init__(self, path, number=1, data='meipai', is_train = True):
#         super(dataloader, self).__init__()
#         self.path = path
#         self.number = number
#         self.is_train = is_train
#         # These are torch tensors
#         if self.is_train:
#             self.visual = scio.loadmat(os.path.join(self.path, 'scene_train.mat'))['train']
#             self.audio = scio.loadmat(os.path.join(self.path, 'mfcc_train.mat'))['train']
#             self.tract = scio.loadmat(os.path.join(self.path, 'tra_train.mat'))['train']
#             self.label = scio.loadmat(os.path.join(self.path, 'Y_train.mat'))['train']
#         else:
#             self.visual = scio.loadmat(os.path.join(self.path, 'scene_test.mat'))['test']
#             self.audio = scio.loadmat(os.path.join(self.path, 'mfcc_test.mat'))['test']
#             self.tract = scio.loadmat(os.path.join(self.path, 'tra_test.mat'))['test']
#             self.label = scio.loadmat(os.path.join(self.path, 'Y_test.mat'))['test']
#
#         # Note: this is STILL an numpy array
#         self.meta = None
#
#         self.data = data
#
#         self.n_modalities = 2  # vision/ text/ audio
#
#     def get_n_modalities(self):
#         return self.n_modalities
#
#     def get_seq_len(self):
#         return 1, 1, 1
#
#     def get_dim(self):
#         dim_t = self.tract.shape[1]
#         dim_a = self.audio.shape[1]
#         dim_v = self.visual.shape[1]
#         return dim_t, dim_a, dim_v
#
#     def get_lbl_info(self):
#         # return number_of_labels, label_dim
#         return self.label.shape[0], self.label.shape[1]
#
#     def __len__(self):
#         return len(self.label)
#
#     def __getitem__(self, index):
#         X = (index, self.tract[index], self.audio[index], self.visual[index])
#         Y = self.label[index]
#         META = (0, 0, 0) if self.meta is None else (self.meta[index][0], self.meta[index][1], self.meta[index][2])
#         return X, Y, META



if __name__ == "__main__":
    path = '/media/Harddisk/lsy/meipai_dataset/shuffle_data'
    loader = dataloader(path, is_train=True)

    print(loader)
