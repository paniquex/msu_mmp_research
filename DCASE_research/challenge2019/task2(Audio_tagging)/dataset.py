## train and test dataset classes

import torch
from torch.utils.data import Dataset
import random
from PIL import Image

import preprocessing
import config

class TrainDataset(Dataset):
    def __init__(self, mels, labels, transforms):
        super().__init__()
        self.mels = mels
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.mels)

    def __getitem__(self, idx):
        # crop 1sec
        image = Image.fromarray(self.mels[idx], mode='RGB')
        time_dim, base_dim = image.size
        crop = random.randint(0, time_dim - base_dim)
        image = image.crop([crop, 0, crop + base_dim, base_dim])
        image = self.transforms(image).div_(255)

        label = self.labels[idx]
        label = torch.from_numpy(label).float()

        return image, label


class TestDataset(Dataset):
    def __init__(self, fnames, mels, transforms, tta=5):
        super().__init__()
        self.fnames = fnames
        self.mels = mels
        self.conf = config.Config(80)
        self.preprocessor = preprocessing.Audio_preprocessor(self.conf, True)
        if self.mels is None:
            self.mels = []
            for fname in fnames:
                self.mels.append(self.preprocessor.read_as_melspectrogram('./data/origin_data/test/' + fname))
        self.transforms = transforms
        self.tta = tta

    def __len__(self):
        return len(self.fnames) * self.tta

    def __getitem__(self, idx):
        new_idx = idx % len(self.fnames)

        image = Image.fromarray(self.mels[new_idx], mode='RGB')
        time_dim, base_dim = image.size
        crop = random.randint(0, time_dim - base_dim)
        image = image.crop([crop, 0, crop + base_dim, base_dim])
        image = self.transforms(image).div_(255)

        fname = self.fnames[new_idx]

        return image, fname