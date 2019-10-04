# TODO: fill all info about
    # 1) preprocessing type, parameters, etc..
    # 2) model type, parameters(amount of conv layers, lstm), etc..
    # 3) info about logger(directory, file name), frequency of logging
    # 4) ..

import random
import os
import numpy as np
import torch
from pathlib import Path
from psutil import cpu_count


def seed_torch(seed=13):
    """For reproducibility of experiments"""

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Config:
    def __init__(self, num_classes):
        # info about data:
        self.dataset_dir = Path('./data/origin_data/')
        self.preprocessed_dir = Path('./data/preprocessed_data/')
        self.csv_dir = Path('./data/csv_files')
        self.csv_file = {
            'train_curated': self.csv_dir / 'train_curated.csv',
            'train_noisy': self.csv_dir / 'train_noisy.csv',
            'sample_submission': self.csv_dir / 'sample_submission.csv'
        }
        self.dataset = {
            'train_curated': self.dataset_dir / 'train_curated',
            'train_noisy': self.dataset_dir / 'train_noisy',
            'test': self.dataset_dir / 'test'
        }

        self.num_classes = num_classes

        # info about CPU:
        self.n_jobs = cpu_count() // 2 + 4  # save 4 threads for work
        os.environ['MKL_NUM_THREADS'] = str(self.n_jobs)
        os.environ['OMP_NUM_THREADS'] = str(self.n_jobs)

        # preprocessing parameters:
        self.sampling_rate = 44100
        self.duration = 2  # in seconds
        self.hop_length = 347 * self.duration  # to make time steps 128
        self.fmin = 20  # minimum frequency
        self.fmax = self.sampling_rate // 2  # maximum frequency
        self.n_mels = 128  # mel coefficients
        self.n_fft = self.n_mels * 20  # fft coeffs
        self.padmode = 'constant'  # padding for made
        self.samples = self.sampling_rate * self.duration  # elements in one audio file
        self.window_type = 'hann'

