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
from model import MainModel


class Config:
    def __init__(self, num_classes):
        #debug
        self.seed = 13
        self.seed_torch()

        self.debug_mode = False
        # info about data:
        self.dataset_dir = Path('./data/origin_data/')
        self.preprocessed_dir = Path('./data/preprocessed_data/')
        self.csv_dir = Path('./data/csv_files')
        self.csv_file = {
            'train_curated': self.csv_dir / 'train_curated.csv',
            'train_noisy': self.csv_dir / 'trn_noisy_best50s.csv',
            'sample_submission': self.csv_dir / 'sample_submission.csv'
        }
        self.dataset = {
            'train_curated': self.dataset_dir / 'train_curated',
            'train_noisy': self.dataset_dir / 'train_noisy',
            'test': self.dataset_dir / 'test'
        }

        self.mels = {
            'train_curated': self.preprocessed_dir / 'mels_train_curated.pkl',
            'train_noisy': self.preprocessed_dir / 'mels_trn_noisy_best50s.pkl',
            'test': self.preprocessed_dir / 'mels_test.pkl',  # NOTE: this data doesn't work at 2nd stage
        }

        self.num_classes = num_classes

        self.model = MainModel('Simple', num_classes=self.num_classes).model

        # info about CPU:
        self.n_jobs = cpu_count() // 2 + 4  # save 4 threads for work
        os.environ['MKL_NUM_THREADS'] = str(self.n_jobs)
        os.environ['OMP_NUM_THREADS'] = str(self.n_jobs)

        # preprocessing parameters:

        self.preprocessing_type = 'log_melspectrogram'

        self.sampling_rate = 44100
        self.duration = 6  # in seconds
        self.n_mels = 128  # mel coefficients
        self.hop_length = 347 // 128 * self.n_mels * self.duration  # to make time steps 128
        self.fmin = 20  # minimum frequency
        self.fmax = self.sampling_rate // 2  # maximum frequency
        self.n_fft = self.n_mels * 100  # fft coeffs
        self.padmode = 'constant'  # padding for made
        self.samples = self.sampling_rate * self.duration  # elements in one audio file
        self.window_type = 'hann'

        # neural net info
        self.num_epochs = 500
        self.batch_size = 64
        self.test_batch_size = 256
        self.lr = 3e-3
        self.eta_min = 1e-5
        self.t_max = 20

    def seed_torch(self):
        """For reproducibility of experiments"""

        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def workers_init_fn(self, worker_id):
        np.random.seed(self.seed + worker_id)
