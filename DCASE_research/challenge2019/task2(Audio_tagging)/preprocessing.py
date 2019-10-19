# Module for preprocessing
# TODO:
#  1) MFCC
#  2) Filterbanks
#  3) Log spectrograms
#  4) ...
import librosa
import librosa.display
import IPython
import IPython.display
import matplotlib.pyplot as plt  # for debug in jupyter-notebook
import numpy as np
from tqdm import tqdm_notebook




class Audio_preprocessor:
    """
    Short description: class for preprocessing audio
    Types of preprocessing:
        1) MFCC
        2) filterbanks
        3) melspectrogram
        4) log mel spectrogram
    """

    def __init__(self, conf, trim_long_data):
        """
        :param conf:
            Config class from config.py with all params for preprocessing, main of them:
                * conf.preprocessing_type:
                    1) 'MFCC' - mel-frequency spectral coefficients
                    2) 'filterbanks'
                    3) 'melspectrogram' - audio -> melspectrogram
                    4) 'log_melspectrogram' - log scaling of melspectrogram
                *  conf.sampling_rate
                   conf.duration  # in seconds
                   conf.hop_length = 347 * conf.duration  # to make time steps 128
                   conf.fmin # minimum frequency
                   conf.fmax  # maximum frequency
                   conf.n_mels = 128  # mel coefficients
                   conf.n_fft = conf.n_mels * 20  # fft coeffs
                   conf.padmode  # padding for made
                   conf.samples = conf.sampling_rate * conf.duration  # elements in one audio file
                   conf.window_type # window type for melspectrogram
        """

        self.conf = conf
        self.conf.seed_torch()
        self.trim_long_data = trim_long_data

    def read_audio(self, pathname, trim_long_data):
        y, sr = librosa.load(pathname, sr=self.conf.sampling_rate)
        # trim silence
        if 0 < len(y):  # workaround: 0 length causes error
            y, _ = librosa.effects.trim(y)  # trim, top_db=default(60)
        # make it unified length to conf.samples
        if len(y) > self.conf.samples:  # long enough
            if trim_long_data:
                y = y[0:0 + self.conf.samples]
        else:  # pad blank
            padding = self.conf.samples - len(y)  # add padding at both ends
            offset = padding // 2
            y = np.pad(y, (offset, self.conf.samples - len(y) - offset), self.conf.padmode)
        return y

    def audio_to_melspectrogram(self, audio):
        spectrogram = librosa.feature.melspectrogram(audio,
                                                     sr=self.conf.sampling_rate,
                                                     n_mels=self.conf.n_mels,
                                                     hop_length=self.conf.hop_length,
                                                     n_fft=self.conf.n_fft + 10,
                                                     fmin=self.conf.fmin,
                                                     fmax=self.conf.fmax,
                                                     window=self.conf.window_type)
        spectrogram = librosa.power_to_db(spectrogram)
        spectrogram = spectrogram.astype(np.float32)
        return spectrogram

    # For debug in jupyter-notebook
    def show_melspectrogram(self, mels, title='Log-frequency power spectrogram'):
        librosa.display.specshow(mels, x_axis='time', y_axis='mel',
                                 sr=self.conf.sampling_rate, hop_length=self.conf.hop_length,
                                 fmin=self.conf.fmin, fmax=self.conf.fmax)
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.show()

    def read_as_melspectrogram(self, pathname):
        x = self.read_audio(pathname, self.trim_long_data)
        mels = self.audio_to_melspectrogram(x)
        if self.conf.debug_mode:
            IPython.display.display(IPython.display.Audio(x, rate=self.conf.sampling_rate))
            self.show_melspectrogram(mels)
        return mels

    def mono_to_color(self, X, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6):
        # Stack X as [X,X,X]
        X = np.stack([X, X, X], axis=-1)

        # Standardize
        mean = mean or X.mean()
        X = X - mean
        std = std or X.std()
        Xstd = X / (std + eps)
        _min, _max = Xstd.min(), Xstd.max()
        norm_max = norm_max or _max
        norm_min = norm_min or _min
        if (_max - _min) > eps:
            # Normalize to [0, 255]
            V = Xstd
            V[V < norm_min] = norm_min
            V[V > norm_max] = norm_max
            V = 255 * (V - norm_min) / (norm_max - norm_min)
            V = V.astype(np.uint8)
        else:
            # Just zero
            V = np.zeros_like(Xstd, dtype=np.uint8)
        return V

    def convert_wav_to_image(self, df, source, conf):
        X = []
        for i, row in tqdm_notebook(df.iterrows()):
            x = self.read_as_melspectrogram(conf,
                                            source / str(row.fname),
                                            trim_long_data=self.trim_long_data)
            x_color = self.mono_to_color(x)
            X.append(x_color)
        return X

    def preprocess(self, pathname):
        if self.conf.preprocessing_type == 'melspectrogram':
            return self.read_as_melspectrogram(pathname=pathname)
        elif self.conf.preprocessing_type == 'log_melspectrogram':
            return np.log(self.read_as_melspectrogram(pathname=pathname))
        else:
            raise TypeError('Unknown type of transformation: ' + self.conf.preprocessing_type)