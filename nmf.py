#!/usr/bin/env python3

from pathlib import Path
import musdb
import matplotlib.pyplot as plt
from sklearn.decomposition import non_negative_factorization
from sklearn.linear_model import Lasso
import librosa
from soundfile import write
import helpers


if __name__ == "__main__":
    root = Path("..", "musdb18hq")

    mus_train = musdb.DB(root=root, subsets="train", is_wav=True)
    mus_test = musdb.DB(root=root, subsets="test", is_wav=True)

    mixture, vocals, accompaniment = helpers.generate_two_stem_data(mus_train, chunk_duration=5.0)

    mixture_mono = 0.5 * (mixture[:, 0] + mixture[:, 1])
    vocals_mono = 0.5 * (vocals[:, 0] + vocals[:, 1])
    accompaniment_mono = 0.5 * (accompaniment[:, 0] + accompaniment[:, 1])

    mixture_stft = librosa.stft(mixture_mono, n_fft=2048, hop_length=256, win_length=2048, center=False)
    vocals_stft = librosa.stft(vocals_mono, n_fft=2048, hop_length=256, win_length=2048, center=False)
    accompaniment_stft = librosa.stft(accompaniment_mono, n_fft=2048, hop_length=256, win_length=2048, center=False)

    mixture_mags, mixture_phases = librosa.magphase(mixture_stft)
    vocals_mags, vocals_phases = librosa.magphase(vocals_stft)
    accompaniment_mags, accompaniment_phases = librosa.magphase(accompaniment_stft)

    print("Starting Vocals NMF")
    vocals_components, vocals_weights = non_negative_factorization(vocals_mags, n_components=4096)
    print("Finished Vocals NMF")

    print("Starting Accomponiment NMF")
    accompaniment_components, accompaniment_weights = non_negative_factorization(accompaniment_mags, n_components=4096)
    print("Finished Accomponiment NMF")

    mixture_components = np.hstack((vocals_components, accompaniments_components))

    print("Starting Separation with Lasso")
    model = Lasso(alpha=0, fit_intercept=False, positive=True)
    model.fit(mixture_components, mixture_stft)
    mixture_weights = model.coef_
    print("Starting Separation with Lasso")

    learned_vocals_weights = mixture_weights[:4096, :]
    learned_accompaniment_weights = mixture_weights[4096:, :]

    learned_vocals_mags = vocals_components @ learned_vocals_weights
    learned_accompaniment_mags = accompaniment_components @ learned_accompaniment_weights

    learned_vocals_stft = learned_vocals_mags * mixture_phases
    learned_accompaniment_stft = learned_accompaniment_mags * mixture_phases

    learned_vocals = librosa.istft(learned_vocals_stft, hop_length=256, win_length=2048, center=False)
    learned_accompaniment = librosa.istft(learned_accompaniment_stft, hop_length=256, win_length=2048, center=False)

    write("mixture.wav", mixture, track.rate)
    write("learned_vocals.wav", learned_vocals, track.rate)
    write("learned_accompaniment.wav", learned_accompaniment, track.rate)
