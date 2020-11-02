#!/usr/bin/env python3

from pathlib import Path
import numpy as np
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

    mixture, vocals, accompaniment, rate = next(helpers.generate_two_stem_data(mus_train, chunk_duration=5.0))

    mixture_mono = 0.5 * (mixture[:, 0] + mixture[:, 1])
    vocals_mono = 0.5 * (vocals[:, 0] + vocals[:, 1])
    accompaniment_mono = 0.5 * (accompaniment[:, 0] + accompaniment[:, 1])

    mixture_stft = librosa.stft(mixture_mono, n_fft=2048, hop_length=256, win_length=2048, center=False)
    vocals_stft = librosa.stft(vocals_mono, n_fft=2048, hop_length=256, win_length=2048, center=False)
    accompaniment_stft = librosa.stft(accompaniment_mono, n_fft=2048, hop_length=256, win_length=2048, center=False)

    mixture_mags, mixture_phases = librosa.magphase(mixture_stft)
    vocals_mags, vocals_phases = librosa.magphase(vocals_stft)
    accompaniment_mags, accompaniment_phases = librosa.magphase(accompaniment_stft)

    # X = W @ H. W (n_channels, n_atoms), H (n_atoms, n_time_frames)
    # Hyperparameter
    n_components = 2048

    print("Starting Vocals NMF")
    vocals_components, vocals_weights, n_iter = non_negative_factorization(vocals_mags, n_components=n_components, beta_loss="kullback-leibler", solver="mu")
    print(f"Finished Vocals NMF in {n_iter} iterations.")

    print("Starting Accomponiment NMF")
    accompaniment_components, accompaniment_weights, n_iter = non_negative_factorization(accompaniment_mags, n_components=n_components, beta_loss="kullback-leibler", solver="mu")
    print(f"Finished Accomponiment NMF in {n_iter} iterations")

    # BV bases for vocals
    # BA bases for acoompaniment
    # B = [BV BA]
    mixture_components = np.hstack((vocals_components, accompaniment_components))

    print("Starting Separation with Lasso")
    model = Lasso(alpha=0.5, fit_intercept=False, positive=True, selection="random")
    model.fit(mixture_components, mixture_mags)
    mixture_weights = model.coef_.T
    print("Finished Separation with Lasso")

    learned_vocals_weights = mixture_weights[:n_components, :]
    learned_accompaniment_weights = mixture_weights[n_components:, :]

    learned_vocals_mags = vocals_components @ learned_vocals_weights
    learned_accompaniment_mags = accompaniment_components @ learned_accompaniment_weights

    learned_vocals_stft = learned_vocals_mags * mixture_phases
    learned_accompaniment_stft = learned_accompaniment_mags * mixture_phases

    learned_vocals = librosa.istft(learned_vocals_stft, hop_length=256, win_length=2048, center=False)
    learned_accompaniment = librosa.istft(learned_accompaniment_stft, hop_length=256, win_length=2048, center=False)

    write("mixture.wav", mixture, rate)
    write("learned_vocals.wav", learned_vocals, rate)
    write("learned_accompaniment.wav", learned_accompaniment, rate)
