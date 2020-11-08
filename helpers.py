#!/usr/bin/env python3

import musdb
import random
import librosa
import numpy as np
from sklearn.decomposition import non_negative_factorization

from typing import Tuple


def generate_two_stem_data(mus: musdb.DB, chunk_duration=5.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Generator for data to extract 2 stems, vocals and accompaniment, from a mixture.

    Parameters
    ----------
    mus : musdb.DB
        Generator of musdb dataset.

    chunk_duration : float, optional, default = 5.0
        Duration of chunks to yield in seconds.

    Returns
    -------
    mixture : np.ndarray, (nsamples, 2)
        Mixture data

    vocals : np.ndarray, (nsamples, 2)
        Vocals data

    accompaniment : np.ndarray, (nsamples, 2)
        Accompaniment data

    rate : int
        Sample rate of audio
    """
    track = random.choice(mus.tracks)
    track.chunk_duration = chunk_duration
    track.chunk_start = random.uniform(
        0, track.duration - track.chunk_duration)

    mixture = track.audio
    vocals = track.targets["vocals"].audio
    accompaniment = track.targets["accompaniment"].audio
    rate = track.rate

    yield mixture, vocals, accompaniment, rate


def generate_four_stem_data(mus: musdb.DB, chunk_duration=5.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Generator for data to extract 4 stems, vocals, drums, bass, and others, from a mixture.

    Parameters
    ----------
    mus : musdb.DB
        Generator of musdb dataset.

    chunk_duration : float, optional, default = 5.0
        Duration of chunks to yield in seconds.

    Returns
    -------
    mixture : np.ndarray, (nsamples, 2)
        Mixture data

    vocals : np.ndarray, (nsamples, 2)
        Vocals data

    drums : np.ndarray, (nsamples, 2)
        Accompaniment data

    bass : np.ndarray, (nsamples, 2)
        Accompaniment data

    others : np.ndarray, (nsamples, 2)
        Accompaniment data

    rate : int
        Sample rate of audio
    """
    track = random.choice(mus.tracks)
    track.chunk_duration = chunk_duration
    track.chunk_start = random.uniform(
        0, track.duration - track.chunk_duration)

    mixture = track.audio
    vocals = track.targets["vocals"].audio
    drums = track.targets["drums"].audio
    bass = track.targets["bass"].audio
    others = track.targets["other"].audio
    rate = track.rate

    yield mixture, vocals, drums, bass, others, rate


def make_mono(audio: np.ndarray) -> np.ndarray:
    """Get single channel audio data from stereo audio.

    Parameters
    ----------
    audio : np.ndarray, shape (n_samples, 2)
        Stereo audio data.

    Returns
    -------
    audio_mono : np.ndarray, shape (n_samples,)
        Mono audio data formed by averaging channels.
    """
    audio_mono = 0.5 * (audio[:, 0] + audio[:, 1])
    return audio_mono


def get_magphase(audio: np.ndarray, win_length: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    """Returns magnitudes and phases of the audio's spectrogram such that ``stft = mags * phases''.

    Parameters
    ----------
    audio : np.ndarray, shape (n_samples,)
        Mono audio data.

    win_length : int, optional, default = 1024
        Length of Hann window to use.

    Returns
    -------
    audio_mags : np.ndarray, shape (8192, n_frames)
        Magnitude spectrogram of stft.

    audio_phases : np.ndarray, shape (8192, n_frames)
        Phase factors of stft.
    """
    audio_stft = librosa.stft(audio, n_fft=win_length,
                              hop_length=win_length // 2,
                              win_length=win_length,
                              center=False)
    audio_mags, audio_phases = librosa.magphase(audio_stft)

    return audio_mags, audio_phases


def get_cqt_magphase(audio: np.ndarray, rate: int = 44100) -> Tuple[np.ndarray, np.ndarray]:
    """Returns magnitudes and phases of the audio's Constant Q-Transform
    such that ``cqt = mags * phases''.

    Parameters
    ----------
    audio : np.ndarray, shape (n_samples,)
        Mono audio data.

    rate : int, optional, default = 44100
        Sample rate of audio.

    Returns
    -------
    audio_mags : np.ndarray, shape (8192, n_frames)
        Magnitude constant Q-transform of stft.

    audio_phases : np.ndarray, shape (8192, n_frames)
        Phase factors of cqt.
    """
    audio_cqt = librosa.cqt(audio,
                            sr=rate,
                            n_bins=252,
                            bins_per_octave=36)
    audio_mags, audio_phases = librosa.magphase(audio_cqt)

    return audio_mags, audio_phases


def reconstruct_audio(components: np.ndarray, weights: np.ndarray, phases: np.ndarray, win_length: int = 1024) -> np.ndarray:
    """Reconstructs audio from learned components and weights and the supplied phases
    using and inverse STFT.

    Parameters
    ----------
    components : np.ndarray, shape (8192, n_atoms)
        Learned components/atoms/bases from data.

    weights : np.ndarray, shape (n_atoms, n_frames)
        Learned weights from data.

    phases : np.ndarray, shape (8192, n_frames)
        Phase factors to use for reconstruction.

    win_length : int, optional, default = 1024
        Length of Hann window used.

    Returns
    -------
    reconstruction : np.ndarray, shape (n_samples,)
        Reconstructed audio.
    """
    mags = components @ weights
    stft = mags * phases
    reconstruction = librosa.istft(stft,
                                   hop_length=win_length // 2,
                                   win_length=win_length,
                                   center=False)
    return reconstruction


def reconstruct_cqt_audio(components: np.ndarray, weights: np.ndarray, phases: np.ndarray, rate: int = 44100) -> np.ndarray:
    """Reconstructs audio from learned components and weights and the supplied phases
    using an inverse Constant Q-Transform.

    Parameters
    ----------
    components : np.ndarray, shape (8192, n_atoms)
        Learned components/atoms/bases from data.

    weights : np.ndarray, shape (n_atoms, n_frames)
        Learned weights from data.

    phases : np.ndarray, shape (8192, n_frames)
        Phase factors to use for reconstruction.

    rate : int, optional, default = 44100
        Sample rate of audio.

    Returns
    -------
    reconstruction : np.ndarray, shape (n_samples,)
        Reconstructed audio.
    """
    mags = components @ weights
    cqt = mags * phases
    reconstruction = librosa.icqt(cqt,
                                  sr=rate,
                                  bins_per_octave=36)
    return reconstruction


def learn_representation(audio: np.ndarray, win_length: int = 1024, n_components: int = 100, max_iter: int = 400):
    mags, phases = get_magphase(audio, win_length=win_length)
    components, weights, n_iters = non_negative_factorization(mags,
                                                              n_components=n_components,
                                                              beta_loss="kullback-leibler",
                                                              solver="mu",
                                                              max_iter=max_iter)
    return components, weights, n_iters


def learn_cqt_representation(audio: np.ndarray, n_components: int = 100, max_iter: int = 400):
    mags, phases = get_cqt_magphase(audio)
    components, weights, n_iters = non_negative_factorization(mags,
                                                              n_components=n_components,
                                                              beta_loss="kullback-leibler",
                                                              solver="mu",
                                                              max_iter=max_iter)

    return components, weights, n_iters


class NMFResults:
    vocals: Tuple[np.ndarray, np.ndarray, int]
    accompaniment: Tuple[np.ndarray, np.ndarray, int]
    drums: Tuple[np.ndarray, np.ndarray, int]
    bass: Tuple[np.ndarray, np.ndarray, int]
    others: Tuple[np.ndarray, np.ndarray, int]

    def __init__(self) -> None:
        self.vocals = None
        self.accompaniment = None
        self.drums = None
        self.bass = None
        self.others = None

    def set_vocals(self, result):
        print("Set Vocals")
        self.vocals = result

    def set_accompaniment(self, result):
        print("Set Accompaniment")
        self.accompaniment = result

    def set_drums(self, result):
        print("Set Drums")
        self.drums = result

    def set_bass(self, result):
        print("Set Bass")
        self.bass = result

    def set_others(self, result):
        print("Set Others")
        self.others = result
