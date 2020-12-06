#!/usr/bin/env python3

__all__ = [
    "generate_two_stem_data",
    "generate_four_stem_data",
    "generate_four_stem_data_batch",
    "make_mono",
    "get_magphase",
    "reconstruct_audio",
    "learn_representation",
    "model_train",
    "model_separate",
    "model_separate_and_evaluate",
    "model_test",
    "NMFResults"
]

import musdb
import museval
import random
import librosa
import numpy as np
from sklearn.decomposition import non_negative_factorization, DictionaryLearning
from sklearn.linear_model import orthogonal_mp
from multiprocessing import Pool

from typing import Tuple, Dict, Generator, List
from functools import partial


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
    while True:
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
    while True:
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


def generate_four_stem_data_batch(mus: musdb.DB, batch_size: int = 10, chunk_duration_train: float = 15.0, chunk_duration_test: float = 15.0):
    while True:
        track = random.choice(mus.tracks)
        for batch in range(batch_size):
            track.chunk_duration = chunk_duration_train
            track.chunk_start = random.uniform(
                0, track.duration - track.chunk_duration
            )

            mixture = track.audio
            vocals = track.targets["vocals"].audio
            drums = track.targets["drums"].audio
            bass = track.targets["bass"].audio
            others = track.targets["other"].audio
            rate = track.rate

            yield mixture, vocals, drums, bass, others, rate

        track.chunk_duration = chunk_duration_test
        track.chunk_start = random.uniform(
            0, track.duration - track.chunk_duration
        )

        # mixture = track.audio
        # vocals = track.targets["vocals"].audio
        # drums = track.targets["drums"].audio
        # bass = track.targets["bass"].audio
        # others = track.targets["other"].audio
        # rate = track.rate

        # yield mixture, vocals, drums, bass, others, rate
        yield track


def stitch_audio(batch: List[Tuple]) -> Tuple:
    batch_size = len(batch)

    stitched_data = list(batch[0])

    for index in range(batch_size - 1):
        new_data = list(batch[index + 1])
        stitched_data = [np.hstack((s1, s2)) for s1, s2 in zip(stitched_data[:-1], new_data[:-1])]
        stitched_data.append(new_data[-1])

    return tuple(stitched_data)


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
    return np.ascontiguousarray(audio_mono)


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


def learn_representation(audio: np.ndarray,
                         win_length: int = 1024,
                         n_components: int = 100,
                         max_iter: int = 400,
                         init: str = None,
                         W: np.ndarray = None,
                         H: np.ndarray = None):
    mags, phases = get_magphase(audio, win_length=win_length)
    components, weights, n_iters = non_negative_factorization(mags,
                                                              init=init,
                                                              W=W,
                                                              H=H,
                                                              n_components=n_components,
                                                              beta_loss="kullback-leibler",
                                                              solver="mu",
                                                              l1_ratio=1.0,
                                                              alpha=0.1,
                                                              max_iter=max_iter)
    # model = DictionaryLearning(n_components=n_components,
    #                            tol=1e-1,
    #                            fit_algorithm="cd",
    #                            transform_algorithm="lasso_cd",
    #                            positive_code=True,
    #                            positive_dict=True,
    #                            max_iter=max_iter)
    # weights = model.fit_transform(mags.T).T
    # components = model.components_.T
    # n_iters = model.n_iter_
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
        print("Set vocals")
        self.vocals = result

    def set_accompaniment(self, result):
        print("Set accompaniment")
        self.accompaniment = result

    def set_drums(self, result):
        print("Set drums")
        self.drums = result

    def set_bass(self, result):
        print("Set bass")
        self.bass = result

    def set_others(self, result):
        print("Set others")
        self.others = result

    def append_vocals(self, result):
        print("Append vocals")
        new = (
            np.hstack((self.vocals[0], result[0])),
            np.vstack((self.vocals[1], result[1])),
            self.vocals[2] + result[2]
        )
        self.vocals = new

    def append_accompaniment(self, result):
        print("Append accompaniment")
        new = (
            np.hstack((self.accompaniment[0], result[0])),
            np.vstack((self.accompaniment[1], result[1])),
            self.accompaniment[2] + result[2]
        )
        self.accompaniment = new

    def append_drums(self, result):
        print("Append drums")
        new = (
            np.hstack((self.drums[0], result[0])),
            np.vstack((self.drums[1], result[1])),
            self.drums[2] + result[2]
        )
        self.drums = new

    def append_bass(self, result):
        print("Append bass")
        new = (
            np.hstack((self.bass[0], result[0])),
            np.vstack((self.bass[1], result[1])),
            self.bass[2] + result[2]
        )
        self.bass = new

    def append_others(self, result):
        print("Append others")
        new = (
            np.hstack((self.others[0], result[0])),
            np.vstack((self.others[1], result[1])),
            self.others[2] + result[2]
        )
        self.others = new


def model_train(data_gen: Generator, n_components: int = 30, batch_size: int = 10, n_iter: int = 1) -> NMFResults:
    # rng = np.random.default_rng()
    results = NMFResults()

    # batch_size = 20

    data_batches = [next(data_gen) for i in range(batch_size)]
    data = stitch_audio(data_batches)

    # for batch in range(batch_size - 1):
    #     new_data = next(data_gen)
    #     stitched_mixture = np.hstack((data[0], new_data[0]))
    #     stitched_vocals = np.hstack((data[1], new_data[1]))
    #     # stitched_accompaniment = np.hstack((data[2], new_data[2]))
    #     stitched_drums = np.hstack((data[2], new_data[2]))
    #     stiched_bass = np.hstack((data[3], new_data[3]))
    #     stitched_others = np.hstack((data[4], new_data[4]))
    #     # data = (stitched_mixture, stitched_vocals, stitched_accompaniment, data[-1])
    #     data = (stitched_mixture, stitched_vocals, stitched_drums,
    #             stiched_bass, stitched_others, data[-1])

    mixture, vocals, drums, bass, others, rate = data
    # mixture, vocals, accompaniment, rate = data

    vocals_mono = make_mono(vocals)
    # accompaniment_mono = make_mono(accompaniment)
    drums_mono = make_mono(drums)
    bass_mono = make_mono(bass)
    others_mono = make_mono(others)

    win_length = 8192

    init = None

    pool = Pool(4)

    print("Iteration 0")
    print("Starting Vocals")
    proc_vocals = pool.apply_async(learn_representation,
                                   args=(vocals_mono,),
                                   kwds={
                                       "init": init,
                                       "win_length": win_length,
                                       "n_components": n_components,
                                       "max_iter": 600
                                   },
                                   callback=results.set_vocals)

    # proc_accompaniment = pool.apply_async(learn_representation,
    #                                       args=(accompaniment_mono,),
    #                                       kwds={
    #                                           "init": init,
    #                                           "win_length": win_length,
    #                                           "n_components": n_components,
    #                                           "max_iter": 600
    #                                       },
    #                                       callback=results.set_accompaniment)

    print("Starting Drums")
    proc_drums = pool.apply_async(learn_representation,
                                  args=(drums_mono,),
                                  kwds={
                                      "init": init,
                                      "win_length": win_length,
                                      "n_components": n_components,
                                      "max_iter": 600
                                  },
                                  callback=results.set_drums)

    print("Starting Bass")
    proc_bass = pool.apply_async(learn_representation,
                                 args=(bass_mono,),
                                 kwds={
                                     "init": init,
                                     "win_length": win_length,
                                     "n_components": n_components,
                                     "max_iter": 600
                                 },
                                 callback=results.set_bass)

    print("Starting Others")
    proc_others = pool.apply_async(learn_representation,
                                   args=(others_mono,),
                                   kwds={
                                       "init": init,
                                       "win_length": win_length,
                                       "n_components": n_components,
                                       "max_iter": 600
                                   },
                                   callback=results.set_others)

    proc_vocals.wait()
    # proc_accompaniment.wait()
    proc_drums.wait()
    proc_bass.wait()
    proc_others.wait()

    # init = "custom"

    # for it in range(n_iter - 1):
    #     print(f"Iteration {it + 1}")

    #     data = next(data_gen)

    #     for batch in range(batch_size - 1):
    #         new_data = next(data_gen)
    #         stitched_mixture = np.hstack((data[0], new_data[0]))
    #         stitched_vocals = np.hstack((data[1], new_data[1]))
    #         # stitched_accompaniment = np.hstack((data[2], new_data[2]))
    #         stitched_drums = np.hstack((data[2], new_data[2]))
    #         stiched_bass = np.hstack((data[3], new_data[3]))
    #         stitched_others = np.hstack((data[4], new_data[4]))
    #         # data = (stitched_mixture, stitched_vocals, stitched_accompaniment, data[-1])
    #         data = (stitched_mixture, stitched_vocals, stitched_drums,
    #                 stiched_bass, stitched_others, data[-1])

    #     mixture, vocals, drums, bass, others, rate = data
    #     # mixture, vocals, accompaniment, rate = data

    #     vocals_mono = make_mono(vocals)
    #     # accompaniment_mono = make_mono(accompaniment)
    #     drums_mono = make_mono(drums)
    #     bass_mono = make_mono(bass)
    #     others_mono = make_mono(others)

    #     proc_vocals = pool.apply_async(learn_representation,
    #                                    args=(vocals_mono,),
    #                                    kwds={
    #                                        "W": results.vocals[0] + rng.uniform(0, 1e-3, results.vocals[0].shape),
    #                                        "H": results.vocals[1] + rng.uniform(0, 1e-3, results.vocals[1].shape),
    #                                        "init": init,
    #                                        "win_length": win_length,
    #                                        "n_components": results.vocals[0].shape[1],
    #                                        "max_iter": 100
    #                                    },
    #                                    callback=results.set_vocals)

    #     # proc_accompaniment = pool.apply_async(learn_representation,
    #     #                                       args=(accompaniment_mono,),
    #     #                                       kwds={
    #     #                                           "W": results.accompaniment[0] + rng.uniform(0, 1e-3, results.accompaniment[0].shape),
    #     #                                           "H": results.accompaniment[1] + rng.uniform(0, 1e-3, results.accompaniment[1].shape),
    #     #                                           "init": init,
    #     #                                           "win_length": win_length,
    #     #                                           "n_components": results.accompaniment[0].shape[1],
    #     #                                           "max_iter": 100
    #     #                                       },
    #     #                                       callback=results.set_accompaniment)

    #     proc_drums = pool.apply_async(learn_representation,
    #                                   args=(drums_mono,),
    #                                   kwds={
    #                                       "W": results.drums[0] + rng.uniform(0, 1e-3, results.drums[0].shape),
    #                                       "H": results.drums[1] + rng.uniform(0, 1e-3, results.drums[1].shape),
    #                                       "init": init,
    #                                       "win_length": win_length,
    #                                       "n_components": results.drums[0].shape[1],
    #                                       "max_iter": 100
    #                                   },
    #                                   callback=results.append_drums)

    #     proc_bass = pool.apply_async(learn_representation,
    #                                  args=(bass_mono,),
    #                                  kwds={
    #                                      "W": results.bass[0] + rng.uniform(0, 1e-3, results.bass[0].shape),
    #                                      "H": results.bass[1] + rng.uniform(0, 1e-3, results.bass[1].shape),
    #                                      "init": init,
    #                                      "win_length": win_length,
    #                                      "n_components": results.bass[0].shape[1],
    #                                      "max_iter": 100
    #                                  },
    #                                  callback=results.append_bass)

    #     proc_others = pool.apply_async(learn_representation,
    #                                    args=(others_mono,),
    #                                    kwds={
    #                                        "W": results.others[0] + rng.uniform(0, 1e-3, results.others[0].shape),
    #                                        "H": results.others[1] + rng.uniform(0, 1e-3, results.others[1].shape),
    #                                        "init": init,
    #                                        "win_length": win_length,
    #                                        "n_components": results.others[0].shape[1],
    #                                        "max_iter": 100
    #                                    },
    #                                    callback=results.append_others)

    #     proc_vocals.wait()
    #     # proc_accompaniment.wait()
    #     proc_drums.wait()
    #     proc_bass.wait()
    #     proc_others.wait()

    pool.close()
    pool.join()

    return results, data


def model_separate(components: NMFResults, mixture: np.ndarray) -> Dict:
    mixture_L = np.ascontiguousarray(mixture[:, 0])
    mixture_R = np.ascontiguousarray(mixture[:, 1])

    win_length = 8192

    mixture_mags_L, mixture_phases_L = get_magphase(
        mixture_L, win_length=win_length)
    mixture_mags_R, mixture_phases_R = get_magphase(
        mixture_R, win_length=win_length)

    vocals_components = components.vocals[0]
    # accompaniment_components = components.accompaniment[0]
    drums_components = components.drums[0]
    bass_components = components.bass[0]
    others_components = components.others[0]

    n_vocals_components = vocals_components.shape[1]
    # n_accompaniment_components = accompaniment_components.shape[1]
    n_drums_components = drums_components.shape[1]
    n_bass_components = bass_components.shape[1]
    n_others_components = others_components.shape[1]

    mixture_components = np.hstack(
        (
            vocals_components,
            drums_components,
            bass_components,
            others_components
            # accompaniment_components
        )
    )

    mixture_weights_LT, _, _ = non_negative_factorization(
        mixture_mags_L.T,
        H=mixture_components.T,
        n_components=mixture_components.shape[1],
        update_H=False,
        solver="mu",
        max_iter=600,
        l1_ratio=1.0,
        alpha=0.1,
        beta_loss="kullback-leibler"
    )
    mixture_weights_L = mixture_weights_LT.T

    mixture_weights_RT, _, _ = non_negative_factorization(
        mixture_mags_R.T,
        H=mixture_components.T,
        n_components=mixture_components.shape[1],
        update_H=False,
        solver="mu",
        max_iter=600,
        l1_ratio=1.0,
        alpha=0.1,
        beta_loss="kullback-leibler"
    )
    mixture_weights_R = mixture_weights_RT.T

    # mixture_weights_L = orthogonal_mp(
    #     mixture_components, mixture_mags_L, n_nonzero_coefs=(mixture_components.shape[1] // 4))
    # mixture_weights_R = orthogonal_mp(
    #     mixture_components, mixture_mags_R, n_nonzero_coefs=(mixture_components.shape[1] // 4))

    zero = 0
    one = zero + n_vocals_components
    # two = one + n_accompaniment_components
    two = one + n_drums_components
    three = two + n_bass_components
    four = three + n_others_components

    learned_vocals_weights_L = mixture_weights_L[zero:one, :]
    # learned_accompaniment_weights_L = mixture_weights_L[one:two, :]
    learned_drums_weights_L = mixture_weights_L[one:two, :]
    learned_bass_weights_L = mixture_weights_L[two:three, :]
    learned_others_weights_L = mixture_weights_L[three:four, :]

    learned_vocals_weights_R = mixture_weights_R[zero:one, :]
    # learned_accompaniment_weights_R = mixture_weights_R[one:two, :]
    learned_drums_weights_R = mixture_weights_R[one:two, :]
    learned_bass_weights_R = mixture_weights_R[two:three, :]
    learned_others_weights_R = mixture_weights_R[three:four, :]

    reconstruct_L = partial(reconstruct_audio,
                            phases=mixture_phases_L,
                            win_length=win_length)
    reconstruct_R = partial(reconstruct_audio,
                            phases=mixture_phases_R,
                            win_length=win_length)

    learned_vocals_L = reconstruct_L(
        vocals_components, learned_vocals_weights_L)
    # learned_accompaniment_L = reconstruct_L(
    #     accompaniment_components, learned_accompaniment_weights_L)
    learned_drums_L = reconstruct_L(drums_components, learned_drums_weights_L)
    learned_bass_L = reconstruct_L(bass_components, learned_bass_weights_L)
    learned_others_L = reconstruct_L(
        others_components, learned_others_weights_L)

    learned_vocals_R = reconstruct_R(
        vocals_components, learned_vocals_weights_R)
    # learned_accompaniment_R = reconstruct_R(
    #     accompaniment_components, learned_accompaniment_weights_R)
    learned_drums_R = reconstruct_R(drums_components, learned_drums_weights_R)
    learned_bass_R = reconstruct_R(bass_components, learned_bass_weights_R)
    learned_others_R = reconstruct_R(
        others_components, learned_others_weights_R)

    learned_vocals = np.vstack((learned_vocals_L, learned_vocals_R)).T
    # learned_accompaniment = np.vstack(
    #     (learned_accompaniment_L, learned_accompaniment_R)).T
    learned_drums = np.vstack((learned_drums_L, learned_drums_R)).T
    learned_bass = np.vstack((learned_bass_L, learned_bass_R)).T
    learned_others = np.vstack((learned_others_L, learned_others_R)).T

    # return learned_vocals, learned_accompaniment
    return learned_vocals, learned_drums, learned_bass, learned_others


def model_separate_and_evaluate(components: NMFResults, track: musdb.MultiTrack, evaldir):
    mixture = track.audio

    print("Separating")
    separated_sources = model_separate(components, mixture)
    estimates = {
        "vocals": separated_sources[0],
        "drums": separated_sources[1],
        "bass": separated_sources[2],
        "other": separated_sources[3]
        # "accompaniment": separated_sources[1]
    }

    print("Evaluating")
    scores = museval.eval_mus_track(track, estimates, output_dir=evaldir)

    print(scores)
    print("Done")

    return separated_sources, scores


def model_test(components: NMFResults, mus: musdb.DB):
    pass
