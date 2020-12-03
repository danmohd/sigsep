__all__ = [
    "generate_four_stem_data_batch",
    "model_train",
    "model_separate",
    "model_separate_and_evaluate"
]

import numpy as np
import musdb
import museval
import random
from sklearn.preprocessing import normalize
from sklearn.decomposition import DictionaryLearning
from sklearn.linear_model import OrthogonalMatchingPursuit
from multiprocessing import Pool

from typing import List, Dict, Tuple, Generator


def generate_four_stem_data_batch(mus: musdb.DB, batch_size: int = 10, chunk_duration_train: float = 15.0, chunk_duration_test: float = 15.0):
    while True:
        track = random.choice(mus.tracks)
        for batch in range(batch_size):
            track.chunk_duration = chunk_duration_train
            track.chuck_start = random.uniform(
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

        yield track


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


def get_features(audio: np.ndarray, n_features: int):
    len_audio = len(audio)
    n_samples = len_audio // n_features
    data = audio[:n_samples * n_features].reshape((n_samples, n_features)).T
    return data


def learn_dictionary_ompprecomp(data: np.ndarray, n_components: int, n_iter: int):
    # Y
    Y = data
    # initial lambdas, (N, n_components)
    # L = np.ones((Y.shape[0], n_components))
    rng = np.random.default_rng()
    L = rng.uniform(low=-1, high=1, size=(Y.shape[0] // 2 + 1, n_components))
    L = 0.5 * (L + np.flipud(L))

    # Dimension of data (fft size) or Number of features
    N = Y.shape[0]
    I = np.eye(N)
    F = np.fft.rfft(I, axis=0, norm="ortho")
    Fstar = np.conj(np.transpose(F))

    # model = OrthogonalMatchingPursuit(fit_intercept=False, precompute=True)
    model = OrthogonalMatchingPursuit(
        n_nonzero_coefs=n_components, fit_intercept=False, precompute=True)

    for it in range(n_iter):
        print(f"Iteration: {it}")

        # Compute Gram
        Lstar = np.fft.irfft(np.diag(np.conj(L[:, 0])), axis=0, norm="ortho")
        for i in range(1, n_components):
            Lstar = np.vstack((Lstar, np.fft.irfft(
                np.diag(np.conj(L[:, i])), axis=0, norm="ortho")))

        G = Lstar @ np.conj(np.transpose(Lstar))
        # G = np.real(G)

        # Compute Inner products
        Lambda = L[:, 0]
        DTy = np.fft.irfft(np.conj(
            Lambda[:, np.newaxis]) * np.fft.rfft(Y, axis=0, norm="ortho"), axis=0, norm="ortho")
        for i in range(1, n_components):
            Lambda = L[:, i]
            DTy = np.vstack((DTy, np.fft.irfft(np.conj(
                Lambda[:, np.newaxis]) * np.fft.rfft(Y, axis=0, norm="ortho"), axis=0, norm="ortho")))

        # DTy = np.real(DTy)

        # Sparse Update
        print(f"Iteration: {it}, Sparse Update")
        model.fit(G, DTy)
        X = model.coef_.T

        # Dictionary update
        # Do Blockwise update
        print(f"Iteration: {it}, Blockwise Update")
        for block in range(n_components):
            print(f"Iteration: {it}, Block: {block}")
            Zl = np.fft.rfft(Y, axis=0, norm="ortho")
            for otherblock in range(n_components):
                if otherblock != block:
                    Zl -= np.diag(L[:, otherblock]) @ np.fft.rfft(
                        X[N * otherblock: N * (otherblock + 1), :], axis=0, norm="ortho")

            # Do lambda-wise update for each row in Zl
            FXl = np.conj(np.fft.rfft(
                X[N * block: N * (block + 1), :], axis=0, norm="ortho"))
            for row in range(Zl.shape[0]):
                FXli = FXl[row, :]  # shape (
                Zli = Zl[row, :]
                L[row, block] = Zli @ FXli / \
                    (1e-16 + np.linalg.norm(FXli) ** 2)

    Lambda = L[:, 0]
    D = np.fft.irfft(Lambda[:, np.newaxis] * F, axis=0, norm="ortho")
    for i in range(1, n_components):
        Lambda = L[:, i]
        D = np.hstack(
            (D, np.fft.irfft(Lambda[:, np.newaxis] * F, axis=0, norm="ortho")))

    return D


def learn_dictionary(data: np.ndarray, n_components: int, n_iter: int):
    # Y
    Y = data
    # initial lambdas, (N, n_components)
    # L = np.ones((Y.shape[0], n_components))
    rng = np.random.default_rng()
    L = rng.uniform(low=-1, high=1, size=(Y.shape[0], n_components))
    L = 0.5 * (L + np.flipud(L))

    # Dimension of data (fft size) or Number of features
    N = Y.shape[0]
    I = np.eye(N)
    # F = F @ I / sqrt(N)
    # divide by sqrt(N) so that Fstar @ F = I
    F = np.fft.fft(I, axis=0, norm="ortho")

    model = OrthogonalMatchingPursuit(
        n_nonzero_coefs=n_components, fit_intercept=False)

    for it in range(n_iter):
        print(f"Iteration: {it}")
        # Construct full dictionary
        # D = Fstar @ (np.diag(L[:, 0]) @ F)
        # D = np.fft.ifft(np.diag(L[:, 0]) @ F, axis=0, norm="ortho")
        Lambda = L[:, 0]
        D = np.fft.ifft(Lambda[:, np.newaxis] * F, axis=0, norm="ortho")
        for i in range(1, n_components):
            # D = np.hstack((D, Fstar @ np.diag(L[:, i]) @ F))
            # D = np.hstack((D, np.fft.ifft(np.diag(L[:, i]) @ F, axis=0, norm="ortho")))
            Lambda = L[:, i]
            D = np.hstack(
                (D, np.fft.ifft(Lambda[:, np.newaxis] * F, axis=0, norm="ortho")))
        D = np.real(D)
        D = normalize(D, norm="l2", axis=0)

        # Sparse Update
        print(f"Iteration: {it}, Sparse Update")
        model.fit(D, Y)
        X = model.coef_.T

        # Dictionary update
        # Do Blockwise update
        print(f"Iteration: {it}, Blockwise Update")
        for block in range(n_components):
            print(f"Iteration: {it}, Block: {block}")
            Zl = np.fft.fft(Y, axis=0, norm="ortho")
            for otherblock in range(n_components):
                if otherblock != block:
                    Zl -= np.diag(L[:, otherblock]) @ np.fft.fft(
                        X[N * otherblock: N * (otherblock + 1), :], axis=0, norm="ortho")

            # Do lambda-wise update for each row in Zl
            FXl = np.conj(np.fft.fft(
                X[N * block: N * (block + 1), :], axis=0, norm="ortho"))
            for row in range(Zl.shape[0]):
                FXli = FXl[row, :]  # shape (
                Zli = Zl[row, :]
                L[row, block] = Zli @ FXli / \
                    (1e-16 + np.linalg.norm(FXli) ** 2)

    Lambda = L[:, 0]
    D = np.fft.ifft(Lambda[:, np.newaxis] * F, axis=0, norm="ortho")
    for i in range(1, n_components):
        Lambda = L[:, i]
        D = np.hstack(
            (D, np.fft.ifft(Lambda[:, np.newaxis] * F, axis=0, norm="ortho")))
    D = np.real(D)
    D = normalize(D, norm="l2", axis=0)

    return D


def stitch_audio(batch: List[Tuple]) -> Tuple:
    batch_size = len(batch)

    stitched_data = list(batch[0])

    for index in range(batch_size - 1):
        new_data = list(batch[index + 1])
        stitched_data = [np.hstack((s1, s2)) for s1, s2 in zip(
            stitched_data[:-1], new_data[:-1])]
        stitched_data.append(new_data[-1])

    return tuple(stitched_data)


class PIDLResults:
    def __init__(self):
        self.vocals = None
        self.drums = None
        self.bass = None
        self.others = None

    def set_vocals(self, components):
        print("Set Vocals")
        self.vocals = components

    def set_drums(self, components):
        print("Set Drums")
        self.drums = components

    def set_bass(self, components):
        print("Set Bass")
        self.bass = components

    def set_others(self, components):
        print("Set Others")
        self.others = components


# TODO: Change to PI-DL
def model_train(data_gen: Generator, win_length: int = 64, n_components: int = 30, batch_size: int = 1, n_iter=5):
    data_batches = [next(data_gen) for i in range(batch_size)]
    data = stitch_audio(data_batches)

    mixture, vocals, drums, bass, others, rate = data
    # mixture, vocals, accompaniment, rate = data

    vocals_mono = make_mono(vocals)
    # accompaniment_mono = make_mono(accompaniment)
    drums_mono = make_mono(drums)
    bass_mono = make_mono(bass)
    others_mono = make_mono(others)

    vocals_features = get_features(vocals_mono, win_length)
    drums_features = get_features(drums_mono, win_length)
    bass_features = get_features(bass_mono, win_length)
    others_features = get_features(others_mono, win_length)

    results = PIDLResults()
    pool = Pool(processes=4)

    print("Starting Vocals")
    pool.apply_async(learn_dictionary,
                     args=(vocals_features,),
                     kwds={
                         "n_components": n_components,
                         "n_iter": n_iter
                     },
                     callback=results.set_vocals)

    print("Starting Drums")
    pool.apply_async(learn_dictionary,
                     args=(drums_features,),
                     kwds={
                         "n_components": n_components,
                         "n_iter": n_iter
                     },
                     callback=results.set_drums)

    print("Starting Bass")
    pool.apply_async(learn_dictionary,
                     args=(bass_features,),
                     kwds={
                         "n_components": n_components,
                         "n_iter": n_iter
                     },
                     callback=results.set_bass)

    print("Starting Others")
    pool.apply_async(learn_dictionary,
                     args=(others_features,),
                     kwds={
                         "n_components": n_components,
                         "n_iter": n_iter
                     },
                     callback=results.set_others)

    pool.close()
    pool.join()

    vocals_components = results.vocals
    drums_components = results.drums
    bass_components = results.bass
    others_components = results.others

    # print("Starting Vocals")
    # vocals_components = learn_dictionary(
    #     vocals_features, n_components=n_components, n_iter=n_iter)
    # print("Finished Vocals")
    # print("Starting Drums")
    # drums_components = learn_dictionary(
    #     drums_features, n_components=n_components, n_iter=n_iter)
    # print("Finished Drums")
    # print("Starting Bass")
    # bass_components = learn_dictionary(
    #     bass_features, n_components=n_components, n_iter=n_iter)
    # print("Finished Bass")
    # print("Starting Others")
    # others_components = learn_dictionary(
    #     others_features, n_components=n_components, n_iter=n_iter)
    # print("Finished Others")

    results = {
        "vocals": vocals_components,
        "drums": drums_components,
        "bass": bass_components,
        "others": others_components,
        "win_length": win_length
    }

    return results, data


# TODO: Change to PI-DL
def model_separate(components: Dict, mixture: np.ndarray, n_nonzero_coeffs: int = 40) -> Dict:
    mixture_L = np.ascontiguousarray(mixture[:, 0])
    mixture_R = np.ascontiguousarray(mixture[:, 1])

    win_length = components["win_length"]

    mixture_L_features = get_features(mixture_L, win_length)
    mixture_R_features = get_features(mixture_R, win_length)

    vocals_components = components["vocals"]
    # accompaniment_components = components.accompaniment[0]
    drums_components = components["drums"]
    bass_components = components["bass"]
    others_components = components["others"]

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

    n_mixture_components = n_vocals_components + \
        n_drums_components + n_bass_components + n_others_components

    model = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coeffs, fit_intercept=False).fit(
        mixture_components, mixture_L_features)
    mixture_weights_L = model.coef_.T
    model = model.fit(mixture_components, mixture_R_features)
    mixture_weights_R = model.coef_.T

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

    learned_vocals_L = np.ravel(
        vocals_components @ learned_vocals_weights_L, order="F")
    learned_drums_L = np.ravel(
        drums_components @ learned_drums_weights_L, order="F")
    learned_bass_L = np.ravel(
        bass_components @ learned_bass_weights_L, order="F")
    learned_others_L = np.ravel(
        others_components @ learned_others_weights_L, order="F")

    learned_vocals_R = np.ravel(
        vocals_components @ learned_vocals_weights_R, order="F")
    learned_drums_R = np.ravel(
        drums_components @ learned_drums_weights_R, order="F")
    learned_bass_R = np.ravel(
        bass_components @ learned_bass_weights_R, order="F")
    learned_others_R = np.ravel(
        others_components @ learned_others_weights_R, order="F")

    learned_vocals = np.vstack((learned_vocals_L, learned_vocals_R)).T
    # learned_accompaniment = np.vstack(
    #     (learned_accompaniment_L, learned_accompaniment_R)).T
    learned_drums = np.vstack((learned_drums_L, learned_drums_R)).T
    learned_bass = np.vstack((learned_bass_L, learned_bass_R)).T
    learned_others = np.vstack((learned_others_L, learned_others_R)).T

    # return learned_vocals, learned_accompaniment
    return learned_vocals, learned_drums, learned_bass, learned_others


# TODO: Change to PI-DL.
def model_separate_and_evaluate(components: Dict, track: musdb.MultiTrack, evaldir, n_nonzero_coeffs: int = 40):
    mixture = track.audio

    print("Separating")
    separated_sources = model_separate(components, mixture, n_nonzero_coeffs=n_nonzero_coeffs)
    estimates = {
        "vocals": separated_sources[0],
        "drums": separated_sources[1],
        "bass": separated_sources[2],
        "other": separated_sources[3]
        # "accompaniment": separated_sources[1]
    }

    scores = None

    try:
        print("Evaluating")
        scores = museval.eval_mus_track(track, estimates, output_dir=evaldir)

        print(scores)
        print("Done")
    except ValueError as e:
        pass
    else:
        print("Evaluation Success")

    return separated_sources, scores
