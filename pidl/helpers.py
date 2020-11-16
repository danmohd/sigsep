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
from sklearn.decomposition import DictionaryLearning
from sklearn.linear_model import OrthogonalMatchingPursuit

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


def learn_dictionary(data, n_components):
    model = DictionaryLearning(n_components=n_components, alpha=1.0, max_iter=200)
    model = model.fit(data.T)
    dictionary = model.components_.T
    return dictionary


def stitch_audio(batch: List[Tuple]) -> Tuple:
    batch_size = len(batch)

    stitched_data = list(batch[0])

    for index in range(batch_size - 1):
        new_data = list(batch[index + 1])
        stitched_data = [np.hstack((s1, s2)) for s1, s2 in zip(stitched_data[:-1], new_data[:-1])]
        stitched_data.append(new_data[-1])

    return tuple(stitched_data)


# TODO: Change to PI-DL
def model_train(data_gen: Generator, n_components: int = 30, batch_size: int = 10):
    data_batches = [next(data_gen) for i in range(batch_size)]
    data = stitch_audio(data_batches)

    mixture, vocals, drums, bass, others, rate = data
    # mixture, vocals, accompaniment, rate = data

    vocals_mono = make_mono(vocals)
    # accompaniment_mono = make_mono(accompaniment)
    drums_mono = make_mono(drums)
    bass_mono = make_mono(bass)
    others_mono = make_mono(others)

    win_length = 8192

    vocals_features = get_features(vocals_mono, win_length)
    drums_features = get_features(drums_mono, win_length)
    bass_features = get_features(bass_mono, win_length)
    others_features = get_features(others_mono, win_length)

    print("Starting Vocals")
    vocals_components = learn_dictionary(vocals_features, n_components=n_components)
    print("Finished Vocals")
    print("Starting Drums")
    drums_components = learn_dictionary(drums_features, n_components=n_components)
    print("Finished Drums")
    print("Starting Bass")
    bass_components = learn_dictionary(bass_features, n_components=n_components)
    print("Finished Bass")
    print("Starting Others")
    others_components = learn_dictionary(others_features, n_components=n_components)
    print("Finished Others")

    results = {
        "vocals": vocals_components,
        "drums": drums_components,
        "bass": bass_components,
        "others": others_components
    }

    return results, data


# TODO: Change to PI-DL
def model_separate(components: Dict, mixture: np.ndarray) -> Dict:
    mixture_L = np.ascontiguousarray(mixture[:, 0])
    mixture_R = np.ascontiguousarray(mixture[:, 1])

    win_length = 8192

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

    model = OrthogonalMatchingPursuit(fit_intercept=False).fit(mixture_components, mixture_L_features)
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

    learned_vocals_L = np.ravel(vocals_components @ learned_vocals_weights_L, order="F")
    learned_drums_L = np.ravel(drums_components @ learned_drums_weights_L, order="F")
    learned_bass_L = np.ravel(bass_components @ learned_bass_weights_L, order="F")
    learned_others_L = np.ravel(others_components @ learned_others_weights_L, order="F")

    learned_vocals_R = np.ravel(vocals_components @ learned_vocals_weights_R, order="F")
    learned_drums_R = np.ravel(drums_components @ learned_drums_weights_R, order="F")
    learned_bass_R = np.ravel(bass_components @ learned_bass_weights_R, order="F")
    learned_others_R = np.ravel(others_components @ learned_others_weights_R, order="F")

    learned_vocals = np.vstack((learned_vocals_L, learned_vocals_R)).T
    # learned_accompaniment = np.vstack(
    #     (learned_accompaniment_L, learned_accompaniment_R)).T
    learned_drums = np.vstack((learned_drums_L, learned_drums_R)).T
    learned_bass = np.vstack((learned_bass_L, learned_bass_R)).T
    learned_others = np.vstack((learned_others_L, learned_others_R)).T

    # return learned_vocals, learned_accompaniment
    return learned_vocals, learned_drums, learned_bass, learned_others


# TODO: Change to PI-DL.
def model_separate_and_evaluate(components: Dict, track: musdb.MultiTrack, evaldir):
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
