#!/usr/bin/env python3

import musdb
import random


def generate_two_stem_data(mus: musdb.DB, chunk_duration=5.0):
    """Generator for data to extract 2 stems, vocals and accompaniment, from a mixture.

    Parameters
    ----------
    mus : musdb.DB
        Generator of musdb dataset.

    chunk_duration : float, default = 5.0
        Duration of chunks to yield in seconds

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
    track.chunk_start = random.uniform(0, track.duration - track.chunk_duration)

    mixture = track.audio
    vocals = track.targets["vocals"].audio
    accompaniment = track.targets["accompaniment"].audio
    rate = track.rate

    yield mixture, vocals, accompaniment, rate
