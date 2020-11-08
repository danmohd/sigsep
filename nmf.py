#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import musdb
# import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from soundfile import write
import helpers
from multiprocessing import Pool


# TODO: MAKE EVERYTHING A FUNCTION
if __name__ == "__main__":
    root = Path("..", "musdb18hq")

    mus_train = musdb.DB(root=root, subsets="train",
                         split="train", is_wav=True)
    mus_valid = musdb.DB(root=root, subsets="train",
                         split="valid", is_wav=True)
    mus_test = musdb.DB(root=root, subsets="test", is_wav=True)

    mixture, vocals, accompaniment, rate = next(
        helpers.generate_two_stem_data(mus_train, chunk_duration=5.0))

    vocals_mono = helpers.make_mono(vocals)
    accompaniment_mono = helpers.make_mono(accompaniment)

    # Hyperparameter
    n_components = 40

    pool = Pool()

    results = helpers.NMFResults()

    print("Starting Vocals")
    pool.apply_async(helpers.learn_representation,
                     args=(vocals_mono,),
                     kwds={
                         "n_components": n_components,
                         "max_iter": 400
                     },
                     callback=results.set_vocals)

    print("Starting Accompaniment")
    pool.apply_async(helpers.learn_representation,
                     args=(accompaniment_mono,),
                     kwds={
                         "n_components": n_components,
                         "max_iter": 400
                     },
                     callback=results.set_accompaniment)

    pool.close()
    pool.join()

    vocals_components, vocals_weights, vocals_iter = results.vocals
    accompaniment_components, accompaniment_weights, accompaniment_iter = results.accompaniment

    mixture_components = np.hstack(
        (vocals_components, accompaniment_components))

    mixture_mono = helpers.make_mono(mixture)
    mixture_mags, mixture_phases = helpers.get_magphase(mixture_mono)

    # TODO: Replace with KL Divergence based iterations
    print("Starting Separation with Lasso")
    model = Lasso(alpha=0.1, fit_intercept=False,
                  positive=True, selection="random")
    model.fit(mixture_components, mixture_mags)
    mixture_weights = model.coef_.T
    print("Finished Separation with Lasso")

    learned_vocals_weights = mixture_weights[:n_components, :]
    learned_accompaniment_weights = mixture_weights[n_components:, :]

    learned_vocals = helpers.reconstruct_audio(
        vocals_components, learned_vocals_weights, mixture_phases)
    learned_accompaniment = helpers.reconstruct_audio(
        accompaniment_components, learned_accompaniment_weights, mixture_phases)

    write("mixture.wav", mixture, rate)
    write("learned_vocals.wav", learned_vocals, rate)
    write("learned_accompaniment.wav", learned_accompaniment, rate)
