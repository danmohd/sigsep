#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import musdb
# import matplotlib.pyplot as plt
from sklearn.linear_model import orthogonal_mp
from soundfile import write
from multiprocessing import Pool

from helpers import NMFResults, generate_four_stem_data, get_magphase, learn_representation, make_mono, reconstruct_audio


# TODO: MAKE EVERYTHING A FUNCTION
if __name__ == "__main__":
    root = Path("..", "musdb18hq")
    results_path = Path(".", "results")

    mus_train = musdb.DB(root=root, subsets="train",
                         split="train", is_wav=True)
    mus_valid = musdb.DB(root=root, subsets="train",
                         split="valid", is_wav=True)
    mus_test = musdb.DB(root=root, subsets="test", is_wav=True)

    mixture, vocals, drums, bass, others, rate = next(
        generate_four_stem_data(mus_train, chunk_duration=10.0))

    vocals_mono = make_mono(vocals)
    drums_mono = make_mono(drums)
    bass_mono = make_mono(bass)
    others_mono = make_mono(others)

    # Hyperparameters
    win_length = 8192
    n_components = 30

    pool = Pool()

    results = NMFResults()

    print("Starting Vocals")
    pool.apply_async(learn_representation,
                     args=(vocals_mono,),
                     kwds={
                         "win_length": win_length,
                         "n_components": n_components,
                         "max_iter": 600
                     },
                     callback=results.set_vocals)

    print("Starting Drums")
    pool.apply_async(learn_representation,
                     args=(drums_mono,),
                     kwds={
                         "win_length": win_length,
                         "n_components": n_components,
                         "max_iter": 600
                     },
                     callback=results.set_drums)

    print("Starting Bass")
    pool.apply_async(learn_representation,
                     args=(bass_mono,),
                     kwds={
                         "win_length": win_length,
                         "n_components": n_components,
                         "max_iter": 600
                     },
                     callback=results.set_bass)

    print("Starting Others")
    pool.apply_async(learn_representation,
                     args=(others_mono,),
                     kwds={
                         "win_length": win_length,
                         "n_components": n_components,
                         "max_iter": 600
                     },
                     callback=results.set_others)

    pool.close()
    pool.join()

    vocals_components, vocals_weights, vocals_iter = results.vocals
    drums_components, drums_weights, drums_iter = results.drums
    bass_components, bass_weights, bass_iter = results.bass
    others_components, others_weights, others_iter = results.others

    mixture_components = np.hstack(
        (vocals_components, drums_components, bass_components, others_components))

    mixture_mono = make_mono(mixture)
    mixture_mags, mixture_phases = get_magphase(
        mixture_mono, win_length=win_length)

    print("Starting Separation with OMP")
    mixture_weights = orthogonal_mp(mixture_components,
                                    mixture_mags,
                                    n_nonzero_coefs=(4 * n_components // 3))
    print("Finished Separation with OMP")

    learned_vocals_weights = mixture_weights[:n_components, :]
    learned_drums_weights = mixture_weights[n_components:(2 * n_components), :]
    learned_bass_weights = mixture_weights[(
        2 * n_components):(3 * n_components), :]
    learned_others_weights = mixture_weights[(3 * n_components):, :]

    learned_vocals = reconstruct_audio(
        vocals_components, learned_vocals_weights, mixture_phases, win_length=win_length)
    learned_drums = reconstruct_audio(
        drums_components, learned_drums_weights, mixture_phases, win_length=win_length)
    learned_bass = reconstruct_audio(
        bass_components, learned_bass_weights, mixture_phases, win_length=win_length)
    learned_others = reconstruct_audio(
        others_components, learned_others_weights, mixture_phases, win_length=win_length)

    write(results_path / "mixture.wav", mixture, rate)

    write(results_path / "true_vocals.wav", vocals, rate)
    write(results_path / "true_drums.wav", drums, rate)
    write(results_path / "true_bass.wav", bass, rate)
    write(results_path / "true_others.wav", others, rate)

    write(results_path / "learned_vocals.wav", learned_vocals, rate)
    write(results_path / "learned_drums.wav", learned_drums, rate)
    write(results_path / "learned_bass.wav", learned_bass, rate)
    write(results_path / "learned_others.wav", learned_others, rate)
