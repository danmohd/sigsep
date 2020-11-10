#!/usr/bin/env python3

from os import mkdir
from nmf.helpers import generate_four_stem_data_batch, generate_two_stem_data, model_separate_and_evaluate
from nmf import NMFResults, generate_four_stem_data, get_magphase, learn_representation, make_mono, reconstruct_audio, model_train, model_test, model_separate
from pathlib import Path
import numpy as np
import musdb
# import matplotlib.pyplot as plt
# from sklearn.linear_model import orthogonal_mp
from sklearn.decomposition import non_negative_factorization
from soundfile import write
from multiprocessing import Pool
import joblib


# TODO: MAKE EVERYTHING A FUNCTION
def temp():
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

    vocals_components = results.vocals[0]
    drums_components = results.drums[0]
    bass_components = results.bass[0]
    others_components = results.others[0]

    mixture_components = np.hstack(
        (vocals_components, drums_components, bass_components, others_components))

    mixture_mono = make_mono(mixture)
    mixture_mags, mixture_phases = get_magphase(
        mixture_mono, win_length=win_length)

    # print("Starting Separation with OMP")
    # mixture_weights = orthogonal_mp(mixture_components,
    #                                 mixture_mags,
    #                                 n_nonzero_coefs=(4 * n_components // 3))
    # print("Finished Separation with OMP")

    print("Starting Separation")
    mixture_weights_T, mixture_components_T, n_iter = non_negative_factorization(
        mixture_mags.T,
        H=mixture_components.T,
        n_components=4 * n_components,
        update_H=False,
        solver="mu",
        max_iter=600,
        beta_loss="kullback-leibler"
    )
    mixture_weights = mixture_weights_T.T
    print("Finished Separation")

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


if __name__ == "__main__":
    root = Path("..", "musdb18hq")
    results_path = Path(".", "results")
    evaldir = results_path / "eval"

    mus_train = musdb.DB(root=root, is_wav=True, subsets="train")
    mus_test = musdb.DB(root=root, is_wav=True, subsets="test")

    data_gen = generate_four_stem_data_batch(mus_train, batch_size=10, chunk_duration_train=30.0, chunk_duration_test=15.0)

    components, data = model_train(data_gen, n_components=30, batch_size=10, n_iter=1)

    data = next(data_gen)
    # data = next(generate_two_stem_data(mus_test, chunk_duration=15.0))
    # data = next(generate_four_stem_data(mus_test, chunk_duration=15.0))

    separations, scores = model_separate_and_evaluate(components, data, evaldir)

    mixture = data.audio
    vocals = data.targets["vocals"].audio
    drums = data.targets["drums"].audio
    bass = data.targets["bass"].audio
    others = data.targets["other"].audio
    rate = data.rate

    learned_vocals, learned_drums, learned_bass, learned_others = separations
    # mixture, vocals, accompaniment, rate = data
    # learned_vocals, learned_accompaniment = separations

    results_path = results_path / data.name

    if not results_path.exists():
        mkdir(results_path)

    joblib.dump(components, results_path / "model.mdl")

    write(results_path / "mixture.wav", mixture, rate)

    write(results_path / "true_vocals.wav", vocals, rate)
    # write(results_path / "true_accompaniment.wav", accompaniment, rate)
    write(results_path / "true_drums.wav", drums, rate)
    write(results_path / "true_bass.wav", bass, rate)
    write(results_path / "true_others.wav", others, rate)

    write(results_path / "learned_vocals.wav", learned_vocals, rate)
    # write(results_path / "learned_accompaniment.wav", learned_accompaniment, rate)
    write(results_path / "learned_drums.wav", learned_drums, rate)
    write(results_path / "learned_bass.wav", learned_bass, rate)
    write(results_path / "learned_others.wav", learned_others, rate)

    with open(results_path / "eval.txt", "w") as file:
        file.write(repr(scores))
