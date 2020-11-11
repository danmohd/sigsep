from pathlib import Path
import musdb
from dl.helpers import generate_four_stem_data_batch, model_train, model_separate_and_evaluate
from os import mkdir
from soundfile import write
import joblib


if __name__ == "__main__":
    root = Path("..", "musdb18hq")
    results_path = Path(".", "results_dl")
    evaldir = results_path / "eval"

    mus_train = musdb.DB(root=root, is_wav=True, subsets="train")

    data_gen = generate_four_stem_data_batch(mus_train, batch_size=10, chunk_duration_train=30.0, chunk_duration_test=15.0)

    components, data = model_train(data_gen, n_components=100, batch_size=10)

    data = next(data_gen)

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
