#!/usr/bin/env python3

from pathlib import Path
import musdb
import matplotlib.pyplot as plt


if __name__ == "__main__":
    root = Path("..", "musdb18hq")
    print(root.is_dir())

    mus_train = musdb.DB(root=root, subsets="train", is_wav=True)
    mus_test = musdb.DB(root=root, subsets="test", is_wav=True)

    audio = mus_train[0].audio

    plt.plot(audio)
    plt.show()
