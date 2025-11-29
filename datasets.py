import numpy as np
from sklearn.datasets import make_blobs, make_moons
import matplotlib.pyplot as plt


def generate_linear_dataset(n_samples_per_class=2048, random_state=0):
    np.random.seed(random_state)
    X, y = make_blobs(n_samples=2 * n_samples_per_class,
                    centers=[(-2, 0), (2, 0)],   
                    cluster_std=0.8,
                    random_state=random_state
                    )
    return X, y


def generate_Circular_dataset(n_samples=4096, noise=0.2, random_state=0):
    X, y = make_moons(n_samples=n_samples,
                    noise=noise,
                    random_state=random_state
                    )
    return X, y


def plot_dataset(X, y, title="Dataset"):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y,edgecolors='k', s=30)
    plt.xlabel("x1 [A.U.]", fontsize=14)
    plt.ylabel("x2 [A.U.]", fontsize=14)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()
