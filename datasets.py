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
    plt.xlabel("x1 [A.U.]", fontsize=18)
    plt.ylabel("x2 [A.U.]", fontsize=18)
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.show()


def visualize_3d_data(X, y, title="3D Spherical Dataset"):
    if X.shape[1] != 3:
        raise ValueError("X must have exactly 3 dimensions for 3D visualization.")
    X0 = X[y == 0]
    X1 = X[y == 1]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X0[:, 0], X0[:, 1], X0[:, 2], 
            c='blue', alpha=0.6, s=20, label='Class 0')
    ax.scatter(X1[:, 0], X1[:, 1], X1[:, 2], 
            c='red', alpha=0.6, s=20, label='Class 1')
    ax.set_title(title, fontsize=20)
    ax.set_xlabel("X1 [A.U.]", fontsize=18)
    ax.set_ylabel("X2 [A.U.]", fontsize=18)
    ax.set_zlabel("X3 [A.U.]", fontsize=18)
    ax.legend()
    plt.tight_layout()
    plt.show()
