import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from datasets import visualize_3d_data
from models import split_train_test,train_kernel_svm
from evaluation import (plot_accuracy_bars,
                        compute_confusion_and_f1,
                        )

## First method

def generate_spherical_dataset(n_samples=4000, dim=10, separation=2, std=1.5, random_state=0):
    rng = np.random.RandomState(random_state)
    mu0 = np.zeros(dim)
    mu1 = np.zeros(dim)
    mu1[0] = separation  
    X0 = rng.normal(loc=mu0, scale=std, size=(n_samples//2, dim))
    X1 = rng.normal(loc=mu1, scale=std, size=(n_samples//2, dim))

    X = np.vstack([X0, X1])
    y = np.array([0]*(n_samples//2) + [1]*(n_samples//2))
    return X, y


dims = [3,5,10,15]
accuracies = []
f1_scores = []

for d in dims:
    X, y = generate_spherical_dataset(dim=d)
    if d==3:
        visualize_3d_data(X,y)
    #split into train/test
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2)
    #train SVM with RBF
    model = train_kernel_svm(X_train, y_train, C=1.0, gamma="scale")
    #evaluate: accuracy + confusion + F1
    acc = model.score(X_test, y_test)
    cm, f1 = compute_confusion_and_f1(model, X_test, y_test)
    accuracies.append(acc)
    f1_scores.append(f1)
    print(f"Dimension {d:3d}: accuracy = {acc:.3f}, F1 = {f1:.3f}")
#accuracy bar plot across dimensions
labels = [str(d) for d in dims]
plot_accuracy_bars(accuracies,
                labels,
                title="SVM-RBF accuracy vs data dimension"
                )
plt.figure(figsize=(6,4))
plt.plot(dims, accuracies, marker='o')
plt.xlabel("Dimension")
plt.ylabel("Accuracy")
plt.title("SVM (RBF) Accuracy vs Dimension")
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()


##  Second method


def generate_Circular_highdim(n_samples=4096,dim=2,noise=0.2, extra_std=1.0,random_state=0):
    X2, y = make_moons(n_samples=n_samples,
                    noise=noise,
                    random_state=random_state)
    rng = np.random.RandomState(random_state)
    X_extra = rng.normal(loc=0.0, scale=extra_std, size=(n_samples, dim - 2))
    X_high = np.hstack([X2, X_extra])
    return X_high, y


dims = [3,5,10,15]
accuracies = []

for d in dims:
    X, y = generate_Circular_highdim(dim=d)
    if d ==3 : 
        visualize_3d_data(X,y)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    model = train_kernel_svm(X_train, y_train)
    acc = model.score(X_test, y_test)
    accuracies.append(acc)
    print(f"dim={d}, accuracy={acc:.3f}")
plot_accuracy_bars(accuracies,
                labels,
                title="SVM-RBF accuracy vs data dimension"
                )
plt.figure(figsize=(6,4))
plt.plot(dims, accuracies, marker='o')
plt.xlabel("Dimension")
plt.ylabel("Accuracy")
plt.title("SVM (RBF) Accuracy vs Dimension")
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()