import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score


def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    #This function draws a map showing where the model thinks the boundary is between the two classes .
    #first creating a coordinate grid 
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    #predict for every point in the grid to see the boundary of the model 
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors='k', s=30)
    plt.title(title, fontsize=16)
    plt.xlabel("x1 [A.U.]")
    plt.ylabel("x2 [A.U.]")
    plt.tight_layout()
    plt.show()


def plot_accuracy_bars(accuracies, labels, title="Model Accuracy Comparison"):
    #bar charts showing all models accuracy
    plt.figure(figsize=(6, 4))
    x = np.arange(len(labels))  
    bars = plt.bar(x, accuracies, color=["red", "blue", "green"])
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{acc:.3f}",
                ha='center',
                va='bottom',
                fontsize=12
                )

    plt.xticks(x, labels, fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title(title, fontsize=14)
    plt.ylim(0, 1.05)

    plt.tight_layout()
    plt.show()


def compute_confusion_and_f1(model, X_test, y_test):
    #this function compute the confusion and f1 
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return cm, f1


def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    #this function plots the boundary that the model predicted 
    #first creating a grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),np.linspace(y_min, y_max, 300))
    #finding the boundary by predicting with the model on the grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=30)
    plt.title(title, fontsize=16)
    plt.xlabel("x1 [A.U.]")
    plt.ylabel("x2 [A.U.]")
    plt.tight_layout()
    plt.show()


def plot_accuracy_bars(accuracies, labels, title="Model Comparison", ylabel="Accuracy"):
    plt.figure(figsize=(6, 4))
    x = np.arange(len(labels))  
    bars = plt.bar(x,accuracies,color=["red", "blue", "green"][:len(labels)])
    for bar, val in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3f}",
                ha="center",
                va="bottom",
                ontsize=12,
                )
    plt.xticks(x, labels, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()

    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, fontsize=10)
    plt.yticks(ticks, class_names, fontsize=10)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),ha="center",va="center",fontsize=12)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.show()
