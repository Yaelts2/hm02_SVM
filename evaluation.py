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
    plt.title(title, fontsize=20)
    plt.xlabel("x1 [A.U.]", fontsize=18)
    plt.ylabel("x2 [A.U.]", fontsize=18)
    plt.tight_layout()
    plt.show()


def plot_accuracy_bars(accuracies, labels, title="Model Accuracy Comparison",ylabel='Accuracy'):
    #bar charts showing all models accuracy
    plt.figure(figsize=(6, 4))
    x = np.arange(len(labels))  
    bars = plt.bar(x, accuracies, color=["red", "blue", "green", "purple"])
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{acc:.3f}",
                ha='center',
                va='bottom',
                fontsize=12
                )

    plt.xticks(x, labels, fontsize=12)
    plt.ylabel(ylabel, fontsize=18)
    plt.title(title, fontsize=20)
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
    plt.title(title, fontsize=20)
    plt.xlabel("x1 [A.U.]", fontsize=18)
    plt.ylabel("x2 [A.U.]", fontsize=18)
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
            plt.text(j, i, str(cm[i, j]),ha="center",va="center",fontsize=18)
    plt.xlabel("Predicted", fontsize=18)
    plt.ylabel("True", fontsize=18)
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.show()
