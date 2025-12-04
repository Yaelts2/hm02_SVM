import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

def split_train_test(X, y, test_size=0.2, random_state=0):
    #This function splits the dataset into training and testing sets while ensuring that the distribution of your target classes remains the same in both sets.
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=test_size, # 20% test , 80% train
                                                        stratify=y,  # Train and Test sets have the exact same proportion
                                                        random_state=random_state,
                                                        )
    return X_train, X_test, y_train, y_test



def train_perceptron(X_train, y_train, max_iter=1000, learning_rate=1.0, random_state=0):
    # This function creates and trains a Perceptron classifier. It fits the model to your provided training data. Finally, it returns the trained model ready for predictions.
    model = Perceptron(max_iter=max_iter,        #This limits the training loops
                    eta0=learning_rate,          #This is the learning rate. It controls how drastically the model updates its weights after a mistake.
                    random_state=random_state,
                    fit_intercept=True,          #This calculates the bias (or offset).
                    tol=1e-3,                    #If the model improves by less than this amount (0.001) between steps, training stops to save time.
                    )
    model.fit(X_train, y_train)
    return model




def train_linear_svm(X_train, y_train, C=1.0, random_state=0):
    # This function creates and trains a Linear Support Vector Machine (SVM). 
    # It tries to find the best straight line (or hyperplane) to separate your data classes.
    model = SVC(kernel="linear",C=C,random_state=random_state)
    model.fit(X_train, y_train)
    return model




def train_kernel_svm(X_train, y_train, C=1.0, gamma="scale", random_state=0):
    #This function trains a Non-Linear Support Vector Machine. separate data that cannot be separated by a straight line.
    model = SVC(kernel="rbf", C=C, gamma=gamma, random_state=random_state)
    model.fit(X_train, y_train)
    return model



def train_poly_svm(X_train, y_train, C=1.0, degree=3, gamma="scale", coef0=0.0, random_state=0):
    #this function trains a Polynomial SVM
    model = SVC(kernel="poly", C=C, degree=degree, gamma=gamma, coef0=coef0, random_state=random_state)
    model.fit(X_train, y_train)
    return model
