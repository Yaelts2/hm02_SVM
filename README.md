# SVM Homework Project – Machine Learning & Neural Networks

This repository contains the full implementation for homework on Support Vector Machines (SVM).  
The project includes data generation, model training, visualization, and performance evaluation for both linear and non-linear datasets.

---

## 📌 Project Structure

hm02_SVM/
│
├── main.py              # Main script: runs Questions 1–5 with training + visualizations
├── datasets.py          # Functions for generating linear and non-linear datasets
├── models.py            # Implementations of Perceptron, Linear SVM, RBF SVM, Poly SVM
├── evaluation.py        # Plotting utilities + confusion matrix + F1 score evaluation
│
├── requirements.txt     # Minimal dependencies needed to run the project
└── README.md            # Project description

---

## 📌 Implemented Models

- **Perceptron classifier**
- **Linear SVM**
- **SVM with RBF kernel**
- **SVM with Polynomial kernel**

Each classifier is trained and evaluated on:
1. A **linearly separable** 2D dataset  
2. A **non-linear “moons” dataset**

---

## 📌 Visualizations Included

The project generates:

- Scatter plots of the datasets  
- Decision boundaries for all models  
- Accuracy comparison bar charts  
- Confusion matrices  
- F1-score comparison (RBF vs Polynomial)

These plots help illustrate why certain kernels perform better on specific data types.

---

## 📌 Key Results Summary

- On the **linear dataset**, all SVM kernels perform almost perfectly.  
- On the **non-linear dataset**, the **RBF kernel significantly outperforms** the Polynomial kernel due to its ability to create flexible, local decision boundaries.  
- Polynomial kernel makes more errors, especially false positives, due to its global shape constraints.

---

## 📌 Requirements

Minimal packages required to run the project:

numpy
matplotlib
scikit-learn

Install with:

```bash
pip install -r requirements.txt

