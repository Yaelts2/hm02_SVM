# SVM Homework Project – Machine Learning & Neural Networks

This repository contains the full implementation for homework on Support Vector Machines (SVM).  
The project includes data generation, model training, visualization, and performance evaluation for both linear and non-linear datasets.

---

## Project Structure

```bash
hm02_SVM/
│
├── main.py              # Main script executing Questions 1–5 with training and visualizations
├── datasets.py          # Functions for generating linear and non-linear datasets
├── models.py            # Implementations of Perceptron, Linear SVM, RBF SVM, Polynomial SVM
├── evaluation.py        # Plotting utilities, confusion matrix, and F1-score evaluation
├── q5c_highdim.py       # High-dimensional data experiments for Question 5c
│
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation

```

## Implemented Models

- **Perceptron classifier**
- **Linear SVM**
- **SVM with RBF kernel**
- **SVM with Polynomial kernel**

Each classifier is trained and evaluated on:
1. A **linearly separable** 2D dataset  
2. A **non-linear “moons” dataset**

---

## Visualizations Included

The project generates:

- Scatter plots of the datasets  
- Decision boundaries for all models  
- Accuracy comparison bar charts  
- Confusion matrices  
- F1-score comparison (RBF vs Polynomial)

These visualizations demonstrate why different kernels succeed or fail depending on the data structure.
---

## Key Results Summary

- On the **linear dataset**, all SVM kernels perform almost perfectly.  
- On the **non-linear dataset**, the **RBF kernel significantly outperforms** the Polynomial kernel due to its ability to create flexible, local decision boundaries.  
- Polynomial kernel makes more errors, especially false positives, due to its global shape constraints.

---

## Requirements

Minimal packages required to run the project:

numpy
matplotlib
scikit-learn

Install with:

```bash
pip install -r requirements.txt
```


## How to Run
Clone the repository:
```bash
git clone https://github.com/Yaelts2/hm02_SVM.git
cd hm02_SVM
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the main assignment script (Questions 1–5):
```bash
python main.py
```

Run the high-dimensional experiment for Question 5c:
```bash
python q5c_highdim.py
```
