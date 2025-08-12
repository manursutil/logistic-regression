# Logistic Regression from Scratch (Breast Cancer Classification)

This project implements logistic regression from scratch using NumPy, without relying on high-level machine learning libraries.

It trains a binary classifier to predict whether a tumor is malignant or benign using the Breast Cancer Wisconsin dataset from scikit-learn.

## Features

- Implements the sigmoid function for probability mapping.
- Computes the gradient of the cost function.
- Uses batch gradient descent for optimization.
- Supports early stopping via gradient norm tolerance.
- Provides prediction functions for both probabilities and binary classes.
- Includes a complete training and evaluation pipeline.

## How It Works

The core algorithm follows these steps:

1.  Data Preprocessing

    - Loads the breast cancer dataset.
    - Splits into training and testing sets.
    - Standardizes features for better convergence.

2.  Model Training

    - Adds a bias term to X.
    - Initializes weights to zero.
    - Iteratively updates weights via gradient descent until convergence or reaching num_iter iterations.

3.  Prediction & Evaluation

    - Predicts both probabilities and class labels.
    - Evaluates accuracy on both training and test sets.

## Usage

1. Install dependencies with uv

```bash
uv clone <repo-url>
```

2. Run the script

```bash
uv run main.py
```

3. Example output

```bash
y train prediction accuracy: 0.985
y test prediction accuracy: 0.956
```

## Functions Overview

- sigmoid(z): Computes the logistic sigmoid function.
- calculate_gradient(theta, X, y): Calculates gradient for logistic regression cost.
- gradient_descent(X, y, alpha, num_iter, tol): Performs batch gradient descent to find optimal weights.
- predict_proba(X, theta): Predicts class probabilities for given features.
- predict(X, theta, threshold): Predicts binary class labels based on threshold.

## Dataset

- Name: Breast Cancer Wisconsin (Diagnostic)
- Source: sklearn.datasets.load_breast_cancer
- Classes: Malignant (1), Benign (0)
- Features: 30 numeric features (mean radius, mean texture, etc.)

## Notes

- This is a bare-bones logistic regression — no regularization, no fancy optimizers.
- Feature scaling is essential for convergence.
- Default alpha=0.1 and num_iter=100 work well for this dataset.
- Early stopping triggers if the gradient norm falls below tol.
