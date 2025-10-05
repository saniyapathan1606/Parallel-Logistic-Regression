# Parallel Logistic Regression

## Overview
This project implements **Parallel Logistic Regression** to speed up the training of a **binary classification model**.  
It compares **sequential vs parallel gradient descent** and demonstrates how parallel computing can reduce training time while maintaining accuracy.

**Dataset:** [Heart Disease Dataset](https://www.kaggle.com/ronitf/heart-disease-uci)

---

## Features / Innovation
- Sequential and Parallel Logistic Regression implemented from scratch.
- Optimized **parallel gradient descent** using **Joblib** with automatic CPU core detection.
- Numerically stable **loss computation** using clipped predictions.
- Smooth **training loss plots** with scatter points for visual analysis.
- Saves trained models and loss data for reproducibility.

---

## Installation & Setup
1. Clone the repository:
```bash
git clone https://github.com/saniyapathan0607/Parallel-Logistic-Regression.git
cd Parallel-Logistic-Regression
Create and activate a virtual environment (optional but recommended):

python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate


Install dependencies:

pip install -r requirements.txt


Dependencies include: numpy, pandas, scikit-learn, matplotlib, joblib, scipy

Place the dataset heart.csv in the data/ folder.

Usage

Run the main training script:

python parallel_vs_sequential_lr.py


Outputs saved in outputs/ folder:

sequential_lr_model.pkl – Sequential model weights

parallel_lr_model.pkl – Parallel model weights

losses_seq.pkl – Sequential training loss

losses_parallel.pkl – Parallel training loss

loss_comparison.png – Smoothed loss curve plot

Results & Analysis

Accuracy: Sequential vs Parallel models have similar test accuracy (~84–86%).

Training Time: Parallel training is faster due to CPU core utilization.

Loss Curves: Smooth curves show stable convergence. Scatter points highlight every 10 epochs.

Example Loss Plot:


Project Structure
Parallel-Logistic-Regression/
│
├─ parallel_vs_sequential_lr.py
├─ data/
│   └─ heart.csv
├─ outputs/
│   ├─ sequential_lr_model.pkl
│   ├─ parallel_lr_model.pkl
│   ├─ losses_seq.pkl
│   ├─ losses_parallel.pkl
│   └─ loss_comparison.png
├─ requirements.txt
└─ README.md
