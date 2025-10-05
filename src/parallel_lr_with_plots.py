# parallel_vs_sequential_lr.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import pickle
import time
import os
from scipy.ndimage import uniform_filter1d
from multiprocessing import cpu_count
from math import ceil

# ---------- Step 0: Create outputs folder ----------
output_dir = r"D:\Parallel-Logistic-Regression\outputs"
os.makedirs(output_dir, exist_ok=True)

# ---------- Step 1: Load dataset ----------
df = pd.read_csv(r"D:\Parallel-Logistic-Regression\data\heart.csv")
X = df.drop('target', axis=1).values
y = df['target'].values.reshape(-1,1)

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ---------- Step 2: Sigmoid ----------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ---------- Step 3: Gradient ----------
def compute_gradient(X_batch, y_batch, weights):
    m = X_batch.shape[0]
    predictions = sigmoid(np.dot(X_batch, weights))
    gradient = (1/m) * np.dot(X_batch.T, (predictions - y_batch))
    return gradient

# ---------- Step 4: Optimized Parallel Gradient Descent ----------
def parallel_gradient_descent_optimized(X, y, weights, learning_rate=0.01, n_jobs=-1):
    n_samples = X.shape[0]
    if n_jobs == -1:
        n_jobs = cpu_count()
    batch_size = ceil(n_samples / n_jobs)
    batches = [(X[i:i+batch_size], y[i:i+batch_size]) for i in range(0, n_samples, batch_size)]
    
    gradients = Parallel(n_jobs=n_jobs)(
        delayed(compute_gradient)(X_batch, y_batch, weights) for X_batch, y_batch in batches
    )
    
    avg_gradient = np.mean(gradients, axis=0)
    weights -= learning_rate * avg_gradient
    return weights

# ---------- Step 5: Training Parameters ----------
n_features = X_train.shape[1]
learning_rate = 0.01
epochs = 100
eps = 1e-15  # for safe log computation

# ---------- Step 6: Sequential Logistic Regression ----------
weights_seq = np.zeros((n_features,1))
losses_seq = []

print("Training Sequential Logistic Regression...")
start_time_seq = time.time()
for epoch in range(epochs):
    # Gradient update
    predictions = np.clip(sigmoid(np.dot(X_train, weights_seq)), eps, 1-eps)
    gradient = (1/X_train.shape[0]) * np.dot(X_train.T, (predictions - y_train))
    weights_seq -= learning_rate * gradient
    
    # Loss
    loss = -np.mean(y_train*np.log(predictions) + (1-y_train)*np.log(1-predictions))
    losses_seq.append(loss)
    
    if epoch % 10 == 0:
        print(f"[Sequential] Epoch {epoch}, Loss: {loss:.4f}")
end_time_seq = time.time()
seq_time = round(end_time_seq - start_time_seq, 2)

# ---------- Step 7: Parallel Logistic Regression ----------
weights_parallel = np.zeros((n_features,1))
losses_parallel = []

print("\nTraining Parallel Logistic Regression...")
start_time_parallel = time.time()
for epoch in range(epochs):
    weights_parallel = parallel_gradient_descent_optimized(
        X_train, y_train, weights_parallel, learning_rate, n_jobs=-1
    )
    
    # Loss
    predictions = np.clip(sigmoid(np.dot(X_train, weights_parallel)), eps, 1-eps)
    loss = -np.mean(y_train*np.log(predictions) + (1-y_train)*np.log(1-predictions))
    losses_parallel.append(loss)
    
    if epoch % 10 == 0:
        print(f"[Parallel] Epoch {epoch}, Loss: {loss:.4f}")
end_time_parallel = time.time()
parallel_time = round(end_time_parallel - start_time_parallel, 2)

# ---------- Step 8: Test Accuracy ----------
y_pred_seq = (sigmoid(np.dot(X_test, weights_seq)) >= 0.5).astype(int)
accuracy_seq = round(np.mean(y_pred_seq == y_test)*100, 2)

y_pred_parallel = (sigmoid(np.dot(X_test, weights_parallel)) >= 0.5).astype(int)
accuracy_parallel = round(np.mean(y_pred_parallel == y_test)*100, 2)

print(f"\nSequential LR Accuracy: {accuracy_seq}%, Training Time: {seq_time}s")
print(f"Parallel LR Accuracy: {accuracy_parallel}%, Training Time: {parallel_time}s")

# ---------- Step 9: Save Models ----------
with open(os.path.join(output_dir, "sequential_lr_model.pkl"), "wb") as f:
    pickle.dump(weights_seq, f)
with open(os.path.join(output_dir, "parallel_lr_model.pkl"), "wb") as f:
    pickle.dump(weights_parallel, f)

# ---------- Step 10: Save Losses ----------
with open(os.path.join(output_dir, "losses_seq.pkl"), "wb") as f:
    pickle.dump(losses_seq, f)
with open(os.path.join(output_dir, "losses_parallel.pkl"), "wb") as f:
    pickle.dump(losses_parallel, f)

# ---------- Step 11: Smooth and Plot Loss ----------
losses_seq_smooth = uniform_filter1d(losses_seq, size=3)
losses_parallel_smooth = uniform_filter1d(losses_parallel, size=3)

plt.figure(figsize=(10,6))
plt.plot(range(epochs), losses_seq_smooth, label='Sequential', color='red', linewidth=2)
plt.plot(range(epochs), losses_parallel_smooth, label='Parallel', color='blue', linewidth=2)

x_points = list(range(0, epochs, 10))
plt.scatter(x_points, losses_seq_smooth[0:epochs:10], color='red', marker='o', s=70, zorder=2, label='_nolegend_')
plt.scatter(x_points, losses_parallel_smooth[0:epochs:10], color='blue', marker='x', s=70, zorder=2, label='_nolegend_')

plt.title("Training Loss Comparison (Smoothed)", fontsize=16)
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()

plt.savefig(os.path.join(output_dir, "loss_comparison.png"), dpi=150)
plt.show()
print("Loss plot saved as loss_comparison.png")
