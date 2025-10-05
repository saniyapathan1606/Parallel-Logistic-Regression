import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time

# Load dataset
df = pd.read_csv("D:\Parallel-Logistic-Regression\data\heart.csv")
X = df.drop('target', axis=1).values
y = df['target'].values.reshape(-1,1)

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Compute gradient
def compute_gradient_seq(X, y, weights):
    m = X.shape[0]
    predictions = sigmoid(np.dot(X, weights))
    gradient = (1/m) * np.dot(X.T, (predictions - y))
    return gradient

# Training
n_features = X_train.shape[1]
weights = np.zeros((n_features,1))
learning_rate = 0.01
epochs = 100
losses_seq = []

start_time = time.time()
for epoch in range(epochs):
    gradient = compute_gradient_seq(X_train, y_train, weights)
    weights -= learning_rate * gradient
    
    # Loss
    predictions = sigmoid(np.dot(X_train, weights))
    loss = -np.mean(y_train*np.log(predictions) + (1-y_train)*np.log(1-predictions))
    losses_seq.append(loss)
end_time = time.time()
seq_time = round(end_time - start_time, 2)

# Accuracy
y_pred = (sigmoid(np.dot(X_test, weights)) >= 0.5).astype(int)
accuracy_seq = round(np.mean(y_pred == y_test)*100,2)

print("Sequential LR Accuracy:", accuracy_seq, "%")
print("Sequential Training Time:", seq_time, "seconds")
