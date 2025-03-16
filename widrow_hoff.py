import numpy as np

# Input features
X = np.array([
    [2,1,-1],   
    [0,-1,-1]  
])

# Target values
d = np.array([-1, 1])  

# Initial weights
w = np.array([ 0,1, 0])  
learning_rate = 1

# Widrow-Hoff Learning Rule
for i in range(len(X)):
    net = np.dot(w, X[i])  # Compute net input
    print(f"Net: {net}")
    error = d[i] - net  # Compute error
    w =  learning_rate * error * X[i]  # Update weights
    print(f"Input: {X[i]}, Target: {d[i]}, Output: {net}, Error: {error}, Updated Weights: {w}")
    print("-" * 50)

print("Final Weights:", w)