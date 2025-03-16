# import numpy as np

# def hebbian_learning(W, X, c=1, lam=1):
#     for i, x in enumerate(X):  # Iterate over each input vector
#         net = np.dot(W.T, x)  # Compute net input
#         fnet = 2 / (1 + np.exp(-lam * net)) - 1  # Activation function
#         W = W + c * fnet * x  # Hebbian weight update
#         print(f"Updated weight vector after X{i+1}:")
#         print(W)
#     return W

# # Initial weight vector W (4x1)
# W = np.array([[1], [-1], [0], [0.5]])

# # Input vectors (each one is 4x1)
# X = [
#     np.array([[1], [-2], [1.5], [0]]), 
#     np.array([[1], [-0.5], [-2], [1.5]]), 
#     np.array([[0], [1], [-1], [-1.5]])
# ]

# # Train using Hebbian learning
# W_new = hebbian_learning(W, X)

# print("Final updated weight vector W:")
# print(W_new)
import numpy as np

def hebbian_learning(W, X, c=1, lam=1):
    for i, x in enumerate(X):  # Iterate over each input vector
        net = np.dot(W.T, x)  # Compute net input
        fnet = 2 / (1 + np.exp(-lam * net)) - 1  # Activation function
        W = W + c * fnet * x  # Hebbian weight update
        print(f"Updated weight vector after X{i+1}:")
        print(np.round(W, 2))  # Round to 2 decimal places
    return W

# Initial weight vector W (4x1)
W = np.array([[1], [-1], [0], [0.5]])

# Input vectors (each one is 4x1)
X = [
    np.array([[1], [-2], [1.5], [0]]), 
    np.array([[1], [-0.5], [-2], [1.5]]), 
    np.array([[0], [1], [-1], [-1.5]])
]

# Ensure correct shape of W and X
W = np.array(W).reshape(4, 1)
X = [np.array(x).reshape(4, 1) for x in X]

# Train using Hebbian learning
W_new = hebbian_learning(W, X)

print("Final updated weight vector W:")
print(np.round(W_new, 2))  # Final output rounded as well
