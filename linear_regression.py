import numpy as np

# Load the data
X = np.load('data.npy')
y = np.load('labels.npy')


# Initialize the model parameters
beta = np.zeros(X.shape[1])

# Set the learning rate and the number of iterations
alpha = 0.1
n_iter = 1000

# Implement gradient descent
for i in range(n_iter):
    # Compute the predicted values
    y_pred = X.dot(beta)

    # Compute the error
    error = y - y_pred

    # Compute the gradient
    gradient = -(1/len(X)) * X.T.dot(error)

    # Update the model parameters
    beta -= alpha * gradient

# Print the model parameters
print(beta)
