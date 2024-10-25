import numpy as np

def init_params(input_size, hidden_size, output_size):
    np.random.seed(42)
    lr = 0.01
    W1 = np.random.randn(input_size, hidden_size) * lr
    b1 = np.zeros((1, hidden_size))

    W2 = np.random.randn(hidden_size, output_size) * lr
    b2 = np.zeros((1, output_size))

    return W1, b1, W2, b2

def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    exp_z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def forward_prop(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    
    cache = (Z1, A1, Z2, A2)
    
    return A2, cache

def get_loss(y_true, y_pred):
    m = y_true.shape[0]

    loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m

    return loss

def backward_prop(X, y_true, cache, W1, W2):
    Z1, A1, Z2, A2 = cache
    m = X.shape[0]

    dZ2 = A2 - y_true
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * (Z1 > 0)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, lr):
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    
    return W1, b1, W2, b2

def predict(X, W1, b1, W2, b2):
    y_pred, _ = forward_prop(X, W1, b1, W2, b2)
    return np.argmax(y_pred, axis=1)
