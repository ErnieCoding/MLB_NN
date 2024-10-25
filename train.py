import pandas as pd
import numpy as np
import cv2
import os
from nn import *

data = 'labels.csv'
images = 'images'

df = pd.read_csv(data)

X = []
y = []

for _, row in df.iterrows():
    image_path = os.path.join(images, row['pth'])
    label = row['label']

    img = cv2.imread(image_path)

    img = cv2.resize(img, (64,64))
    img = img / 255.0

    X.append(img)
    y.append(label)


unique_labels = list(set(y))
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

y = [label_mapping[label] for label in y]

X = np.array(X).reshape(len(X), -1)
y = np.array(y).astype(int)

num_classes = len(np.unique(y))
y_onehot = np.zeros((y.shape[0], num_classes))
y_onehot[np.arange(y.shape[0]), y] = 1

input_size = X.shape[1]
hidden_size = 256
output_size = num_classes
lr = 0.02
num_epochs = 500

W1, b1, W2, b2 = init_params(input_size, hidden_size, output_size)

for epoch in range(num_epochs):
    y_pred, cache = forward_prop(X, W1, b1, W2, b2)

    loss = get_loss(y_onehot, y_pred)

    dW1, db1, dW2, db2 = backward_prop(X, y_onehot, cache, W1, W2)

    W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, lr)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

y_pred_train = predict(X, W1, b1, W2, b2)
accuracy = np.mean(y_pred_train == np.argmax(y_onehot, axis=1))
print(f"Training Accuracy: {accuracy * 100:.2f}%")


