import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

file_handler = open("heartdisease.csv", "r")
data = pd.read_csv(file_handler, sep=",")
file_handler.close()

df = pd.DataFrame(data)
df = pd.DataFrame.dropna(df)
data = np.array(df)

xdata = np.array(data[:, :-1])
ydata = np.array(data[:, -1])
ydata = ydata.astype(int)

# print(xdata, ydata)

train_x, test_x, train_y, test_y = train_test_split(xdata, ydata, test_size=0.10, random_state=0)

cols = len(train_x[0])


def train_scaling():
    for i in range(cols):
        mean_train = np.mean(train_x[:, i])
        stdev_train = np.std(train_x[:, i])
        mean_test = np.mean(test_x[:, i])
        stdev_test = np.std(test_x[:, i])
        train_x[:, i] = (train_x[:, i] - mean_train) / stdev_train
        test_x[:, i] = (test_x[:, i] - mean_test) / stdev_test


train_scaling()

weights = [0.] * cols
bias = 0.


def sigmoid(w, x, b):
    z = np.dot(x, w) + b
    return 1 / (1 + np.exp(-z))


def deriv_w(w, b, j, lam):
    return np.dot(sigmoid(w, train_x, b) - train_y, train_x[:, j]) / len(train_x) + (lam * w[j]) / len(train_x)


def deriv_b(w, b):
    return np.sum(sigmoid(w, train_x, b) - train_y) / len(train_x)


def gradient_descent(w, b, lam, epoch, alpha):
    for i in range(epoch):
        temp = w
        for j in range(cols):
            w[j] = w[j] - alpha * deriv_w(w, b, j, lam)
        b = b - alpha * deriv_b(temp, b)
        # print(w, b)
    return w, b


model_w, model_b = gradient_descent(weights, bias, 0, 3000, 1)

print(model_w, model_b)


def predict(features, w, b, threshold):
    temp = sigmoid(w, features, b)
    if temp > threshold:
        return 1
    else:
        return 0


y_hat = []


def test_model(threshold):
    for i in range(len(test_y)):
        y_hat.append(predict(test_x[i], model_w, model_b, threshold))


def find_accuracy():
    total = 0
    for i in range(len(test_y)):
        # print("y_hat:", y_hat[i], "y:", test_y[i])
        if y_hat[i] == test_y[i]:
            total += 1
    return (total / len(test_y)) * 100


# for p in range(1, 9):
    # test_model(p / 10)
    # print("Model Accuracy:", find_accuracy(), "Threshold:", p / 10)
    # y_hat.clear()

test_model(0.5)
print("Model Accuracy:", find_accuracy())
