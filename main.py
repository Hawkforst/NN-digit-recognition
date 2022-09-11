class NN:
    def __init__(self, N=500, alpha=0.1):
        self.W1 = np.random.normal(size=(10, 784)) * np.sqrt(1./784)
        self.b1 = np.random.normal(size=(10, 1)) * np.sqrt(1./10)
        self.W2 = np.random.normal(size=(10, 10)) * np.sqrt(1./100)
        self.b2 = np.random.normal(size=(10, 1)) * np.sqrt((1./10))
        self.N = N
        self.leaky_relu_a = 0.05
        self.alpha = alpha
        self.predictions = None

    #@staticmethod
    def ReLu(self,Z):
        return np.where(Z > 0, Z, Z * self.leaky_relu_a)
        #np.maximum(0, Z)

    #@staticmethod
    def dReLu(self,Z):
        return np.where(Z > 0, 1, self.leaky_relu_a)
        #return 1 if Z > 0 else self.leaky_relu_a
            #Z > 0

    @staticmethod
    def sigmoid(Z):
        return 1 / (1 + np.exp(-Z))

    def dSigmoid(self,Z):
        return self.sigmoid(Z) / (1 - self.sigmoid(Z))

    @staticmethod
    def softmax(Z):
        return np.exp(Z - np.max(Z)) / np.sum(np.exp(Z - np.max(Z)), axis=0)


    @staticmethod
    def _one_hot(Y):
        one_hot_Y = np.zeros((Y.max() + 1, Y.size))
        one_hot_Y[Y, np.arange(Y.size)] = 1
        return one_hot_Y

    def _forward_prop(self, X):
        Z1 = self.W1.dot(X) + self.b1
        A1 = self.ReLu(Z1)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = self.softmax(Z2)
        return Z1, A1, Z2, A2

    def _back_prop(self, Z1, A1, Z2, A2, X, Y):
        m = X.shape[1]
        dZ2 = 2*(A2 - self._one_hot(Y))
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2, axis=1)
        dZ1 = self.W2.T.dot(dZ2) * self.dReLu(Z1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1, axis=1)
        return dW1, db1, dW2, db2

    def _update_param(self, dW1, db1, dW2, db2):
        self.W1 -= self.alpha * dW1
        self.b1 -= self.alpha * db1.reshape(self.b1.shape)
        self.W2 -= self.alpha * dW2
        self.b2 -= self.alpha * db2.reshape(self.b2.shape)

    def train(self, X, Y):
        for i in range(self.N):
            Z1, A1, Z2, A2 = self._forward_prop(X)
            dW1, db1, dW2, db2 = self._back_prop(Z1, A2, Z2, A2, X, Y)
            print(db1)
            self._update_param(dW1, db1, dW2, db2)
            #if i == 900:
            #    self.alpha = 0.05
            print(f"Learning... iteration: {i} out of {self.N}.")
            print(f"Accuracy: {self.accuracy(self.get_prediction(A2), Y)}")

        print(f"""
        ______________
        Training done.
        ______________
        Accuracy: {self.accuracy(self.get_prediction(A2), Y)}
        """)
        #print(f"Accuracy: {self.accuracy(self.get_prediction(A2), Y)}")

    @staticmethod
    def accuracy(pred, real):
        return np.sum(pred == real) / real.size

    @staticmethod
    def get_prediction(A2):
        return np.argmax(A2, 0)

    def predict(self, X_test, *y_test):
        _, _, _, A2 = self._forward_prop(X_test)
        self.predictions = self.get_prediction(A2)
        if y_test is not None:
            print(f"""
        ______________
        Prediction done.
        ______________
        Accuracy: {self.accuracy(self.predictions, np.array(y_test))}
            """)


import numpy as np
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

data = pd.read_csv('train.csv')

data.head()
data = np.array(data)

m, n = data.shape

np.random.shuffle(data)

X_ = data[0:1000]
y_train = X_[:,0]
X_train = X_[:,1:] / 255

test_set = data[1000:]
X_test = test_set[:,1:] / 255
y_test = test_set[:,0]


cnn = NN()
cnn.train(X_train.T, y_train)
cnn.predict(X_test.T, y_test)