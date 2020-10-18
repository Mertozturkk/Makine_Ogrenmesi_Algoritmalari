import numpy as np
from matplotlib import pyplot as plt

class LogisticRegression:
    def __init__(self, n_features, n_observations):
        self.weights = np.zeros((n_features, 2))
        self.n_features = n_features
        self.n_observations = n_observations
        self.loss = []
        self.acc = []
    
    def magic_sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def forward_pass(self, X, w):
        return self.magic_sigmoid(np.matmul(X, w))
    
    def classify_or_prediction(self, X, w):
        return np.argmax(self.forward_pass(X, w), axis=1).reshape(-1,1)
    
    def compute_loss(self, X, Y, w):
        y_hat = self.forward_pass(X, w)
        return -np.average(Y * np.log(y_hat) + (1-Y)*np.log(1-y_hat))

    def compute_gradient(self, X, Y, w):
        return np.matmul(X.T, (self.forward_pass(X, w) - Y)) / self.n_observations

    def evaluate(self, X, Y, Y_test_encoded, w):
        loss = self.compute_loss(X, Y_test_encoded, w)
        correct_predictions = np.sum(self.classify_or_prediction(X, w) == Y)
        accuracy = correct_predictions / X.shape[0]
        return accuracy, loss
    def fit(self, X_train, Y_train, X_test, Y_test, Y_test_encoded, max_iter, learning_rate):
        print("Training..")
        p_loss = 0.0000
        for i in range(max_iter):            
            self.weights -= self.compute_gradient(X_train, Y_train, self.weights) * learning_rate
            validation_acc, validation_loss = self.evaluate(X_test, Y_test, Y_test_encoded, self.weights)
            training_acc, training_loss = self.evaluate(X_train, Y_train, Y_test_encoded,self.weights)
            self.acc.append(validation_acc)
            self.loss.append(validation_loss)
            if abs(p_loss - training_loss) < 0.0001:
                break
            p_loss = training_loss
            print("Iter : {} Validation Accuracy: {:.5f} Validation Loss: {:.5f}".format(i, validation_acc, validation_loss))
        print("Training process has been completed!")
    
    def plot(self):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(range(len(self.loss)), self.acc, 'r-')
        plt.ylim([0., 1.])
        plt.ylabel('Validation Accuracy')
        plt.xlabel('Iterations')
        plt.title('Accuracy: {:.3f}'.format(self.acc[-1]))

        plt.subplot(1, 2, 2)
        plt.plot(range(len(self.loss)), self.loss, 'b-')
        plt.ylabel('Loss')
        plt.xlabel('Iterations')
        plt.title('Loss: {:.3f}'.format(self.loss[-1]))
        plt.show()