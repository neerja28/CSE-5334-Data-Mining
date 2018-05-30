#!/usr/bin/python3

# http://scikit-learn.org/stable/modules/svm.html

import numpy as np
import pandas as pd
from sklearn import svm


class Svm(object):
    def __init__(self, train_filename, test_filename):
        # self.read_train_frame = pd.read_csv("ATNT50/trainDataXY.txt", delimiter=',', dtype=None, header=None)
        # self.read_test_frame = pd.read_csv("ATNT50/testDataXY.txt", delimiter=",", dtype=None, header=None)

        self.read_train_frame = pd.read_csv(train_filename, delimiter=',', dtype=None, header=None)
        self.read_test_frame = pd.read_csv(test_filename, delimiter=",", dtype=None, header=None)

        self.clf = None

        # Use the following code for train and test
        self.X_train = np.transpose(np.array(self.read_train_frame[1:]))
        self.Y_train = np.array(self.read_train_frame.head(1))
        self.X_test = np.transpose(np.array(self.read_test_frame[1:]))
        self.Y_test = np.array(self.read_test_frame.head(1))

        # print(self.X_train.shape)
        # print(self.Y_train.shape)
        # print(self.X_test.shape)
        # print(self.Y_test.shape)

    def train(self):
        self.clf = svm.LinearSVC()
        self.clf.fit(self.X_train, self.Y_train[0, :])

        predicted_labels = self.test()
        self.calculate_accuracy(self.Y_test[0, :], predicted_labels)

    def test(self):
        predicted_labels = self.clf.predict(self.X_test)

        print("Predicted labels:")
        print(predicted_labels)
        print("Actual labels:")

        print(self.Y_test[0, :])

        return predicted_labels

    def calculate_accuracy(self, true_labels, predicted_labels):
        count = 0

        for temp in range(len(true_labels)):
            if true_labels[temp] == predicted_labels[temp]:
                count += 1

        accuracy = count/len(true_labels) * 100

        print("Accuracy: ", accuracy)


if __name__ == "__main__":
    obj = Svm("ATNT50/trainDataXY.txt", "ATNT50/testDataXY.txt")

    obj.train()
