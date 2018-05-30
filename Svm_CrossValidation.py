#!/usr/bin/python3

# http://scikit-learn.org/stable/modules/svm.html

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import StratifiedKFold


class Svm(object):
    def __init__(self):
        self.read_train_frame = pd.read_csv("ATNT50/trainDataXY.txt", delimiter=',', dtype=None, header=None)

        self.clf = None

        # Use the below code for cross validation
        self.total_data = np.array(self.read_train_frame)
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None

    def cross_validation(self, split):
        """Use for cross validation task"""
        skf = StratifiedKFold(n_splits=split, shuffle=False)
        total_accuracy = 0.0

        # print(kf.get_n_splits())

        for train_indices, test_indices in skf.split(self.total_data[0, :], self.total_data[0, :]):
            print(train_indices, test_indices)
            self.X_train, self.X_test = self.total_data[1:, train_indices], self.total_data[1:, test_indices]
            self.Y_train, self.Y_test = self.total_data[0, train_indices], self.total_data[0, test_indices]
            # SVM requires instances to be represented one per row, hence transpose
            self.X_train = np.transpose(self.X_train)
            self.X_test = np.transpose(self.X_test)
            # print(self.X_train[:, 1])
            accuracy = self.train()
            total_accuracy += accuracy

        print("Average accuracy: ", total_accuracy / split)

    def train(self):
        self.clf = svm.LinearSVC()

        # Use the below code for cross validation
        self.clf.fit(self.X_train, self.Y_train[:])

        predicted_labels = self.test()

        # Use the below code for cross validation
        accuracy = self.calculate_accuracy(self.Y_test[:], predicted_labels)

        print("Accuracy: ", accuracy)

        return accuracy

    def test(self):
        predicted_labels = self.clf.predict(self.X_test)
        print("Predicted labels:")
        print(predicted_labels)
        print("Actual labels:")

        # Use the below code for cross validation
        print(self.Y_test[:])

        return predicted_labels

    def calculate_accuracy(self, true_labels, predicted_labels):
        count = 0

        for temp in range(len(true_labels)):
            if true_labels[temp] == predicted_labels[temp]:
                count += 1

        return count/len(true_labels) * 100


if __name__ == "__main__":
    obj = Svm()

    # Use the below code for cross validation
    obj.cross_validation(5)
