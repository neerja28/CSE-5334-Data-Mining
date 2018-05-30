#!/usr/bin/python3

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


class LinearRegression(object):
    def __init__(self):
        self.read_train_frame = pd.read_csv("ATNT50/trainDataXY.txt", delimiter=',', dtype=None, header=None)

        self.total_data = np.array(self.read_train_frame)
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None

        self.max_Y_train = None
        self.max_Y_test = None

        # print(self.X_train.shape)
        # print(self.Y_train.shape)
        # print(self.X_test.shape)
        # print(self.Y_test.shape)

        self.X_train_rows, self.X_train_cols = None, None
        self.X_test_rows, self.X_test_cols = None, None
        self.Y_train_cols = None
        self.Y_test_cols = None

        self.Y_train_formatted = None
        self.Y_test_formatted = None

        # N_train is X_train_cols
        self.A_train = None
        # N_test is X_test_cols
        self.A_test = None

        # Final train and test data
        # Xtrain, Xtest is X_Train, X_test
        self.Xtrain_padding = None
        self.Xtest_padding = None

    def format_class_labels(self):
        for temp in range(self.Y_train_cols):
            self.Y_train_formatted[self.Y_train[temp]-1, temp] = 1

        # print(self.Y_train_formatted)

        for temp in range(self.Y_test_cols):
            self.Y_test_formatted[self.Y_test[temp]-1, temp] = 1

    def cross_validation(self, split):
        """Use for cross validation task"""
        skf = StratifiedKFold(n_splits=split, shuffle=False)
        total_accuracy = 0.0

        # print(kf.get_n_splits())

        for train_indices, test_indices in skf.split(self.total_data[0, :], self.total_data[0, :]):
            # print(train_indices, test_indices)
            self.X_train, self.X_test = self.total_data[1:, train_indices], self.total_data[1:, test_indices]
            self.Y_train, self.Y_test = self.total_data[0, train_indices], self.total_data[0, test_indices]

            self.max_Y_train = np.max(self.Y_train)
            self.max_Y_test = np.max(self.Y_test)

            self.X_train_rows, self.X_train_cols = self.X_train.shape
            self.X_test_rows, self.X_test_cols = self.X_test.shape
            self.Y_train_cols = len(self.Y_train)
            self.Y_test_cols = len(self.Y_test)

            # print(self.X_train.shape, self.X_test.shape)

            self.Y_train_formatted = np.zeros((self.max_Y_train, self.X_train_cols))
            self.Y_test_formatted = np.zeros((self.max_Y_test, self.X_test_cols))

            # print(self.Y_train_formatted.shape, self.Y_test_formatted.shape)

            # N_train is X_train_cols
            self.A_train = np.ones((1, self.X_train_cols))  # N_train : number of training instance
            # N_test is X_test_cols
            self.A_test = np.ones((1, self.X_test_cols))  # N_test  : number of test instance

            # Final train and test data
            # Xtrain, Xtest is X_Train, X_test
            self.Xtrain_padding = np.row_stack((self.X_train, self.A_train))
            self.Xtest_padding = np.row_stack((self.X_test, self.A_test))

            self.format_class_labels()

            # print(self.X_train[:, 1])

            accuracy = self.compute_coefficients()

            total_accuracy += accuracy

        print("Average accuracy: ", total_accuracy / split)

    def compute_coefficients(self):
        """computing the regression coefficients"""

        # Ytrain is Y_train
        B_padding = np.dot(np.linalg.pinv(self.Xtrain_padding.T), self.Y_train_formatted.T)

        Ytest_padding = np.dot(B_padding.T, self.Xtest_padding)

        print(Ytest_padding)

        # We have to use axis=0 here not axis=1
        Ytest_padding_argmax = np.argmax(Ytest_padding, axis=0) + 1

        print(self.Y_train)
        print("Predicted labels")
        print(Ytest_padding_argmax)
        print("True/Ground truth labels")
        print(self.Y_test)

        err_test_padding = self.Y_test - Ytest_padding_argmax

        print("Error")
        print(err_test_padding)

        # Calculate accuracy
        count = 0
        for temp in err_test_padding:
            if temp == 0:
                count += 1

        print(count)

        TestingAccuracy_padding = (count / len(err_test_padding)) * 100

        print("Accuracy: ", TestingAccuracy_padding)

        return TestingAccuracy_padding


if __name__ == "__main__":
    obj = LinearRegression()
    obj.cross_validation(5)
