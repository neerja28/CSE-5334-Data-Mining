#!/usr/bin/python3

import numpy as np
import pandas as pd


class LinearRegression(object):
    def __init__(self, train_filename, test_filename):
        # self.read_train_frame = pd.read_csv("ATNT50/trainDataXY.txt", delimiter=',', dtype=None, header=None)
        # self.read_test_frame = pd.read_csv("ATNT50/testDataXY.txt", delimiter=",", dtype=None, header=None)

        self.read_train_frame = pd.read_csv(train_filename, test_filename, delimiter=',', dtype=None, header=None)
        self.read_test_frame = pd.read_csv(test_filename, delimiter=",", dtype=None, header=None)

        # Use the following code for train and test
        self.X_train = np.array(self.read_train_frame[1:])
        self.Y_train = np.array(self.read_train_frame.head(1))
        self.X_test = np.array(self.read_test_frame[1:])
        self.Y_test = np.array(self.read_test_frame.head(1))

        self.max_Y_train = np.max(self.Y_train)
        self.max_Y_test = np.max(self.Y_test)

        # print(self.X_train.shape)
        # print(self.Y_train.shape)
        # print(self.X_test.shape)
        # print(self.Y_test.shape)

        self.X_train_rows, self.X_train_cols = self.X_train.shape
        self.X_test_rows, self.X_test_cols = self.X_test.shape
        self.Y_train_rows, self.Y_train_cols = self.Y_train.shape
        self.Y_test_rows, self.Y_test_cols = self.Y_test.shape

        self.Y_train_formatted = np.zeros((self.max_Y_train, self.X_train_cols))
        self.Y_test_formatted = np.zeros((self.max_Y_test, self.X_test_cols))

        # N_train is X_train_cols
        self.A_train = np.ones((1, self.X_train_cols))    # N_train : number of training instance
        # N_test is X_test_cols
        self.A_test = np.ones((1, self.X_test_cols))      # N_test  : number of test instance

        # Final train and test data
        # Xtrain, Xtest is X_Train, X_test
        self.Xtrain_padding = np.row_stack((self. X_train,self.A_train))
        self.Xtest_padding = np.row_stack((self.X_test, self.A_test))

        self.format_class_labels()

    def format_class_labels(self):
        for temp in range(self.Y_train_cols):
            self.Y_train_formatted[self.Y_train[0, temp]-1, temp] = 1

        # print(self.Y_train_formatted)

        for temp in range(self.Y_test_cols):
            self.Y_test_formatted[self.Y_test[0, temp]-1, temp] = 1

    def compute_coefficients(self):
        """computing the regression coefficients"""

        # Ytrain is Y_train
        B_padding = np.dot(np.linalg.pinv(self.Xtrain_padding.T), self.Y_train_formatted.T)   # (XX')^{-1} X  * Y'  #Ytrain : indicator matrix

        Ytest_padding = np.dot(B_padding.T, self.Xtest_padding)

        Ytest_padding_argmax = np.argmax(Ytest_padding, axis=0) + 1

        print(Ytest_padding)

        print("Predicted labels")
        print(Ytest_padding_argmax)
        print("True/Ground truth labels")
        print(self.Y_test[:])

        err_test_padding = self.Y_test - Ytest_padding_argmax

        print("Error")
        print(err_test_padding)

        # Calculate accuracy
        count = 0
        for temp in err_test_padding[0, :]:
            if temp == 0:
                count += 1

        print(count)

        TestingAccuracy_padding = (count / len(err_test_padding[0, :])) * 100

        print("Accuracy: ", TestingAccuracy_padding)


if __name__ == "__main__":
    obj = LinearRegression("ATNT50/trainDataXY.txt", "ATNT50/testDataXY.txt")
    obj.compute_coefficients()
