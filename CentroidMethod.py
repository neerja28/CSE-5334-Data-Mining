#!/usr/bin/python3

import numpy as np
import pandas as pd
import math
import operator
from collections import Counter

class CentroidMethod(object):
    def __init__(self, train_filename, test_filename):
        self.read_train_frame = pd.read_csv(train_filename, delimiter=',', dtype=None, header=None)
        self.read_test_frame = pd.read_csv(test_filename, delimiter=",", dtype=None, header=None)

        # self.read_train_frame = pd.read_csv("ATNT50/trainDataXY.txt", delimiter=',', dtype=None, header=None)
        # self.read_test_frame = pd.read_csv("ATNT50/testDataXY.txt", delimiter=",", dtype=None, header=None)

        # Use the below code for training data and testing data
        self.X_train = np.array(self.read_train_frame[1:])
        self.Y_train = np.array(self.read_train_frame.head(1))
        self.X_test = np.array(self.read_test_frame[1:])
        self.Y_test = np.array(self.read_test_frame.head(1))

        self.Y_unique_train = np.unique(self.Y_train)
        self.Y_train_count = dict(Counter(self.Y_train[0, :]))

        self.X_train_rows, self.X_train_cols = self.X_train.shape
        self.X_test_rows, self.X_test_cols = self.X_test.shape
        self.Y_unique_train_cols = len(self.Y_unique_train)

        # print(self.X_train_rows, self.Y_unique_train_cols)

        self.X_train_centroid = np.empty((self.X_train_rows, self.Y_unique_train_cols), dtype=None)

    def pre_process(self):
        temp1 = 0
        for temp in self.Y_unique_train:

            # Use the below code for train and test
            index_list = np.where(self.Y_train == temp)[1]

            X_train_temp = self.X_train[:, index_list]

            centroid_temp = self.calculate_centroid(X_train_temp, temp)

            for row in range(self.X_train_rows):
                self.X_train_centroid[row, temp1] = centroid_temp[row]

            temp1 += 1

    def calculate_accuracy(self, true_labels, predicted_labels):
        count = 0

        for value in range(len(true_labels)):
            if true_labels[value] == predicted_labels[value]:
                count += 1

        return count/len(true_labels) * 100

    def calculate_centroid(self, X_train_temp, index):
        cols = self.Y_train_count[index]

        return np.sum(X_train_temp, axis=1) / cols

    def calculate_distance(self, X, Y):
        total_sqr = 0.00
        for row in range(0, self.X_train_rows):
            # print(X)
            total_sqr += math.pow(X[row] - Y[row], 2)

        distance = np.sqrt(total_sqr)

        return distance

    def calculate_majority_class(self, closest_centroid):

        class_label = self.Y_unique_train[closest_centroid[0]]

        return class_label

    def train(self):
        predicted_class_labels = []
        for test_instance in range(self.X_test_cols):
            distance_list = []

            for train_instance in range(self.Y_unique_train_cols):
                distance = self.calculate_distance(self.X_train_centroid[:, train_instance],
                                                   self.X_test[:, test_instance])

                distance_list.append((train_instance, distance))

                # print(distance_list)

            distance_list.sort(key=operator.itemgetter(1))

            predicted_class_labels.append(self.calculate_majority_class(distance_list[0]))

        print("Predicted labels")
        print(predicted_class_labels)

        # Use the following code for training data and testing data
        print("True/Ground truth labels")
        print(self.Y_test[0, :])

        #  For Calculating accuracy
        accuracy = self.calculate_accuracy(self.Y_test[0, :], predicted_class_labels)
        print("Accuracy: ", accuracy)


if __name__ == "__main__":
    obj = CentroidMethod("ATNT50/trainDataXY.txt", "ATNT50/testDataXY.txt")
    obj.pre_process()
    obj.train()
