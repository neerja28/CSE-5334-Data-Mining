#!/usr/bin/python3

import numpy as np
import pandas as pd
import math
import operator


class KnnClassification(object):
    def __init__(self, k, train_filename, test_filename):
        # self.read_train_frame = pd.read_csv("ATNT50/trainDataXY.txt", delimiter=',', dtype=None, header=None)
        # self.read_test_frame = pd.read_csv("ATNT50/testDataXY.txt", delimiter=",", dtype=None, header=None)

        self.read_train_frame = pd.read_csv(train_filename, delimiter=',', dtype=None, header=None)
        self.read_test_frame = pd.read_csv(test_filename, delimiter=",", dtype=None, header=None)

        self.k = k

        # Use the following code for trainind data and testing data
        self.X_train = np.array(self.read_train_frame[1:])
        self.Y_train = np.array(self.read_train_frame.head(1))
        self.X_test = np.array(self.read_test_frame[1:])
        self.Y_test = np.array(self.read_test_frame.head(1))

        # print(self.X_train.shape)
        # print(self.Y_train.shape)
        # print(self.X_test.shape)
        # print(self.Y_test.shape)

        # print(self.Y_train)

        self.train_rows, self.train_cols = self.X_train.shape
        self.test_rows, self.test_cols = self.X_test.shape

    def calculate_accuracy(self, true_labels, predicted_labels):
        count = 0

        for value in range(len(true_labels)):
            if true_labels[value] == predicted_labels[value]:
                count += 1

        return count/len(true_labels) * 100

    def calculate_distance(self, X, Y):
        total_sqr = 0.00
        for row in range(0, self.train_rows):
            # print(X)
            total_sqr += math.pow(X[row] - Y[row], 2)

        distance = np.sqrt(total_sqr)

        return distance

    def calculate_majority_class(self, neighbors_list):
        class_dict = {}

        # print(neighbors_list)

        for neighbor in range(len(neighbors_list)):

            class_label = self.Y_train[0, neighbors_list[neighbor]]

            if class_label in class_dict:
                class_dict[class_label] += 1
            else:
                class_dict[class_label] = 1

        sorted(class_dict.items(), key=operator.itemgetter(1), reverse=True)

        max_class = list(class_dict.keys())[0]

        return max_class

    def train(self):
        predicted_class_labels = []
        for test_instance in range(self.test_cols):
            distance_list = []
            neighbors_list = []

            for train_instance in range(self.train_cols):
                distance = self.calculate_distance(self.X_train[:, train_instance],
                                                   self.X_test[:, test_instance])
                distance_list.append((train_instance, distance))
                # print(distance_list)
            distance_list.sort(key=operator.itemgetter(1))
            for temp in range(self.k):
                neighbors_list.append(distance_list[temp][0])

            predicted_class_labels.append(self.calculate_majority_class(neighbors_list))

        print("Predicted labels")
        print(predicted_class_labels)
        print("True/Ground truth labels")
        print(self.Y_test[0, :])
        accuracy = self.calculate_accuracy(self.Y_test[0, :], predicted_class_labels)
        print("Accuracy: ", accuracy)


if __name__ == "__main__":
    obj = KnnClassification(10, "DataHandler_files/train_data.txt", "DataHandler_files/test_data.txt")
    obj.train()
