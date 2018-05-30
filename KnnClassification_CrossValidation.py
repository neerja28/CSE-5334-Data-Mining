#!/usr/bin/python3

import numpy as np
import pandas as pd
import math
import operator
from sklearn.model_selection import StratifiedKFold

# Plot graph using this program
import DisplayGraph


class KnnClassification(object):
    def __init__(self, k):
        self.read_train_frame = pd.read_csv("ATNTFaceImages400.txt", delimiter=',', dtype=None, header=None)

        self.k = k

        # Use the following code for cross validation train
        self.total_data = np.array(self.read_train_frame)
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None

        self.train_rows, self.train_cols = None, None
        self.test_rows, self.test_cols = None, None

        # To plot accuracies
        self.accuracy_list = []

    def cross_validation(self, split):
        """Use for cross validation task"""
        skf = StratifiedKFold(n_splits=split, shuffle=False)
        total_accuracy = 0.0

        # print(kf.get_n_splits())

        for train_indices, test_indices in skf.split(self.total_data[0, :], self.total_data[0, :]):
            # print(train_indices, test_indices)
            self.X_train, self.X_test = self.total_data[1:, train_indices], self.total_data[1:, test_indices]
            self.Y_train, self.Y_test = self.total_data[0, train_indices], self.total_data[0, test_indices]

            # print(self.X_train[:, 1])

            self.train_rows, self.train_cols = self.X_train.shape
            self.test_rows, self.test_cols = self.X_test.shape

            accuracy = self.train()

            self.accuracy_list.append(accuracy)

            total_accuracy += accuracy

        print("Average accuracy: ", total_accuracy/split)

        # Call this method to display cv plot
        DisplayGraph.display_graph(self.accuracy_list)

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
            # class_label = self.Y_train[0, neighbors_list[neighbor]]

            # Use the below line for cross validation
            class_label = self.Y_train[neighbors_list[neighbor]]

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
        # Use the following code for train and test
        # print(self.Y_test[0, :])

        print(self.Y_test[:])

        # Use the following code for train and test

        accuracy = self.calculate_accuracy(self.Y_test[:], predicted_class_labels)
        print("Accuracy: ", accuracy)

        return accuracy


if __name__ == "__main__":
    obj = KnnClassification(3)

    # Use the below line for cross validation
    obj.cross_validation(10)
