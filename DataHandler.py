#!/usr/bin/python3
import numpy as np
import pandas as pd

# import a program
import KnnClassification
import CentroidMethod
import LinearRegression
import Svm

classes = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
data_partitions = (5, 34)
filename = "HandWrittenLetters.txt"
train_filename = "DataHandler_files/train_data.txt"
test_filename = "DataHandler_files/test_data.txt"


def data_handler():
    index_list = list()
    train_list = list()
    test_list = list()

    train_num, test_num = data_partitions[0], data_partitions[1]

    print(train_num, test_num)

    data = pd.read_csv(filename, delimiter=',', dtype=None, header=None)

    # We should Create numpy array for manipulation
    numpy_data = np.array(data)
    labels = np.array(data.head(1))

    for data_class in classes:

        # index_list is a list of numpy array int64 type
        index_list.append(np.where(labels == data_class)[1])

    for one_class in index_list:
        train_list.extend(one_class[0:train_num])
        test_list.extend(one_class[train_num:])

    print(train_list)
    print(test_list)

    train = np.array(numpy_data[:, train_list])
    test = np.array(numpy_data[:, test_list])

    print(train.shape, test.shape)

    np.savetxt(train_filename, train, delimiter=',', fmt='%i')
    np.savetxt(test_filename, test, delimiter=',', fmt='%i')

    # Call programs
    # Knn
    print("KNN classifier")
    obj = KnnClassification.KnnClassification(10, train_filename, test_filename)
    obj.train()

    # Centroid method
    print("Centroid classifier")
    obj = CentroidMethod.CentroidMethod(train_filename, test_filename)
    obj.pre_process()
    obj.train()

    # Linear Regression
    print("Linear regression")
    obj = LinearRegression.LinearRegression(train_filename, test_filename)
    obj.compute_coefficients()

    # SVM
    print("SVM classifier")
    obj = Svm.Svm(train_filename, test_filename)
    obj.train()


if __name__ == "__main__":
    data_handler()
