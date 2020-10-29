from math import floor
import numpy as np
from statistics import stdev as stdevfun


def readDataset():
    """Read the Image Segmentation Dataset: https://archive.ics.uci.edu/ml/datasets/Image+Segmentation"""
    # Read .data file
    with open('Data//segmentation.data') as raw:
        data = [line.strip().split(',') for line in raw.readlines()]
    # Get rid of meta data
    data = data[5:]
    # List all the classes (first column)
    class_name_list = []
    for x in data:
        if x[0] not in class_name_list:
            class_name_list.append(x[0])
    # Initialize list
    dataset = [[] for _ in range(len(class_name_list))]
    # Loop through every sample in the data set
    for sample in data:
        # Get the index of the class
        classX = class_name_list.index(sample[0])
        # Initialize list of features
        features = []
        for x in sample[1:]:
            features.append(float(x))
        # # Add the class in the last column
        # features.append(classX)
        # Append the row to the dataset
        dataset[classX].append(features)
    return normalize(dataset)


def normalize(dataset):
    """Applies z-normalization to each of the attributes"""
    for val in range(len(dataset[0][0])):
        # creates an array to hold all the real values of each attribute
        total = []
        # Go through the various classes
        for cn in range(len(dataset)):
            for i in range(len(dataset[cn])):
                # Add each of the real values in the dataset attribute to the total[]
                total.append(dataset[cn][i][val])
        # Calculate the mean of total[]
        mean = (sum(total) / len(total))
        # Calculates the standard deviation of total[]
        stdev = stdevfun(total)
        # Once the mean and std were found, transform the data
        for tcn in range(len(dataset)):
            for ti in range(len(dataset[tcn])):
                # Normalized Z-score value = (value - mean ) / standard Deviation
                if stdev != 0:
                    dataset[tcn][ti][val] = (dataset[tcn][ti][val] - mean) / stdev
    return dataset


def k_fold(dataset, kval):
    """Separates the dataset in folds ready to use cos-validation
    @param dataset: Takes the dataset coming from the readDataset function.
    @param kval: Number of folds to be used.
    """
    # Initialize folds
    train = [[] for _ in range(kval)]
    test = [[] for _ in range(kval)]
    # Populate folds
    for c in dataset:
        # Repeat for each class (Stratification)
        for k in range(1, kval + 1):
            # Append the k-th slice to "test" and the rest to "train"
            test[k - 1].append(c[floor(len(c) * ((k - 1) / kval)): floor(len(c) * (k / kval))])
            train[k - 1].append(c[floor(len(c) * (k / kval)):len(c)] + c[:floor(len(c) * ((k - 1) / kval))])

    # Reshape folds
    train2 = [[] for _ in range(kval)]
    test2 = [[] for _ in range(kval)]
    for k in range(kval):
        for i in range(len(train[0])):
            for t in train[k][i]:
                # Append the class in the last column
                t2 = t.copy()
                t2.append(i)
                train2[k].append(t2)
        for i in range(len(test[0])):
            for t in test[k][i]:
                # Append the class in the last column
                t2 = t.copy()
                t2.append(i)
                test2[k].append(t2)
    return np.array(train2), np.array(test2)


def f1_score(y_true, y_pred, n_labels):
    """Implement macro F1-score"""
    total_f1 = 0.
    for i in range(n_labels):
        yt = y_true == i
        yp = y_pred == i

        tp = np.sum(yt & yp)

        tpfp = np.sum(yp)
        tpfn = np.sum(yt)

        # Calculate precision
        if tpfp == 0:  # If there are labels with no predicted samples
            precision = 0.
        else:
            precision = tp / tpfp

        # Calculate recall
        if tpfn == 0:  # If label is not found
            recall = 0.
        else:
            recall = tp / tpfn

        # Calculate F1-score
        if precision == 0. or recall == 0.:
            f1 = 0.
        else:
            f1 = 2 * precision * recall / (precision + recall)
        total_f1 += f1
    return total_f1 / n_labels


if __name__ == '__main__':
    datas = readDataset()
    tr, te = k_fold(datas, kval=10)
