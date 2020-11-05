from utils import *
import pandas as pd
from collections import Counter
import heapq

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    Static Functions
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def distance(x1, x2):
    """Euclidean distance"""
    d = 0
    # For each of the features in the feature vector x1
    for i in range(len(x1)):
        d += (abs(x1[i] - x2[i]) ** 2)
    d = d ** (1 / 2)
    return d


def most_common_class(votes: list):
    """Order by frequency and select the most common vote"""
    freqDict = Counter(votes)
    return int(freqDict.most_common(1)[0][0])


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    Class Definition
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class node:
    def __init__(self, c, split=None):
        self.c = c
        self.split = split
        self.child = []

    def getVal(self):
        if self.split:
            return self.split
        else:
            return self.c

    def getLeft(self):
        return self.child[0]

    def getRight(self):
        return self.child[1]


class Split:
    def __init__(self, xi, v):
        self.xi = xi
        self.v = v


class Tree:
    """Decision Tree Classifier"""

    def __init__(self):
        """Decision Tree constructor
        """
        self.node = None

    def evaluate(self, data, X):
        return 1,2

    def tree(self, trainData: np.ndarray, eda, pi):
        data = trainData.tolist()
        n = len(data[:-1])
        y, nj = np.unique(data[-1], return_counts=True)
        purity = np.max(nj/n)
        if n <= eda or purity >= pi:
            return y[np.argmax(nj/n)]
        split = None
        score = 0
        for i in range(len(data[:-1][0])):
            v, s = self.evaluate(trainData, data[:, i])
            if s > score:
                split = Split(i, v)
                score = s
        DY = np.take(data, np.argwhere(data[:, split.xi] <= split.v))


    def classify(self, tree, testData: np.ndarray):
        """Estimate the class of each of the data point in the testData array"""
        classifications = []
        for sample in testData:
            # Create a list from the data source that we take in
            new_vector = sample.tolist()[:-1]
            estimate = None
            # Add the ground truth-estimate pair to the list too be returned
            classifications.append([int(sample.tolist()[-1]), estimate])
        return classifications

if __name__ == '__main__':
    # Load Image Segmentation dataset
    datas = readDataset()

    # Get k-folds
    kval = 10  # Number of folds that will be used for k-fold cross-validation
    train, test = k_fold(datas, kval)

    # Try different values of k (4 - 10)
    f1scores = np.zeros((kval, 7))
    count = 0
    for k in range(4, 11):
        # Initialize Decision Tree object
        tree_classifier = Tree()
        tree = tree_classifier.tree(trainData=train, eda=5, pi=0.1)
        # Cross-validation
        f1sc = []
        for v in range(kval):
            # Get the prediction results for the validation set
            results = np.array(tree_classifier.classify(train[v], test[v]))
            f1sc.append(f1_score(results[:, 0], results[:, 1], len(np.unique(results[:, 0]))))

        f1scores[:, count] = f1sc
        count += 1

    pd.DataFrame({'kNN-10': f1scores[:, 0], 'kNN-9': f1scores[:, 1], 'kNN-8': f1scores[:, 2], 'kNN-7': f1scores[:, 3],
                  'kNN-6': f1scores[:, 4], 'kNN-5': f1scores[:, 5],
                  'kNN-4': f1scores[:, 5]}).plot(kind='box', title="Macro F1-score. k=4-10")