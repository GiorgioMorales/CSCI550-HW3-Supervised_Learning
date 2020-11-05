from utils import *
import pandas as pd
from collections import Counter
import math
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


class Decision:
    def __init__(self, split=None, left=None, right=None):
        if right is None:
            right = []
        if left is None:
            left = []
        self.split = split
        self.right = right
        self.left = left

    def addLeft(self, l):
        self.left = l

    def addRight(self, r):
        self.right = r


class Leaf:
    def __init__(self, c):
        self.c = c


class Split:
    def __init__(self, xi, v):
        self.xi = xi
        self.v = v


class Tree:
    """Decision Tree Classifier"""

    def __init__(self, eda, pi):
        """Decision Tree constructor
        """
        self.eda = eda
        self.pi = pi


    def gain(self, D, Dy, Dn, k):
        temp, pD = np.unique(D[:, -1], return_counts=True)
        temp, pY = np.unique(Dy[:, -1], return_counts=True)
        temp, pN = np.unique(Dn[:, -1], return_counts=True)
        hD = 0
        hDy = 0
        hDn = 0
        for i in pD:
            hD += i/len(D)*math.log2(i/len(D))
        for i in pY:
            hDy += i/len(Dy)*math.log2(i/len(Dy))
        for i in pN:
            hDn += i/len(Dn)*math.log2(i/len(Dn))
        hD *= -1
        hDy *= -1
        hDn *= -1
        temp = len(Dy)/len(D)*hDy + len(Dn)/len(D)*hDn
        return hD - temp

    def evaluate(self, data, X, xi, c):
        data = data[X.argsort()]
        M = []
        ni = np.zeros(len(c))
        Nvi = []
        for j in range(len(data) - 1):
            ni[c.tolist().index(data[j][-1])] += 1
            if X[j + 1] != X[j]:
                v = X[j] + (X[j + 1] - X[j]) / 2
                M.append(v)
                Nvi.append(ni.copy())
        ni[c.tolist().index(data[-1][-1])] += 1
        value = None
        score = 0
        for v in M:
            Dy = data[np.where(data[:, xi] <= v)[0]]
            Dn = data[np.where(data[:, xi] > v)[0]]
            s = self.gain(data, Dy, Dn, len(c))
            if s > score:
                value = v
                score = s
        return value, score

    def tree(self, trainData: np.ndarray):
        data = trainData
        n = len(data)
        y, nj = np.unique(data[:, -1], return_counts=True)
        purity = np.max(nj / n)
        if n <= self.eda or purity >= self.pi:
            return y[np.argmax(nj / n)]
        split = None
        score = 0
        for i in range(len(data[:-1][0])):
            v, s = self.evaluate(trainData, data[:, i], i, y)
            if s > score:
                split = Split(i, v)
                score = s
        DY = data[np.where(data[:, split.xi] <= split.v)[0]]
        DN = data[np.where(data[:, split.xi] > split.v)[0]]
        return Decision(split=split, left=self.tree(DY), right=self.tree(DN))

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
    # Initialize Decision Tree object
    tree_classifier = Tree(eda=5, pi=0.7)
    # Cross-validation
    f1sc = []
    for v in range(kval):
        tree = tree_classifier.tree(trainData=train[v])
        # Get the prediction results for the validation set
        results = np.array(tree_classifier.classify(tree, test[v]))
        f1sc.append(f1_score(results[:, 0], results[:, 1], len(np.unique(results[:, 0]))))

        f1scores[:, count] = f1sc
        count += 1

    pd.DataFrame({'kNN-10': f1scores[:, 0], 'kNN-9': f1scores[:, 1], 'kNN-8': f1scores[:, 2], 'kNN-7': f1scores[:, 3],
                  'kNN-6': f1scores[:, 4], 'kNN-5': f1scores[:, 5],
                  'kNN-4': f1scores[:, 5]}).plot(kind='box', title="Macro F1-score. k=4-10")
