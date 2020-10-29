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


class KNN:
    """K-NN Classifier"""

    def __init__(self, neighbors: int):
        """k-NN constructor
        @param neighbors: Number of nearest neighbors
        """
        self.k = neighbors

    def get_k_neighbors(self, data: np.ndarray, new_sample: list):
        """Get the k nearest neighbors"""
        neighbors = []
        for n in range(len(data)):
            # Extract the features
            x = data[n].tolist()[:-1]
            # Extract the last column (target)
            y = data[n].tolist()[-1]
            # Calculate distances
            dist = distance(x, new_sample)
            # Push the distance and neighbors onto the heap
            heapq.heappush(neighbors, (dist, n, y))
        # Store off the neighbors with the smallest distance
        kNeighbors = heapq.nsmallest(self.k, neighbors)
        # Return the neighbors with the smallest distance
        return kNeighbors

    def classify(self, trainData: np.ndarray, testData: np.ndarray):
        """Estimate the class of each of the data point in the testData array"""
        classifications = []
        for sample in testData:
            # Create a list from the data source that we take in
            new_vector = sample.tolist()[:-1]
            # Get the k-nearest neighbors
            neighbors = self.get_k_neighbors(trainData, new_vector)
            # Get the target values of the k-neighbors
            votes = [trainData[n[1]].tolist()[-1] for n in neighbors]
            # Most common class depending on the vote data above
            estimate = most_common_class(votes)
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
        # Initialize kNN object
        knn_classifier = KNN(neighbors=k)

        # Cross-validation
        f1sc = []
        for v in range(kval):
            # Get the prediction results for the validation set
            results = np.array(knn_classifier.classify(train[v], test[v]))
            f1sc.append(f1_score(results[:, 0], results[:, 1], len(np.unique(results[:, 0]))))

        f1scores[:, count] = f1sc
        count += 1

    pd.DataFrame({'kNN-10': f1scores[:, 0], 'kNN-9': f1scores[:, 1], 'kNN-8': f1scores[:, 2], 'kNN-7': f1scores[:, 3],
                  'kNN-6': f1scores[:, 4], 'kNN-5': f1scores[:, 5],
                  'kNN-4': f1scores[:, 5]}).plot(kind='box', title="Macro F1-score. k=4-10")
