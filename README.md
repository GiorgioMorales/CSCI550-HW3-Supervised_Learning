# Supervised Learning: Decision Trees and k-NN

Our system uses either the K-Nearest-Neighbor (KNN) or the Decision Tree algorithm to predict the class of input vectors. The KNN algorithm determines the nearest k neighbors and predicts the classification of the vector based on the neighboring values. This algorithm benefits when the dataset is naturally clustered. This algorithm is a lazy training algorithm, being that the model is only generated when a test set and input data is submitted, making it slower to classify

The Decision Tree is also a predictive algorithm, but it creates splitting nodes to form a tree. The splitting node looks at a vector's $x_i$ value and directs the vector based on a truthy value. A given vector then follows the paths returned by the splitting nodes until a classification is predicted. This algorithm is benefited when there are clear single axis hyper-planes in the data. This algorithm is more proactive in it's learning then the K-NN algorithm, so it is slower to train, but significantly faster to classify then K-NN.

## Usage

Below, we list the instructions that should be followed in order to use our system as well as some comments about the structure of the project:

* Clone or download our repository using the SSH URI: 

  ```git clone GiorgioMorales/CSCI550-HW3-Supervised_Learning.git```.

* Configure the project using a Python 3.x environment. The only libraries required are Numpy and Pandas. Note that the program also uses MatPlotLib 3.3.1 to visualize the results.
    
* Both the ```KNN``` and the ```Tree``` classes contain their own drivers.
    
* The segmentation dataset can be read and processed in the ```utils.py``` file.
    
* The KNN algorithm is executed typing ```python KNN.py``` in a Python terminal. By default, it will sample k-values from 4 through 10.
    
* Similarly, the Decision Tree algorithm is executed typing ```python Tree.py``` in a Python terminal. By default, it will run a single execution of the decision tree.
    
* If the user decides to utilize another dataset, what is required it so make minor modifications to the utils class. All that is necessary is to have a ```m x n``` sized discrete. input file. Then, the metadata needs to be processed as well as the index of the classifications needs to be adjusted.

## Example Use

Now, we demonstrate the behavior of our system typing the following commands in a Python terminal:

```python KNN.py```

<img src=https://www.cs.montana.edu/~moralesluna/images/supervised550/KNN.png alt="alt text" width=400 height=280>
Fig. 1: Results of using KNN on the segmentation dataset utilizing a Macro F1-Score with a 10 Fold Cross Validation.

Similarly, we show the behavior of the Decision Tree algorithm using the following command:

'''python Tree.py'''

<img src=https://www.cs.montana.edu/~moralesluna/images/supervised550/Tree.png alt="alt text" width=400 height=280>
Fig. 2: Results of using Decision Tree on the segmentation dataset utilizing a Macro F1-Score with a 10 Fold Cross Validation. The Decision Tree algorithm was tuned to have an eda=7 and pi=0.95.

## Discussion

### Choosing a k-value

To determine the optimal k-value of the KNN algorithm, we iterated through the range ```4 <= k <= 10``` and ran a 10-Fold Cross Validation on each k value. We will be comparing the results that were generated in Figure 1. We see that for the segmentation dataset, either ```k=5```, ```k=7```, or ```k=8``` appears to be the optimal values. Generally, the performance did not significantly increase or decrease except for those k values. $k=5$ had the balance of general stability as well as overall performance. $k=7$ had the highest performance, but also one of the highest variability of results.  ```k=8``` was the most stable with only a couple of outliers, but it has one of the worst mean performances of all of the measured k values. 

### Performance on subset of data

To experiment with the effectiveness of the algorithms, we took decreasing sized subsets of the original segmentation dataset. We randomly selected subsets at intervals of a $10\%$ reduction with a range of $100\%$ to $30\%$ for each algorithm. We tuned each algorithm with the most optimal values as determined by iterations of each of the parameters. While we expected to see a reduction in the performance of each algorithm, we will be comparing the means of the two macro F1-scores of each algorithm to compare each of their performances to each other. Below is the result:

<img src=https://www.cs.montana.edu/~moralesluna/images/supervised550/KNNData.png alt="alt text" width=400 height=280>
Fig. 3: Results of using KNN on the segmentation dataset utilizing a Macro F1-Score with a 10 Fold Cross Validation.

<img src=https://www.cs.montana.edu/~moralesluna/images/supervised550/Data.png alt="alt text" width=400 height=280>
Fig. 3: Results of using KNN on the segmentation dataset utilizing a Macro F1-Score with a 10 Fold Cross Validation.

As we can see in Figures 3 and 4, both algorithms had a relatively similar reduction in performance. Towards the end of the iteration, the KNN algorithm performed slightly better then the Decision tree with the reduction in the size of the dataset. With the complete dataset though, the Decision Tree had a slight reduction in variance in performance then the KNN algorithm and a smaller worst performance than the KNN algorithm.
