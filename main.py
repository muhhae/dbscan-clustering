import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def lib_clustering():

    X = np.array([[1, 2], [2, 1], [3, 8], [8, 8], [6, 5]])

    dbscan = DBSCAN(eps=0.5, min_samples=2)

    dbscan.fit(X)

    labels = dbscan.labels_

    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.show()

def main():
    lib_clustering()

if __name__ == "__main__":
    main()