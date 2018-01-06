import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import sklearn.neighbors as skn

def KNNClassifier(X_train, y_train, X_test, y_test, n_neigh):
    
    knn = skn.KNeighborsClassifier(n_neighbors = n_neigh, n_jobs = -1)
    
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    score = f1_score(y_test,y_pred)
    
    return score




