import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV


def score(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 1024, stratify=y)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred)
    return score

def RFE_score(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 1024, stratify=y)
    selector = RFECV(model, cv=3, scoring='f1')
    selector.fit(X_train,y_train)
    y_pred = selector.predict(X_test)
    score = f1_score(y_test, y_pred)
    return selector.get_support(indices=True), score