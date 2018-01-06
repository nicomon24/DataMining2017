import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
'''
folds[i]['X_train'] = X.iloc[train, clf[2]]  
folds[i]['X_test'] = X.iloc[test]
folds[i]['y_train'] = y.iloc[train, clf[2]]
folds[i]['y_test'] = y.iloc[test]
'''            

'''
classifiers: ('name', model, training_set(cols index))
'''
def compute_prediction(classifiers, dataset):
    preds = pd.DataFrame()
    preds['y_true'] = dataset['TARGET']
    for clf in classifiers:
        print('Computing prediction for ',clf[0], '...')
        X = dataset.iloc[:, clf[2]]
        y = dataset['TARGET']
        tmp = [x[1] for x in (cross_val_predict(clf[1], X, y, cv=10, n_jobs=-1, method='predict_proba'))]
        preds[clf[0]] = pd.Series(tmp)
    return preds

def voting(preds, weights, threshold=[0.5]):
    #preds = compute_prediction(classifiers, dataset)
    pred = pd.DataFrame(columns=('w', 'thresh', 'score'))
    i = 0
    for w in weights:
        for thresh in threshold:
            pred.loc[i] = [w, thresh, f1_score(preds['y_true'], 
                                            pd.Series(np.average(preds.drop('y_true',axis=1), 
                                                                 axis = 1, 
                                                                 weights = w)).map(lambda x: 1 if x > thresh else 0))]
            i += 1
    return pred

def stack_predict(preds):
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_selection import RFECV
    from sklearn.model_selection import GridSearchCV
    lr = LogisticRegression(class_weight='balanced', n_jobs=-1)
    parameters = {'penalty':('l1','l2'),
                  'C':list(np.arange(0.5, 10,0.5)) + list(np.arange(0.01, 0.1, 0.01))}
    lr = GridSearchCV(lr, parameters)
    #selector = RFECV(lr, cv=3, scoring='f1')
    y_pred = cross_val_predict(lr, 
                               preds.drop('y_true',axis=1), 
                               preds['y_true'], 
                               n_jobs=-1)
    return f1_score(preds['y_true'], y_pred)

def voting_fit(preds, weights, threshold=[0.5]):
    #preds = compute_prediction(classifiers, dataset)
    pred = pd.DataFrame(columns=('w', 'thresh', 'score'))
    i = 0
    for w in weights:
        for thresh in threshold:
            pred.loc[i] = [w, thresh, f1_score(preds['y_true'], 
                                            pd.Series(np.average(preds.drop('y_true',axis=1), 
                                                                 axis = 1, 
                                                                 weights = w)).map(lambda x: 1 if x > thresh else 0))]
            i += 1
    return pred

def voting_score(preds_test, weights, threshold):
    return f1_score(preds_test['y_true'], 
             pd.Series(np.average(preds_test.drop('y_true',axis=1), axis = 1, weights = weights))
             .map(lambda x: 1 if x > threshold else 0))
    
def compute_test_pred(classifiers, train, test, weights, method = 'voting', stack_model = None):
    preds = pd.DataFrame()
    for clf in classifiers:
        print('Fitting ',clf[0], '...')
        X_train = train.iloc[:, clf[2]]
        y_train = train['TARGET']
        X_test = test.iloc[:, clf[2]]
        clf[1].fit(X_train, y_train)
        preds[clf[0]] = [x[1] for x in clf[1].predict_proba(X_test)]
    if method == 'voting':
        return pd.Series(np.average(preds, axis = 1, weights = weights[:-1])).map(lambda x: 1 if x > weights[-1] else 0)
    else:
        return stack_model.predict(preds)
    

    


        
        

        