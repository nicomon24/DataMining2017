import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score

def weight_tuning_voting(models, X, y, voting_method='fixed', max_w = 3):
    np.random.seed(123)
    if voting_method != 'fixed':
        N = len(models)
        MAX_W = max_w
        df = pd.DataFrame(columns=('w', 'mean', 'std'))
        a = np.arange(int(10/9*10**(N-1)), 10**(N))
        a = [list(str(x)) for x in a]
        for i in range(len(a)):
            for j in range(len(a[i])):
                a[i][j] = int(a[i][j]) % (MAX_W + 1)
        a = [tuple(x) for x in a if len(set(x)) > 1]
        a = set(a)
        #a = set([x for x in a if 0 not in x])
    else:
        a = 'hard'
    i = 0
    for w in a:
        eclf = VotingClassifier(estimators = models, weights=list(w))
        scores = cross_val_score(estimator=eclf, X=X, y=y, cv=5, scoring='f1',  n_jobs=-1)
        df.loc[i] = [w, scores.mean(), scores.std()]
        i += 1
        print 'iteration: ' + str(i) + ' w = ' + str(w) + str([w, scores.mean(), scores.std()])
    #df.sort(columns=['mean', 'std'], ascending=False)
    return df

