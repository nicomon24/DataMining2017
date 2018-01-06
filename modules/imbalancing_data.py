#imbalancing dataset module

import imblearn.combine as imb
import imblearn.over_sampling as imbov
import imblearn.under_sampling as imbun



#USAGE: imbalance_set ( records, labels, operation)

#Possible operations:
#   'smoteen' : combined (under + over) sampling, SMOTE + ENN
#   'smotetom' : comined sampling, SMOTE + Tomek 
#   'adasyn' : oversampling, ADASYN
#   'randomunder' : undersampling, random selection and removal


def imbalance_set(X, y, operation):
    
    methods = {'smoteen' : imb.SMOTEENN(), 'smotetom' : imb.SMOTETomek(), 'adasyn' : imbov.ADASYN(), 'randomunder' : imbun.RandomUnderSampler(), 'condensed' : imbun.CondensedNearestNeighbour(n_jobs=-1)}
    
    sm = methods[str(operation)]
    
    X_resampl, y_resampl = sm.fit_sample(X, y)
    
    return X_resampl, y_resampl

