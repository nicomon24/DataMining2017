import pandas as pd
import numpy as np
import math
# CSV loader
from transform.loader import load_dataset
# Encoder
from transform import missing, encodings, normalization, extract_time, outliers

# sklearn objects to test
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from modules.holdout_score import *
#from modules.voting_ensemble import *

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from modules.compute_pred import *
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from modules.feature_sel import optimize_params_selected

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Embedding,LSTM
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Conv1D
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

df = load_dataset('data/TrainingSet_refined.csv', ',')
# Drop missing values
#df = drop_missing(df)
# Fill missing value, strategy TBD
df = missing.fill_missing(df)
# Label encoder
df = encodings.label_encode(df)
# One Hot encoder
#train = onehot_encode(df)
df = normalization.remove_pay_negatives(df)
df = normalization.normalize(df)
df = outliers.removeZeros(df)

#drop categorical
#df.drop(['AGE','SEX','EDUCATION', 'MARRIAGE'],axis=1,inplace=True)

extract_time.build_features(df, 'pay_delay', p_weights = [1,1,1] + [2,4,6])
extract_time.build_features(df, 'pay_amt', p_weights = [1,1,1] + [2,4,6])
extract_time.build_features(df, 'bill', p_weights = [1,1,1] + [2,4,6])

df_one_hot = df.copy()
df_one_hot = encodings.onehot_encode(df_one_hot)

target = df['TARGET']
df.drop('TARGET',axis=1,inplace=True)
df['TARGET'] = target
df = df.reset_index(drop=True)

df_one_hot.drop('TARGET',axis=1,inplace=True)
#from sklearn.preprocessing import StandardScaler
#sc_x = StandardScaler()
#df_one_hot = pd.DataFrame(sc_x.fit_transform(df_one_hot))
df_one_hot['TARGET'] = target
df_one_hot = df_one_hot.reset_index(drop=True)

print ('Dataset is ready...')

#CLASSIFIER TUNING
lr = LogisticRegression(class_weight='balanced', n_jobs=-1)
parameters = {'penalty':('l1','l2'),
              'C':list(np.arange(0.001, 0.1, 0.001))}
print ('Tuning Logistic Regression...')
opt_param_lr, score_lr, score_test_lr, support_lr = optimize_params_selected(lr, parameters, df, 12)

rfc = RandomForestClassifier(class_weight = 'balanced', n_jobs = 4)
parameters = {'n_estimators':[10,30,50,100,150],
              'max_features':('log2','auto',None),
              'max_depth':[4,5,6,10],
              'min_samples_leaf':[1,10]}
print ('Tuning Random Forest...')
opt_param_rfc, score_rfc, score_test_rfc, support_rfc = optimize_params_selected(rfc, parameters, df, 12)

#First we tune n_estimators with some fixed parameters
xgb = XGBClassifier(learning_rate =0.1,
                    max_depth=5,
                    min_child_weight=1,
                    gamma=0.01,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective= 'binary:logistic',
                    nthread=4,
                    scale_pos_weight=3,
                    seed=27)

parameters = {'n_estimators': list(np.arange(100, 300, 50))}

opt_param_xgb1, score_xgb1, score_test_xgb1, support_xgb = optimize_params_selected(xgb, parameters, df, 10)

#second round --> max_depth and min_child_weight
xgb = XGBClassifier(learning_rate =0.1,
                    max_depth=5,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective= 'binary:logistic',
                    nthread=4,
                    scale_pos_weight=3,
                    seed=27,
                    n_estimators = opt_param_xgb1['n_estimators'])

parameters = {'max_depth': list(np.arange(1, 5, 1)),
              'min_child_weight': list(np.arange(1, 10, 1)),
              'scale_pos_weight': list(np.arange(2,4, 0.5)),
              'n_estimators': list(np.arange(10, 300, 50))}

opt_param_xgb2, score_xgb2, score_test_xgb2, support_xgb = optimize_params_selected(xgb, parameters, df, 10)

xgb = XGBClassifier(learning_rate =0.1,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective= 'binary:logistic',
                    nthread=4,
                    scale_pos_weight=opt_param_xgb2['scale_pos_weight'],
                    seed=27,
                    n_estimators=opt_param_xgb2['n_estimators'],
                    max_depth = opt_param_xgb2['max_depth'],
                    min_child_weight = opt_param_xgb2['min_child_weight'])

parameters = {'gamma': list(np.arange(0.1,1,0.1))}

opt_param_xgb3, score_xgb3, score_test_xgb3, support_xgb = optimize_params_selected(xgb, parameters, df, 10)

xgb = XGBClassifier(learning_rate =0.1,
                    gamma=opt_param_xgb3['gamma'],
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective= 'binary:logistic',
                    nthread=4,
                    scale_pos_weight=opt_param_xgb2['scale_pos_weight'],
                    seed=27,
                    n_estimators=opt_param_xgb1['n_estimators'],
                    max_depth = opt_param_xgb2['max_depth'],
                    min_child_weight = opt_param_xgb2['min_child_weight'])

parameters = {'subsample': list(np.arange(0.1, 1, 0.1)),
              'colsample_bytree': list(np.arange(0.5, 1, 0.1))}

opt_param_xgb4, score_xgb4, score_test_xgb4, support_xgb = optimize_params_selected(xgb, parameters, df, 10)

xgb = XGBClassifier(learning_rate =0.1,
                    gamma=opt_param_xgb3['gamma'],
                    subsample=opt_param_xgb4['subsample'],
                    colsample_bytree=opt_param_xgb4['colsample_bytree'],
                    objective= 'binary:logistic',
                    nthread=4,
                    scale_pos_weight=opt_param_xgb2['scale_pos_weight'],
                    seed=27,
                    n_estimators=opt_param_xgb1['n_estimators'],
                    max_depth = opt_param_xgb2['max_depth'],
                    min_child_weight = opt_param_xgb2['min_child_weight'])

parameters = {'reg_alpha': list(np.arange(0.1, 1, 0.1)),
              'reg_lambda': list(np.arange(0.1, 1, 0.1))}

opt_param_xgb5, score_xgb5, score_test_xgb5, support_xgb = optimize_params_selected(xgb, parameters, df, 10)

xgb = XGBClassifier(learning_rate =0.1,
                    gamma=opt_param_xgb3['gamma'],
                    subsample=opt_param_xgb4['subsample'],
                    colsample_bytree=opt_param_xgb4['colsample_bytree'],
                    objective= 'binary:logistic',
                    nthread=4,
                    scale_pos_weight=opt_param_xgb2['scale_pos_weight'],
                    seed=27,
                    n_estimators=opt_param_xgb1['n_estimators'],
                    max_depth = opt_param_xgb2['max_depth'],
                    min_child_weight = opt_param_xgb2['min_child_weight'],
                    reg_alpha = opt_param_xgb5['reg_alpha'],
                    reg_lambda = opt_param_xgb5['reg_lambda'])


parameters = {'max_delta_step': list(np.arange(0.1,1,0.1)),
              'n_estimators': list(np.arange(10, 300, 50))}

opt_param_xgb6, score_xgb6, score_test_xgb6, support_xgb = optimize_params_selected(xgb, parameters, df, 10)

knn = KNeighborsClassifier(n_neighbors = 10, n_jobs = 4)
parameters = {'n_neighbors': list(np.arange(5,100,5)),
              'weights': ('uniform','distance'),
              'p':[1,2,5]}
print ('Tuning KNN...')
opt_param_knn, score_knn, score_test_knn, support_knn = optimize_params_selected(knn, parameters, df, 6)

parameters = {'kernel':('linear', 'sigmoid'),
              'C':list(np.arange(0.01, 1, 0.1)),
              'degree':[3,5,7],
              'gamma':[0.1, 0.05, 0.5]}

svr = SVC(class_weight='balanced', max_iter = 10000)
print ('Tuning SVC...')
opt_param_svc, score_svc, score_test_svc, support_svc = optimize_params_selected(svr, parameters, df, 6)

base_clf = DecisionTreeClassifier(class_weight='balanced', max_depth = 1)
ada = AdaBoostClassifier(base_clf)
parameters={'n_estimators':list(np.arange(10, 100, 15)),
            'learning_rate':[1.5, 2.5,3, 3.5]}
print ('Tuning AdaBoost...')
opt_param_ada, score_ada, score_test_ada, support_ada = optimize_params_selected(ada, parameters, df, 10)

score_nn = 0.52
scores = [score_test_lr, score_test_rfc, score_test_knn, score_test_xgb6, score_test_svc, score_test_ada, score_nn]

#CLASSIFIERS DECLARATION
lr = LogisticRegression(class_weight='balanced',
                        penalty=opt_param_lr['penalty'],
                        n_jobs=-1,
                        C=opt_param_lr['C'])

rfc = RandomForestClassifier(n_estimators = opt_param_rfc['n_estimators'],
                             class_weight = 'balanced',
                             max_depth = opt_param_rfc['max_depth'],
                             max_features = opt_param_rfc['max_features'],
                             min_samples_leaf=opt_param_rfc['min_samples_leaf'],
                             n_jobs = 4)

knn = KNeighborsClassifier(n_neighbors = opt_param_knn['n_neighbors'],
                           n_jobs = 4,
                           weights = opt_param_knn['weights'],
                           p = opt_param_knn['p'])

xgb = XGBClassifier(learning_rate =0.1,
                    gamma=opt_param_xgb3['gamma'],
                    subsample=opt_param_xgb4['subsample'],
                    colsample_bytree=opt_param_xgb4['colsample_bytree'],
                    objective= 'binary:logistic',
                    nthread=4,
                    scale_pos_weight=opt_param_xgb2['scale_pos_weight'],
                    seed=27,
                    n_estimators=opt_param_xgb6['n_estimators'],
                    max_depth = opt_param_xgb2['max_depth'],
                    min_child_weight = opt_param_xgb2['min_child_weight'],
                    reg_alpha = opt_param_xgb5['reg_alpha'],
                    max_delta_step = opt_param_xgb6['max_delta_step'],
                    reg_lambda = opt_param_xgb5['reg_lambda'])


svr = SVC(class_weight='balanced',
          max_iter = 10000,
          degree = opt_param_svc['degree'],
          C = opt_param_svc['C'],
          gamma= opt_param_svc['gamma'],
          kernel= opt_param_svc['kernel'],
          probability=True)

base_clf = DecisionTreeClassifier(class_weight='balanced', max_depth = 1)
ada = AdaBoostClassifier(base_clf,
                         learning_rate = opt_param_ada['learning_rate'],
                         n_estimators = opt_param_ada['n_estimators'])

def create_model():
    model = Sequential()
    model.add(Dense(30, input_dim=30,kernel_initializer='uniform', activation='softmax'))
    model.add(Dense(18, activation='softmax', kernel_initializer='normal'))
    model.add(Dense(36, activation='softmax', kernel_initializer='normal'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    return model

def create_NN(epochs=36, verbose=1):
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    model = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=128, verbose=verbose)
    estimators.append(('mlp', model))
    pipeline = Pipeline(estimators)
    return pipeline
pipeline = create_NN()

classifiers = [('lr', lr, support_lr),
               ('rfc', rfc, support_rfc),
               ('knn', knn, support_knn),
               ('xgb',xgb, support_xgb),
               ('svc', svr, support_svc),
               ('ada', ada, support_ada),
               ('nn', pipeline, np.arange(len(df.axes[1])-1))]

print ('PREDICTIONS')
pred = compute_prediction(classifiers, df)

#pred.to_csv("data/prediction.csv",index=False)

pred_train, pred_test, true_train, true_test = train_test_split(pred.drop('y_true', axis=1), pred['y_true'], test_size = 0.33, random_state = 1024, stratify = pred['y_true'])

def fit_weight(weights):
    return 1 - f1_score(true_train, pd.Series(np.average(pred_train, axis = 1, weights = weights[:-1])).map(lambda x: 1 if x > weights[-1] else 0))

def score_weight(weights):
    return f1_score(true_test, pd.Series(np.average(pred_test, axis = 1, weights = weights[:-1])).map(lambda x: 1 if x > weights[-1] else 0))

opt_weights = minimize(fit_weight, [(10*(x**2))/sum(scores) for x in scores] + [0.5],method='Powell')

print ('Voting Score: ' + str(score_weight(opt_weights['x'])))
# 0.564825253664
print ('Stack Score: ' + str(stack_predict(pred)))
# 0.547671644429

pred.to_csv("predidctions_5mod.csv",index=False)

#FINAL COMPUTATION

test_Set = load_dataset('data/Project Test Dataset.csv', ';')

# Fill missing value, strategy TBD
test_Set = missing.fill_missing(test_Set)
# Label encoder
test_Set = encodings.label_encode(test_Set)

test_Set = normalization.remove_pay_negatives(test_Set)
test_Set = normalization.normalize(test_Set)
#test_Set = outliers.removeZeros(test_Set)

test_Set.drop('TARGET', axis = 1, inplace=True)
extract_time.build_features(test_Set, 'pay_delay', p_weights = [1,1,1] + [2,4,6])
extract_time.build_features(test_Set, 'pay_amt', p_weights = [1,1,1] + [2,4,6])
extract_time.build_features(test_Set, 'bill', p_weights = [1,1,1] + [2,4,6])


test_pred = compute_test_pred(classifiers, df, test_Set, opt_weights['x'])

test_pred.to_csv('test_predictions.csv', index=False)

test_pred_id = pd.concat([test_Set.CUST_COD, test_pred], axis=1)

test_pred_id.to_csv('test_predictions_custcod.csv', index=False)