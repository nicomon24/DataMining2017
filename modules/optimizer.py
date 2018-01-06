'''
from xgboost import XGBClassifier
xgb = XGBClassifier(
n_estimators=100,
subsample=0.8,
colsample_bytree=0.8,
objective= 'binary:logistic',
nthread=-1,
scale_pos_weight=1,
seed=27)


parameters = {'learning_rate':list(np.arange(0.1, 0.3,0.05)),
              'gamma':list(np.arange(0, 0.5,0.1)),
              'max_depth':[4,5,6,7,8,9,10]}
'''
#per ottimizzare parametri dei modelli
#per questioni di performance uso hold out sennò ci mette ore
#esempi di utilizzi sotto
from modules.feature_sel import optimize_params_selected
from sklearn.model_selection import GridSearchCV
def optimize_params(model, params, dataset):
    X_train, X_test, y_train, y_test = train_test_split(dataset.iloc[:,1:-1], dataset['TARGET'], test_size = 0.33, random_state = 1024, stratify=dataset['TARGET'])
    clf = GridSearchCV(model, params, n_jobs=4, scoring='f1')
    clf.fit(X_train, y_train)
    test_score = f1_score(y_test, clf.predict(X_test))
    return clf.best_params_, clf.best_score_, test_score


lr = LogisticRegression(class_weight='balanced', penalty='l2', n_jobs=-1, C=31)

parameters = {'penalty':('l1','l2'),
                  'C':list(np.arange(0.001, 0.1, 0.001))}
#best {'penalty': 'l2', 'C': 1}
best, score = optimize_params(lr, parameters, df)
#best {'penalty': 'l1', 'C': 0.004} 0.554503651609 k = 10 
best_sel, score_sel, score_test_sel, support = optimize_params_selected(lr, parameters, df, 12)

#best {'learning_rate': 0.10000000000000001, 'max_depth': 4, 'gamma': 0.10000000000000001}
best_xgb, score_xgb = optimize_params(xgb, parameters, df)


rfc = RandomForestClassifier(n_estimators = 50, class_weight = 'balanced', max_depth = 10, n_jobs = 3, max_features = 'log2')

parameters = {'n_estimators':[10,30,50,100,150],
              'max_features':('log2','auto',None),
              'max_depth':[4,5,6,10],
              'min_samples_leaf':[1,10]}

#best {'max_features': 'log2', 'n_estimators': 50, 'max_depth': 10, 'min_samples_leaf': 1}
best_rfc, score_rfc = optimize_params(rfc, parameters, df)
#best {'max_features': 'log2', 'n_estimators': 30, 'max_depth': 4, 'min_samples_leaf': 10}  0.565821749796 k =12
best_sel, score_sel, score_test_sel, support = optimize_params_selected(rfc, parameters, df, 12)

parameters = {'kernel':('linear', 'sigmoid'),
              'C':list(np.arange(0.01, 1, 0.1)),
              'degree':[3,5,7],
              'gamma':[0.1, 0.05, 0.5]}

svr = SVC(class_weight='balanced', max_iter = 10000, degree = 3, C = 0.6, gamma=0.1, kernel='poly')

#best {'kernel': 'linear', 'C': 1, 'gamma': 0, 'degree': 3}
best_svc, score_svc = optimize_params(svr, parameters, df)

#best {'kernel': 'linear', 'C': 0.81000000000000005, 'gamma': 0.1, 'degree': 3} 0.538055764883 k = 6
best_svc, score_svc, score_test_svc, support = optimize_params_selected(svr, parameters, df, 6)

knn = KNeighborsClassifier(n_jobs = 4)

parameters = {'n_neighbors': list(np.arange(5,100,5)),
              'weights': ('uniform','distance'),
              'p':[1,2,5]}
#best {'n_neighbors': 65, 'weights': 'uniform', 'p': 2} 0.489307720188 k = 6
best_knn, score_knn, score_test_knn, support_knn = optimize_params_selected(knn, parameters, df, 6)


base_clf = DecisionTreeClassifier(class_weight='balanced', max_depth = 1)
ada = AdaBoostClassifier(base_clf)
parameters={'n_estimators':list(np.arange(10, 100, 15)),
            'learning_rate':[1.5, 2.5,3, 3.5]}
#best {'n_estimators': 25, 'learning_rate': 0.25}  0.525 k = 15 ??? 0.549477267202 con {'n_estimators': 25, 'learning_rate': 1.5} e max_depth = 1
# 0.562805403852 {'n_estimators': 25, 'learning_rate': 3} k = 10 max_depth = 1
opt_param_ada, score_ada, score_test_ada, support_ada = optimize_params_selected(ada, parameters, df, 10)



