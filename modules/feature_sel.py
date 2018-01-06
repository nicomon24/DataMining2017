from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

def optimize_params_selected(model, params, dataset, n_features):
    selector = SelectKBest(score_func=f_classif, k=n_features)
    X = selector.fit_transform(dataset.iloc[:,1:-1], dataset['TARGET'])
    X_train, X_test, y_train, y_test = train_test_split(X, dataset['TARGET'], test_size = 0.33, random_state = 1024, stratify=dataset['TARGET'])
    #X_test = selector.transform(X_test)
    clf = GridSearchCV(model, params, n_jobs=4, scoring='f1')
    clf.fit(X_train, y_train)
    test_score = f1_score(y_test, clf.predict(X_test))
    return clf.best_params_, clf.best_score_, test_score, selector.get_support(indices=True)
