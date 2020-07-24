# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import matplotlib.pyplot as plt

# machine learning
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# other
import os

jobs = -1


def main():
    os.system('cls')

    print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n")
    print("\tKaggle Titanic Dataset\n")
    print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n")

    # get the data
    print("Parsing data with pandas... ")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    print("Done.\n")

    # preprocess - training
    print("Preprocessing the data... ")
    train_df = train_df.drop(['Name'], axis=1)
    y = train_df['Survived']
    train_df = train_df.drop(['Survived', 'Ticket'], axis=1)
    train_df = train_df.fillna(0)
    print("Shape before One-Hot-Encoding: {}".format(train_df.shape))
    X = pd.get_dummies(train_df)
    print("Shape after One-Hot-Encoding: {}".format(X.shape))
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    print("Done.\n")

    # menu
    choice = 0

    print("\nModelling\n")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=0)

    model = 0

    while choice != 9:
        print("Choose an option:\n")
        print("1 - Classify with KNN")
        print("2 - Classify with LinearSVC")
        print("3 - Classify with Random Forests")
        print("8 - Classify with all")
        print("9 - Quit\n")

        try:
            choice = int(input('> '))
        except ValueError:
            print("Porra é essa?\n")

        if choice == 1:
            model = classify_knn(X_scaled, y, X_train, X_test, y_train, y_test)
            break
        if choice == 2:
            model = classify_linearsvc(X_scaled, y, X_train, X_test, y_train, y_test)
            break
        if choice == 3:
            model = classify_randomforests(X_scaled, y, X_train, X_test, y_train, y_test)
            break
        if choice == 8:
            classify_knn(X_scaled, y, X_train, X_test, y_train, y_test)
            classify_linearsvc(X_scaled, y, X_train, X_test, y_train, y_test)
            classify_randomforests(X_scaled, y, X_train, X_test, y_train, y_test)

            # preprocess - test
    print("Preprocessing the testing data... ")
    test_df = test_df.drop(['Name'], axis=1)
    test_df = test_df.drop(['Ticket'], axis=1)
    test_df = test_df.fillna(0)
    print("Shape before One-Hot-Encoding: {}".format(test_df.shape))

    X_testfinal = pd.get_dummies(test_df)
    for col in X:
        if len(X_testfinal.columns) == 160:
            break
        if col not in X_testfinal:
            X_testfinal[col] = 0
    print("Shape after One-Hot-Encoding: {}".format(X_testfinal.shape))
    scaler_test = MinMaxScaler()
    X_scaledfinal = scaler_test.fit_transform(X_testfinal)
    print("Done.\n")
    ids = test_df['PassengerId']
    gen_submission(model, X_scaledfinal, ids)


def classify_knn(X_scaled, y, X_train, X_test, y_train, y_test):
    print("KNN:\n")
    clf = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    clf.fit(X_train, y_train)
    cross_val_KNN = cross_val_score(clf, X_scaled, y, cv=5)
    param_grid_KNN = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                      'n_jobs': [jobs]}
    grid_search_KNN = GridSearchCV(KNeighborsClassifier(), param_grid_KNN, cv=5)
    grid_search_KNN.fit(X_train, y_train)
    print("KNN - Training set accuracy: {:.3f}".format(clf.score(X_train, y_train)))
    print("KNN - Test set accuracy: {:.3f}".format(clf.score(X_test, y_test)))
    print("KNN - Average cross-validation score: {:.3f}".format(cross_val_KNN.mean()))
    print("KNN - Test set score with Grid Search: {:.3f}".format(grid_search_KNN.score(X_test, y_test)))
    print("KNN - Best parameters: {}".format(grid_search_KNN.best_params_))
    print("KNN - Best estimator: {}\n".format(grid_search_KNN.best_estimator_))

    return grid_search_KNN


def classify_linearsvc(X_scaled, y, X_train, X_test, y_train, y_test):
    print("LinearSVC:\n")
    clf_linear = LinearSVC(C=0.01)
    clf_linear.fit(X_train, y_train)
    cross_val_LinearSVC = cross_val_score(clf_linear, X_scaled, y, cv=5)
    param_grid_LinearSVC = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
                            'dual': [True, False]}
    grid_search_LinearSVC = GridSearchCV(LinearSVC(), param_grid_LinearSVC, cv=5)
    grid_search_LinearSVC.fit(X_train, y_train)
    print("LinearSVC - Training set accuracy: {:.3f}".format(clf_linear.score(X_train, y_train)))
    print("LinearSVC - Test set accuracy: {:.3f}".format(clf_linear.score(X_test, y_test)))
    print("LinearSVC - Average cross-validation score: {:.3f}".format(cross_val_LinearSVC.mean()))
    print("LinearSVC - Test set score with Grid Search: {:.3f}".format(grid_search_LinearSVC.score(X_test, y_test)))
    print("LinearSVC - Best parameters: {}".format(grid_search_LinearSVC.best_params_))
    print("LinearSVC - Best estimator: {}\n".format(grid_search_LinearSVC.best_estimator_))

    return grid_search_LinearSVC


def classify_randomforests(X_scaled, y, X_train, X_test, y_train, y_test):
    print("Random Forests:\n")
    forest = RandomForestClassifier(n_estimators=15, n_jobs=jobs)
    forest.fit(X_train, y_train)
    cross_val_RandomForests = cross_val_score(forest, X_scaled, y, cv=5)
    param_grid_Forest = {'n_estimators': [5, 10, 15, 30, 40, 50],
                         'n_jobs': [jobs],
                         'min_samples_split': [2, 3, 5],
                         'min_samples_leaf': [1, 5, 8]}
    grid_search_Forest = GridSearchCV(RandomForestClassifier(n_jobs=jobs, max_depth=3), param_grid_Forest, cv=5)
    grid_search_Forest.fit(X_train, y_train)
    print("RandomForests - Training set accuracy: {:.3f}".format(forest.score(X_train, y_train)))
    print("RandomForests - Test set accuracy: {:.3f}".format(forest.score(X_test, y_test)))
    print("RandomForests - Average cross-validation score: {:.3f}".format(cross_val_RandomForests.mean()))
    print("RandomForests - Test set score with Grid Search: {:.3f}".format(grid_search_Forest.score(X_test, y_test)))
    print("RandomForests - Best parameters: {}".format(grid_search_Forest.best_params_))
    print("RandomForests - Best estimator: {}\n".format(grid_search_Forest.best_estimator_))

    return grid_search_Forest


def gen_submission(model, X_test, pass_ids):
    print("\n\nGenerating submission file...\n")
    predictions = model.predict(X_test)
    predictions = np.array(predictions)
    pass_ids = np.array(pass_ids)
    # pass_ids = X_test['PassengerId']
    print("Predictions shape: {}".format(predictions.shape))
    print("PassengerIDs shape: {}".format(pass_ids.shape))
    print("IDs type: {}\tPredictions type: {}".format(type(predictions), type(pass_ids)))
    submission = pd.DataFrame({'PassengerId': pass_ids, 'Survived': predictions})
    submission.to_csv("Submission.csv", index=False)
    print("File generated successfully!")


if __name__ == '__main__':
    main()
