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
    print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n")
    print("\tTitanic Dataset - Prediction With Random Forests\n")
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

    # preprocess - test
    print("Preprocessing the testing data... ")
    test_df = test_df.drop(['Name'], axis=1)
    test_df = test_df.drop(['Ticket'], axis=1)
    test_df = test_df.fillna(0)

    print("Done.\n")

    print("\nModelling\n")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=0)

    model = classify_randomforests(X_scaled, y, X_train, X_test, y_train, y_test)

    print("Finished modelling\n")
    # print(f"Format of the dataset after all the encoding:\n{X_train[0]}")
    print(f"Shape of the dataset:\n{X_train.shape}")
    # print(f"Type of the dataset:\n{type(X_train)}")
    prediction = predict_single(model, train_df, 892, 1, 'male', 23, 1, 0, 82.2667, 'C156', 'S')
    print(f"Single prediction:")
    if prediction:
        print("You survived, congrats!")
    else:
        print("Sorry, you died.")


def predict_single(model, train_df, passenger_id, pclass, sex, age, sibsp, parch, fare, cabin, embarked):
    # Given parameters, predict a single user. All data entries must be nonempty.

    df = [passenger_id, pclass, sex, age, sibsp, parch, fare, cabin, embarked]
    train_df.loc[len(train_df)] = df
    train_df = train_df.fillna(0)
    train_df = pd.get_dummies(train_df)
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(train_df)
    df_scaled = np.delete(df_scaled, -1, 1)
    print(f"Shape:\n{df_scaled.shape}")
    return model.predict(df_scaled[-1].reshape(1, -1))


def classify_randomforests(X_scaled, y, X_train, X_test, y_train, y_test):
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


if __name__ == '__main__':
    main()
