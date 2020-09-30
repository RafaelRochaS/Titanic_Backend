# data analysis and wrangling
import pandas as pd
import numpy as np

# machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


class Titanic:

    # TODO: change print to logging
    # TODO: Add documentation
    # TODO: Add type hints to the methods

    def __init__(self, debug=False, jobs=-1):
        self.jobs = jobs
        self.debug = debug
        self.__train_df = ""
        self.model = ""

    def train(self):

        if self.debug:
            print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n")
            print("\tTitanic Dataset - Prediction With Random Forests\n")
            print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n")

        # get the data
        if self.debug:
            print("Parsing data with pandas... ")
        self.__train_df = pd.read_csv('train.csv')
        X_scaled, y = self.__preprocessor()
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,
                                                            random_state=0)

        self.model = self.__classify_random_forests(X_scaled, y, X_train, X_test, y_train, y_test)

    def __preprocessor(self):

        # preprocessor - training
        if self.debug:
            print("Preprocessing the data... ")
        self.__train_df = self.__train_df.drop(['Name'], axis=1)
        y = self.__train_df['Survived']
        self.__train_df = self.__train_df.drop(['Survived', 'Ticket'], axis=1)
        self.__train_df = self.__train_df.fillna(0)
        if self.debug:
            print("Shape before One-Hot-Encoding: {}".format(self.__train_df.shape))
        X = pd.get_dummies(self.__train_df)
        if self.debug:
            print("Shape after One-Hot-Encoding: {}".format(X.shape))
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        if self.debug:
            print("Done preprocessing.")

        return X_scaled, y

    def test_default(self):
        if self.debug:
            print("Finished modelling\n")
        prediction = self.__predict_single(892, 1, 'female', 23, 1, 0, 82.2667, 'C156', 'S')
        if self.debug:
            print(f"Single prediction:")
        if prediction:
            print("You survived, congrats!")
        else:
            print("Sorry, you died.")

    def predict_single(self, passenger_id, pclass, sex, age, sibsp, parch, fare, cabin, embarked):
        # Wrapper for the __predict_single method
        # TODO: Find a better way to do this
        prediction = self.__predict_single(passenger_id, pclass, sex, age, sibsp, parch, fare, cabin, embarked)

        if prediction:
            print("You survived, congrats!")
        else:
            print("Sorry, you died.")
        
        return prediction

    def __predict_single(self, passenger_id, pclass, sex, age, sibsp, parch, fare, cabin, embarked):
        # Given parameters, predict a single user. All data entries must be nonempty.

        df = [passenger_id, pclass, sex, age, sibsp, parch, fare, cabin, embarked]
        new_train_df = self.__train_df
        new_train_df.loc[len(new_train_df)] = df
        new_train_df = new_train_df.fillna(0)
        new_train_df = pd.get_dummies(new_train_df)
        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(new_train_df)
        df_scaled = np.delete(df_scaled, -1, 1)
        if self.debug:
            print(f"Shape:\n{df_scaled.shape}")
        return self.model.predict(df_scaled[-1].reshape(1, -1))

    def __classify_random_forests(self, X_scaled, y, X_train, X_test, y_train, y_test):

        forest = RandomForestClassifier(n_estimators=15, n_jobs=self.jobs)
        forest.fit(X_train, y_train)
        cross_val_RandomForests = cross_val_score(forest, X_scaled, y, cv=5)
        param_grid_Forest = {'n_estimators': [5, 10, 15, 30, 40, 50],
                             'n_jobs': [self.jobs],
                             'min_samples_split': [2, 3, 5],
                             'min_samples_leaf': [1, 5, 8]}
        grid_search_Forest = GridSearchCV(RandomForestClassifier(n_jobs=self.jobs, max_depth=3), param_grid_Forest,
                                          cv=5)
        grid_search_Forest.fit(X_train, y_train)
        if self.debug:
            print("RandomForests - Training set accuracy: {:.3f}".format(forest.score(X_train, y_train)))
            print("RandomForests - Test set accuracy: {:.3f}".format(forest.score(X_test, y_test)))
            print("RandomForests - Average cross-validation score: {:.3f}".format(cross_val_RandomForests.mean()))
            print(
                "RandomForests - Test set score with Grid Search: {:.3f}".format(
                    grid_search_Forest.score(X_test, y_test)))
            print("RandomForests - Best parameters: {}".format(grid_search_Forest.best_params_))
            print("RandomForests - Best estimator: {}\n".format(grid_search_Forest.best_estimator_))

        return grid_search_Forest
