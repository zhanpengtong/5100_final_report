from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd

def tune_hyperparameters(X_train, y_train):
    param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
    clf = RandomForestClassifier()
    grid_search = GridSearchCV(clf, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def main(X_train, y_train):
    best_model = tune_hyperparameters(X_train, y_train)
    return best_model

if __name__ == "__main__":
    # Assuming data is preprocessed and split
    X_train = pd.read_csv('X_train.csv')  # Adjust path as necessary
    y_train = pd.read_csv('y_train.csv')  # Adjust path as necessary
    main(X_train, y_train)
