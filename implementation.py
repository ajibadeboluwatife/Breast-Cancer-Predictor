import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from time import time

def random_forest_train():
    # Importing the dataset
    dataset = pd.read_csv('Breast Cancer Data.csv')
    X = dataset.iloc[:, 2:32].values
    y = dataset.iloc[:, 1].values

    # Encoding categorical data (only if needed; check your data)
    labelencoder_X_1 = LabelEncoder()
    y = labelencoder_X_1.fit_transform(y)

    # Splitting the dataset (do this *once* during training)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Feature Scaling (do this *once* during training, fit on training data only)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)  # Transform test data based on training data fit

    clf = RandomForestClassifier(n_estimators=100, random_state=0) # Add random_state for reproducibility
    clf.fit(X_train, y_train)

    # Store the scaler and test data as attributes of the classifier object
    clf.scaler = sc
    clf.X_test = X_test
    clf.y_test = y_test

    return clf  # Return the trained classifier object

def randorm_forest_test(clf):
    t = time()
    output = clf.predict(clf.X_test)  # Use the stored X_test
    acc = accuracy_score(clf.y_test, output)  # Use the stored y_test
    print("The accuracy of testing data:", acc)
    print("The running time:", time() - t)

def random_forest_predict(clf, inp):
    t = time()
    inp_scaled = clf.scaler.transform(inp) # Use the stored scaler
    output = clf.predict(inp_scaled)
    acc = clf.predict_proba(inp_scaled)  # Get probabilities for both classes
    return output, acc, time() - t