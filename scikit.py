import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np

file_path = 'nursery.data'
df = pd.read_csv(file_path, header=None)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

le_y = LabelEncoder()
y = le_y.fit_transform(y)

categorical_mask = np.array([True] * X.shape[1])

for column in X.columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])

list_of_percent_train_data = [0.1, 0.5, 1, 5, 10, 40, 60, 80]

results = []

for percent in list_of_percent_train_data:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - percent) / 100)
    
    clf = HistGradientBoostingClassifier(categorical_features=categorical_mask)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    accuracy = f1_score(y_test, y_pred, average='weighted')
    print(f'Percent of training data: {percent}%, F1 Score: {accuracy:.2f}')
    results.append((percent, accuracy))

for percent, accuracy in results:
    print(f'Percent of training data: {percent}%, F1 Score: {accuracy:.2f}')
