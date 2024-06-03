import h2o
from h2o.estimators import H2ORandomForestEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

h2o.init()

file_path = 'nursery.data'
df = pd.read_csv(file_path, header=None)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

list_of_percent_train_data = [0.1, 0.5, 1, 5, 10, 40, 60, 80]

results = []

for percent in list_of_percent_train_data:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100-percent)/100)
    
    train_df = pd.concat([X_train, y_train], axis=1)
    train_h2o = h2o.H2OFrame(train_df)

    test_df = pd.concat([X_test, y_test], axis=1)
    test_h2o = h2o.H2OFrame(test_df)

    y_col = str(y_train.name)
    x_cols = [str(col) for col in X_train.columns]

    train_h2o[y_col] = train_h2o[y_col].asfactor()
    test_h2o[y_col] = test_h2o[y_col].asfactor()

    model = H2ORandomForestEstimator(ntrees=1)
    model.train(x=x_cols, y=y_col, training_frame=train_h2o)

    predictions = model.predict(test_h2o).as_data_frame()

    y_pred = predictions['predict'].astype(str)
    y_test_str = y_test.astype(str)

    accuracy = accuracy_score(y_test_str, y_pred)
    print(f'Percent of training data: {percent}%, Accuracy: {accuracy:.2f}')
    results.append((percent, accuracy))

h2o.shutdown(prompt=False)

for percent, accuracy in results:
    print(f'Percent of training data: {percent}%, Accuracy: {accuracy:.2f}')
