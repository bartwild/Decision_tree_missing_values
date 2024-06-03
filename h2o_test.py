import h2o
from h2o.estimators import H2ORandomForestEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from visualization_utils import visulate_f1_and_acc
import pandas as pd
from dataset_utils import split_random_to_train_and_test_data_diff_methods, get_data
from decision_tree import DecisionTree
from utils import ATTRS_NAMES, MAX_DEPTH
import numpy as np
from sklearn.metrics import f1_score

h2o.init()

file_path = 'nursery.data'
df = pd.read_csv(file_path, header=None)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

list_of_percent_train_data = [0.1, 0.5, 1, 5, 10, 40, 60, 80]

f1 = []
acc = []
labels_for_percent_of_train_data = []

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
    f1_s = f1_score(y_test_str, y_pred, average='weighted')
    print(f'Percent of training data: {percent}%, Accuracy: {accuracy:.2f}')
    labels_for_percent_of_train_data.append('%.1f%%' % (percent))
    f1.append(f1_s)
    acc.append(accuracy)

h2o.shutdown(prompt=False)
row_attrs, class_vals = get_data("nursery.data")

list_of_f1_2 = []
unique_values, counts = np.unique(class_vals, return_counts=True)
default_prediction = unique_values[np.argmax(counts)]
for i in list_of_percent_train_data:
    print(i)
    train_data,test_data,_,_,_,_,_, _ = split_random_to_train_and_test_data_diff_methods(row_attrs, class_vals, i, 0.0)
    decision_tree = DecisionTree(train_data, MAX_DEPTH, default_prediction, method='entropy', FEM=False)
    f12 = f1_score(test_data[1], [decision_tree.predict_decision_tree(row) for i, row in enumerate(test_data[0]["attrs_vals"])], average='weighted')
    list_of_f1_2.append(f12)
visulate_f1_and_acc(f1, list_of_f1_2, labels_for_percent_of_train_data)
