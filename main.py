from dataset_utils import get_data, split_random_to_train_and_test_data, split_random_to_train_and_test_data_diff_methods
from decision_tree import DecisionTree
from visualization_utils import visualize_acc, visualize_metrics_of_confusion_matrix, visualize_class_counter, visualize_tree, visulate_acc_per_input_method,visulate_f1_per_input_method, visulate_acc_per_replacement_method, visulate_f1_per_replacement_method, visulate_f1_and_acc
from utils import ATTRS_NAMES, CLASS_VALUES, MAX_DEPTH, PERCENT_OF_TRAIN_DATA, ATTR_TO_INDEX
import numpy as np
import random
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from multiprocessing import Pool

row_attrs, class_vals = get_data("nursery.data")
"""
uniq_attr_vals = np.unique([i[ATTR_TO_INDEX.get("persons")] for i in row_attrs])
class_val_counter_by_attr = {
    i: {
        x: 0 for x in CLASS_VALUES
        } for i in uniq_attr_vals
}
for index, row in enumerate(row_attrs):
    class_val_counter_by_attr[row[ATTR_TO_INDEX.get("persons")]][class_vals[index]] += 1
print(class_val_counter_by_attr)
"""
"""
class_val_counter = {
    i: {} for i in ATTRS_NAMES
}

for index, row in enumerate(row_attrs):
    for i, attr in enumerate(row):
        if attr not in class_val_counter[ATTRS_NAMES[i]]:
            class_val_counter[ATTRS_NAMES[i]][attr] = {
                x: 0 for x in CLASS_VALUES
            }
        class_val_counter[ATTRS_NAMES[i]][attr][class_vals[index]] += 1

counter_1 = {}
counter_2 = {}

for index, i in enumerate(class_val_counter):
    if index < 6:
        counter_1[i] = class_val_counter[i]  
    else:
        counter_2[i] = class_val_counter[i]

visualize_class_counter(counter_1, 1)
"""
"""
train_data, test_data = split_random_to_train_and_test_data(row_attrs, class_vals, PERCENT_OF_TRAIN_DATA)
random.seed(42137)
random_forest = RandomForest(train_data, N_TREES, MAX_DEPTH, percent_of_drawn_attrs=PERCENT_OF_DRAWN_ROWS, n_attrs=N_ATTRS)
acc = random_forest.calculate_acc(test_data)
print(acc)
#for tree in random_forest.trees:
#    visualize_tree(tree, ATTRS_NAMES, "tree-small.png")
visualize_tree(random_forest.trees[0], ATTRS_NAMES, "tree-small.png")
"""
######
# compare confusion matrix metrics by class val
######
"""
train_data, test_data = split_random_to_train_and_test_data(row_attrs, class_vals, 5)
random_forest = RandomForest(train_data, N_TREES, MAX_DEPTH, percent_of_drawn_attrs=PERCENT_OF_DRAWN_ROWS, n_attrs=N_ATTRS, method='gini')
list_of_metrics = []
for class_val in CLASS_VALUES:
    matrix = random_forest.calculate_confusion_matrix(test_data, class_val)
    print(matrix)
    metrics = random_forest.calculate_metrics_of_confusion_matrix(matrix)
    list_of_metrics.append(metrics)

visualize_metrics_of_confusion_matrix(list_of_metrics, CLASS_VALUES)
"""
######
# compare acc of diff entropy calc
######
"""
list_of_percent_train_data = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80]
labels_for_percent_of_train_data = []
list_of_acc = []
for i in list_of_percent_train_data:
    print(i)
    train_data, test_data = split_random_to_train_and_test_data(row_attrs, class_vals, i)
    random_forest = DecisionTree(train_data, MAX_DEPTH, "", method='entropy')
    acc = random_forest.calculate_acc(test_data)
    list_of_acc.append(acc)
    random_forest = DecisionTree(train_data, MAX_DEPTH,"", method='gini')
    acc = random_forest.calculate_acc(test_data)
    list_of_acc.append(acc)
    labels_for_percent_of_train_data.append('%.2f%%' % (i))

visulate_acc_per_input_method(list_of_acc, labels_for_percent_of_train_data)
"""
"""
# decision tree
list_of_percent_train_data = [0.1, 0.5, 1, 5, 10, 40, 60, 80]
labels_for_percent_of_train_data = []
list_of_acc = []
list_of_f1 = []
for i in list_of_percent_train_data:
    print(i)
    train_data, test_data = split_random_to_train_and_test_data(row_attrs, class_vals, i, 0.0)
    decision_tree = DecisionTree(train_data, MAX_DEPTH, '', method='entropy', FEM=True)
    acc = decision_tree.calculate_acc(test_data)
    f1 = f1_score(test_data[1], [decision_tree.predict_decision_tree(row) for i, row in enumerate(test_data[0]["attrs_vals"])], average='weighted')
    list_of_acc.append(acc)
    list_of_f1.append(f1)
    decision_tree = DecisionTree(train_data, MAX_DEPTH,'', method='gini', FEM=True)
    acc = decision_tree.calculate_acc(test_data)
    f1 = f1_score(test_data[1], [decision_tree.predict_decision_tree(row) for i, row in enumerate(test_data[0]["attrs_vals"])], average='weighted')
    list_of_acc.append(acc)
    labels_for_percent_of_train_data.append('%.1f%%' % (i))
    list_of_f1.append(f1)
# visualize_tree(tree=decision_tree.tree, attrs_names=ATTRS_NAMES,output_name="tree.png")
visulate_acc_per_input_method(list_of_acc, labels_for_percent_of_train_data)
visulate_f1_per_input_method(list_of_f1, labels_for_percent_of_train_data)
"""
######
# compare acc and f1 of all methods
######
"""
list_of_percent_train_data = [0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80]
labels_for_percent_of_train_data = []
list_of_acc = []
list_of_f1 = []

unique_values, counts = np.unique(class_vals, return_counts=True)
default_prediction = unique_values[np.argmax(counts)]

for i in list_of_percent_train_data:

    print(i)
    train_data, test_data, train_data_mode, test_data_mode, train_data_distrib, test_data_distrib, skip_train, skip_test = \
        split_random_to_train_and_test_data_diff_methods(row_attrs, class_vals, PERCENT_OF_TRAIN_DATA, i)

    decision_tree = DecisionTree(train_data, MAX_DEPTH, default_prediction, method='entropy', FEM=False)
    acc = decision_tree.calculate_acc(test_data)
    #cm = confusion_matrix(test_data[1], [decision_tree.predict_decision_tree(row) for i, row in enumerate(test_data[0]["attrs_vals"])])
    f1 = f1_score(test_data[1], [decision_tree.predict_decision_tree(row) for i, row in enumerate(test_data[0]["attrs_vals"])], average='weighted')
    list_of_acc.append(acc)
    list_of_f1.append(f1)

    decision_tree = DecisionTree(train_data_mode, MAX_DEPTH, default_prediction, method='entropy', FEM=False)
    acc = decision_tree.calculate_acc(test_data_mode)
    #cm = confusion_matrix(test_data_mode[1], [decision_tree.predict_decision_tree(row) for i, row in enumerate(test_data_mode[0]["attrs_vals"])])
    f1 = f1_score(test_data_mode[1], [decision_tree.predict_decision_tree(row) for i, row in enumerate(test_data_mode[0]["attrs_vals"])], average='weighted')
    list_of_acc.append(acc)
    list_of_f1.append(f1)

    decision_tree = DecisionTree(train_data, MAX_DEPTH, default_prediction, method='entropy', FEM=True)
    acc = decision_tree.calculate_acc(test_data)
    #cm = confusion_matrix(test_data[1], [decision_tree.predict_decision_tree(row) for i, row in enumerate(test_data[0]["attrs_vals"])])
    f1 = f1_score(test_data[1], [decision_tree.predict_decision_tree(row) for i, row in enumerate(test_data[0]["attrs_vals"])], average='weighted')
    list_of_acc.append(acc)
    list_of_f1.append(f1)

    decision_tree = DecisionTree(skip_train, MAX_DEPTH, default_prediction, method='entropy', FEM=False)
    acc = decision_tree.calculate_acc(skip_test)
    #cm = confusion_matrix(skip_test[1], [decision_tree.predict_decision_tree(row) for i, row in enumerate(skip_test[0]["attrs_vals"])])
    f1 = f1_score(skip_test[1], [decision_tree.predict_decision_tree(row) for i, row in enumerate(skip_test[0]["attrs_vals"])], average='weighted')
    list_of_acc.append(acc)
    list_of_f1.append(f1)

    labels_for_percent_of_train_data.append('%.0f%%' % (i))

visualize_tree(tree=decision_tree.tree, attrs_names=ATTRS_NAMES,output_name="tree.png")
visulate_acc_per_replacement_method(list_of_acc, labels_for_percent_of_train_data)
visulate_f1_per_replacement_method(list_of_f1, labels_for_percent_of_train_data)
"""
######
# compare f1 and acc by missing data
######
"""
list_of_percent_train_data = [0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80]
labels_for_percent_of_train_data = []
list_of_acc = []
list_of_f1 = []
unique_values, counts = np.unique(class_vals, return_counts=True)
default_prediction = unique_values[np.argmax(counts)]
for i in list_of_percent_train_data:
    print(i)
    train_data_2,_,_,_,_,_,_, test_data = split_random_to_train_and_test_data_diff_methods(row_attrs, class_vals, PERCENT_OF_TRAIN_DATA, i)
    decision_tree = DecisionTree(train_data_2, MAX_DEPTH, default_prediction, method='entropy', FEM=True)
    f1 = f1_score(test_data[1], [decision_tree.predict_decision_tree(row) for i, row in enumerate(test_data[0]["attrs_vals"])], average='weighted')
    labels_for_percent_of_train_data.append('%.0f%%' % (i))
    list_of_f1.append(f1)
    decision_tree = DecisionTree(train_data_2, MAX_DEPTH, default_prediction, method='entropy', FEM=False)
    f1 = f1_score(test_data[1], [decision_tree.predict_decision_tree(row) for i, row in enumerate(test_data[0]["attrs_vals"])], average='weighted')
    list_of_acc.append(f1)
visulate_f1_and_acc(list_of_acc, list_of_f1, labels_for_percent_of_train_data)
visualize_tree(tree=decision_tree.tree, attrs_names=ATTRS_NAMES,output_name="tree.png")
"""
######
# compare f1 and acc of all methods (grouped)
######

list_of_percent_train_data = [0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80]

def train_and_evaluate(i):
    results = []
    train_data, test_data, train_data_mode, test_data_mode, train_data_distrib, test_data_distrib, skip_train, skip_test = \
        split_random_to_train_and_test_data_diff_methods(row_attrs, class_vals, PERCENT_OF_TRAIN_DATA, i)

    decision_tree = DecisionTree(train_data, MAX_DEPTH, default_prediction, method='entropy', FEM=False)
    acc = accuracy_score(test_data[1], [decision_tree.predict_decision_tree(row) for row in test_data[0]["attrs_vals"]])
    f1 = f1_score(test_data[1], [decision_tree.predict_decision_tree(row) for row in test_data[0]["attrs_vals"]], average='weighted')
    results.append((acc, f1))

    decision_tree = DecisionTree(train_data_mode, MAX_DEPTH, default_prediction, method='entropy', FEM=False)
    acc = accuracy_score(test_data_mode[1], [decision_tree.predict_decision_tree(row) for row in test_data[0]["attrs_vals"]])
    f1 = f1_score(test_data_mode[1], [decision_tree.predict_decision_tree(row) for row in test_data_mode[0]["attrs_vals"]], average='weighted')
    results.append((acc, f1))

    decision_tree = DecisionTree(train_data, MAX_DEPTH, default_prediction, method='entropy', FEM=True)
    acc = accuracy_score(test_data[1], [decision_tree.predict_decision_tree(row) for row in test_data[0]["attrs_vals"]])
    f1 = f1_score(test_data[1], [decision_tree.predict_decision_tree(row) for row in test_data[0]["attrs_vals"]], average='weighted')
    results.append((acc, f1))

    decision_tree = DecisionTree(skip_train, MAX_DEPTH, default_prediction, method='entropy', FEM=False)
    acc = accuracy_score(skip_test[1], [decision_tree.predict_decision_tree(row) for row in skip_test[0]["attrs_vals"]])
    f1 = f1_score(skip_test[1], [decision_tree.predict_decision_tree(row) for row in skip_test[0]["attrs_vals"]], average='weighted')
    results.append((acc, f1))

    return results

def parallel_execution(iteration):
    all_acc = []
    all_f1 = []

    with Pool() as pool:
        results = pool.map(train_and_evaluate, list_of_percent_train_data)

    for result in results:
        for acc, f1 in result:
            all_acc.append(acc)
            all_f1.append(f1)
    
    return all_acc, all_f1

unique_values, counts = np.unique(class_vals, return_counts=True)
default_prediction = unique_values[np.argmax(counts)]

all_acc = []
all_f1 = []

for iteration in range(25):
    print(f"Iteration {iteration}")
    acc, f1 = parallel_execution(iteration)
    all_acc.append(acc)
    all_f1.append(f1)

    if iteration == 0:
        labels_for_percent_of_train_data = ['%.0f%%' % (i) for i in list_of_percent_train_data]

average_acc = np.mean(all_acc, axis=0)
average_f1 = np.mean(all_f1, axis=0)

print("acc:")
for i in range(len(list_of_percent_train_data)):
    print(f"{labels_for_percent_of_train_data[i]}: " + ", ".join(f"{average_acc[i*4+j]:.4f}%".replace('.', ',') for j in range(4)))

print("f1:")
for i in range(len(list_of_percent_train_data)):
    print(f"{labels_for_percent_of_train_data[i]}: " + ", ".join(f"{average_f1[i*4+j]:.4f}%".replace('.', ',') for j in range(4)))

visulate_acc_per_replacement_method(average_acc, labels_for_percent_of_train_data)
visulate_f1_per_replacement_method(average_f1, labels_for_percent_of_train_data)