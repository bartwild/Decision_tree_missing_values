from dataset_utils import get_data, split_random_to_train_and_test_data
from decision_tree import DecisionTree
from visualization_utils import visualize_acc, visualize_metrics_of_confusion_matrix, visualize_class_counter, visualize_tree, visulate_acc_per_input_method
from utils import ATTRS_NAMES, CLASS_VALUES, MAX_DEPTH, PERCENT_OF_TRAIN_DATA, ATTR_TO_INDEX
import numpy as np
import random

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
#######
 # compare acc by percent of train data
#######
"""
list_of_percent_train_data = [i*0.01 for i in range(5, 20, 5)] + [i*0.1 for i in range(2, 10, 4)] + [i for i in range(1, 5, 2)] + [i for i in range(5, 35, 5)] + [i for i in range(35, 85, 10)]
labels_for_percent_of_train_data = []
list_of_acc = []
for i in list_of_percent_train_data:
    print(i)
    train_data, test_data = split_to_sorted_train_and_test_data(row_attrs, class_vals, i)
    random_forest = RandomForest(train_data, N_TREES, MAX_DEPTH, percent_of_drawn_attrs=PERCENT_OF_DRAWN_ROWS, n_attrs=N_ATTRS)
    acc = random_forest.calculate_acc(test_data)
    list_of_acc.append(acc)
    labels_for_percent_of_train_data.append('%.2f%%' % (i))
    print(acc)

visualize_acc(list_of_acc, labels_for_percent_of_train_data)
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
list_of_percent_train_data = [0.1, 0.5, 1, 5, 10]
labels_for_percent_of_train_data = []
list_of_acc = []
for i in list_of_percent_train_data:
    print(i)
    train_data, test_data = split_random_to_train_and_test_data(row_attrs, class_vals, i)
    random_forest = RandomForest(train_data, N_TREES, MAX_DEPTH, percent_of_drawn_attrs=PERCENT_OF_DRAWN_ROWS, n_attrs=N_ATTRS, method='entropy')
    acc = random_forest.calculate_acc(test_data)
    list_of_acc.append(acc)
    print(acc)
    random_forest = RandomForest(train_data, N_TREES, MAX_DEPTH, percent_of_drawn_attrs=PERCENT_OF_DRAWN_ROWS, n_attrs=N_ATTRS, method='gini')
    acc = random_forest.calculate_acc(test_data)
    list_of_acc.append(acc)
    labels_for_percent_of_train_data.append('%.2f%%' % (i))
    print(acc)

visulate_acc_per_input_method(list_of_acc, labels_for_percent_of_train_data)
"""
list_of_percent_train_data = [0.1, 0.5, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80]
labels_for_percent_of_train_data = []
list_of_acc = []
for i in list_of_percent_train_data:
    train_data, test_data = split_random_to_train_and_test_data(row_attrs, class_vals, i)

    decision_tree = DecisionTree(train_data[0:2], MAX_DEPTH, train_data[2], method='entropy')
    acc = decision_tree.calculate_acc(test_data[0:2])
    list_of_acc.append(acc)
    print(f"Accuracy (Entropy): {acc}")

    decision_tree = DecisionTree(train_data[0:2], MAX_DEPTH, train_data[2], method='gini')
    acc = decision_tree.calculate_acc(test_data[0:2])
    list_of_acc.append(acc)
    print(f"Accuracy (Gini): {acc}")
    labels_for_percent_of_train_data.append(f'{i:.2f}%')
visualize_tree(tree=decision_tree.tree, attrs_names=ATTRS_NAMES,output_name="tree.png")
visulate_acc_per_input_method(list_of_acc, labels_for_percent_of_train_data)
