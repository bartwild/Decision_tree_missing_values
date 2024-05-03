import numpy as np
import math
import random
import copy

from sympy import total_degree


class RandomForest():
    """
    Random Forest classifier.

    Parameters:
    - train_data (tuple): Tuple containing the training data, where the first element is a dictionary with attribute indices and values, and the second element is a list of class labels.
    - number_of_trees (int): Number of decision trees to include in the random forest.
    - max_depth (int): Maximum depth of each decision tree.
    - percent_of_drawn_attrs (float, optional): Percentage of attributes to be randomly drawn for each tree. Defaults to 60.
    - n_attrs (int, optional): Number of attributes to be randomly drawn for each tree. Overrides percent_of_drawn_attrs if provided.

    Methods:
    - calculate_acc(test_data): Calculates the accuracy of the random forest on the given test data.
    - calculate_confusion_matrix(test_data, checked_class): Calculates the confusion matrix for a specific class on the given test data.
    - calculate_metrics_of_confusion_matrix(confusion_matrix): Calculates various metrics (True Positive Rate, False Positive Rate, Precision, Accuracy, F1 Score) based on the given confusion matrix.
    - predict_random_forest(input_data): Predicts the class label for the given input data using the random forest.
    """
    trees = []
    train_data = None

    def __init__(self, train_data, number_of_trees, max_depth, percent_of_drawn_attrs=60, n_attrs=None, method="entropy"):
        self.train_data = copy.deepcopy(train_data)
        for _ in range(number_of_trees):
            new_train_data = self.get_drawn_attrs_and_rows(percent_of_drawn_attrs, n_attrs=n_attrs)
            tree = self.genenerate_tree(new_train_data, max_depth, method)
            self.trees.append(tree)

    def get_drawn_attrs_and_rows(self, percent_of_data, n_attrs=None):
        """
        Randomly selects a subset of attributes and rows from the training data.

        Args:
            percent_of_data (float): The percentage of data to be included in the subset.
            n_attrs (int, optional): The number of attributes to be included in the subset. If not specified,
                                     the square root of the total number of attributes will be used.

        Returns:
            tuple: A tuple containing a dictionary with the selected attribute indices and values,
                   and a list with the corresponding class labels.
        """
        number_of_attrs = int(math.sqrt(len(self.train_data[0]["attrs_index"])))
        if n_attrs:
            number_of_attrs = n_attrs
        size_of_new_train_data = int(len(self.train_data[0]["attrs_vals"]) * (percent_of_data/100))
        new_train_data = ({"attrs_index": [], "attrs_vals": []}, [])
        while len(new_train_data[0]["attrs_index"]) < number_of_attrs:
            index = random.randrange(len(self.train_data[0]["attrs_index"]))
            if index not in new_train_data[0]["attrs_index"]:
                new_train_data[0]["attrs_index"].append(index)
        while len(new_train_data[0]["attrs_vals"]) < size_of_new_train_data:
            index = random.randrange(len(self.train_data[0]["attrs_vals"]))
            new_train_data[0]["attrs_vals"].append(self.train_data[0]["attrs_vals"][index])
            new_train_data[1].append(self.train_data[1][index])
        return new_train_data

    def calculate_entropy(self, class_vals, uniq_class_vals):
        """
        Calculate the entropy of a given set of class values.

        Parameters:
            class_vals (list): A list of class values.
            uniq_class_vals (list): A list of uniq class values.

        Returns:
            float: The entropy value.
        """
        entropy = 0
        for class_val in uniq_class_vals:
            number_of_filtered_rows_of_specific_val = len(list(filter(lambda x: x == class_val, class_vals)))
            if number_of_filtered_rows_of_specific_val > 0:
                prob_of_class = number_of_filtered_rows_of_specific_val/len(class_vals)
                entropy -= prob_of_class*math.log(prob_of_class)
        return entropy

    def calculate_gini_impurity(self, class_vals, uniq_class_vals):
        """
        Calculate the gini impurity of a given set of class values.

        Parameters:
            class_vals (list): A list of class values.
            uniq_class_vals (list): A list of uniq class values.

        Returns:
            float: The gini impurity value.
        """
        impurity = 1
        for class_val in uniq_class_vals:
            number_of_filtered_rows_of_specific_val = len(list(filter(lambda x: x == class_val, class_vals)))
            if number_of_filtered_rows_of_specific_val > 0:
                prob_of_class = number_of_filtered_rows_of_specific_val/len(class_vals)
                impurity -= prob_of_class*prob_of_class
        return impurity

    def calculate_acc(self, test_data):
        """
        Calculate the accuracy of the random forest model on the given test data.

        Parameters:
        - test_data (tuple): A tuple containing the test data, where the first element is a dictionary with the attribute values
                             and the second element is a list of corresponding labels.

        Returns:
        - float: The accuracy of the random forest model on the test data, expressed as a value between 0 and 1.
        """
        correct_predictions = 0
        wrong_predictions = 0
        for i, row in enumerate(test_data[0]["attrs_vals"]):
            decision = self.predict_random_forest(row)
            if decision == test_data[1][i]:
                correct_predictions += 1
            else:
                wrong_predictions += 1
        return correct_predictions / (correct_predictions + wrong_predictions)

    def calculate_confusion_matrix(self, test_data, checked_class):
        """
        Calculates the confusion matrix for a given test data and checked class.

        Args:
            test_data (list): The test data containing attribute values and corresponding class labels.
            checked_class (str): The class label to be checked in the confusion matrix.

        Returns:
            dict: The confusion matrix containing the counts of true positives (tp), false positives (fp),
                  false negatives (fn), and true negatives (tn).
        """
        confusion_matrix = {
            "tp": 0,  # detected and it's true
            "fp": 0,  # detected but not true
            "fn": 0,  # not detected but it's this class
            "tn": 0  # not detected and it's not this class
        }
        for i, row in enumerate(test_data[0]["attrs_vals"]):
            decision = self.predict_random_forest(row)
            if decision == checked_class:
                if checked_class == test_data[1][i]:
                    confusion_matrix["tp"] += 1
                else:
                    confusion_matrix["fp"] += 1
            else:
                if checked_class != test_data[1][i]:
                    confusion_matrix["tn"] += 1
                else:
                    confusion_matrix["fn"] += 1
        return confusion_matrix

    def calculate_metrics_of_confusion_matrix(self, confusion_matrix):
        """
        Calculate various metrics based on a given confusion matrix.

        Args:
            confusion_matrix (dict): A dictionary containing the counts of true positives (tp), false negatives (fn),
                                     false positives (fp), and true negatives (tn).

        Returns:
            dict: A dictionary containing the calculated metrics including True Positive Rate (Ttr), False Positive Rate (Ffr),
                  Positive Predictive Value (Ppv), Accuracy (Acc), and F1 score (F1).
        """
        ttr = None
        ffr = None
        ppv = None
        acc = None
        f1 = None
        if confusion_matrix["tp"] + confusion_matrix["fn"] != 0:
            ttr = confusion_matrix["tp"] / (confusion_matrix["tp"] + confusion_matrix["fn"])
        if confusion_matrix["fp"] + confusion_matrix["tn"] != 0:
            ffr = confusion_matrix["fp"] / (confusion_matrix["fp"] + confusion_matrix["tn"])
        if confusion_matrix["tp"] + confusion_matrix["fp"] != 0:
            ppv = confusion_matrix["tp"] / (confusion_matrix["tp"] + confusion_matrix["fp"])
        if confusion_matrix["tp"] + confusion_matrix["tn"] + confusion_matrix["fp"] + confusion_matrix["fn"]:
            acc = (confusion_matrix["tp"] + confusion_matrix["tn"]) / (
                    confusion_matrix["tp"] + confusion_matrix["tn"] + confusion_matrix["fp"] + confusion_matrix["fn"])
        if ttr and ppv and (ttr + ppv) > 0:
            f1 = (2 * ppv * ttr) / (ttr + ppv)
        metrics = {
            "Ttr": ttr,
            "Ffr": ffr,
            "Ppv": ppv,
            "Acc": acc,
            "F1": f1
        }
        return metrics

    def inf_gain(self, attr_index, new_train_data, method):
        """
        Calculates the information gain for a given attribute index and new training data.

        Parameters:
            attr_index (int): The index of the attribute for which the information gain is calculated.
            new_train_data (list): The new training data containing attribute values and class labels.

        Returns:
            float: The information gain value.
        """
        number_of_rows = len(new_train_data[0]["attrs_vals"])
        attr_vals = [i[attr_index] for i in new_train_data[0]["attrs_vals"]]
        uniq_attr_vals = np.unique(attr_vals)
        uniq_class_vals = np.unique(new_train_data[1])
        total_entropy = 0
        if method == "gini":
            total_entropy = self.calculate_gini_impurity(new_train_data[1], uniq_class_vals)
        else:
            total_entropy = self.calculate_entropy(new_train_data[1], uniq_class_vals)
        info = 0
        for attr_val in uniq_attr_vals:
            filtered_class_vals = []
            for index, val in enumerate(attr_vals):
                if val == attr_val:
                    filtered_class_vals.append(new_train_data[1][index])
            attr_vals_prob = len(filtered_class_vals)/number_of_rows
            if method == "gini":
                entropy = self.calculate_gini_impurity(filtered_class_vals, uniq_class_vals)
            else:
                entropy = self.calculate_entropy(filtered_class_vals, uniq_class_vals)
            info += attr_vals_prob*entropy
        return total_entropy - info

    def genenerate_tree(self, new_train_data, max_depth, method):
        """
        Generates a decision tree based on the given training data.

        Args:
            new_train_data (tuple): A tuple containing the training data.
                The first element is a dictionary with keys "attrs_index" and "attrs_vals",
                representing the attribute indices and attribute values for each training instance.
                The second element is a list of class labels for each training instance.
            max_depth (int): The maximum depth of the decision tree.

        Returns:
            Node: The root node of the generated decision tree.
        """
        max_inf_gain = -1
        max_inf_gain_attr_index = None
        uniq_class_vals = np.unique(new_train_data[1])
        for attr_index in new_train_data[0]["attrs_index"]:
            info_gain = self.inf_gain(attr_index, new_train_data, method)
            if max_inf_gain < info_gain:
                max_inf_gain = info_gain
                max_inf_gain_attr_index = attr_index
        uniq_attr_vals = {}
        tree = Node(max_inf_gain_attr_index)
        for i, row in enumerate(new_train_data[0]["attrs_vals"]):
            if row[max_inf_gain_attr_index] not in uniq_attr_vals:
                uniq_attr_vals[row[max_inf_gain_attr_index]] = {}
                uniq_attr_vals[row[max_inf_gain_attr_index]]["attrs_vals"] = [row]
                uniq_attr_vals[row[max_inf_gain_attr_index]]["class_vals"] = [new_train_data[1][i]]
                uniq_attr_vals[row[max_inf_gain_attr_index]]["class_vals_count"] = {}
                for k in uniq_class_vals:
                    uniq_attr_vals[row[max_inf_gain_attr_index]]["class_vals_count"][k] = 0
                uniq_attr_vals[row[max_inf_gain_attr_index]]["count"] = 1
                uniq_attr_vals[row[max_inf_gain_attr_index]]["class_vals_count"][new_train_data[1][i]] += 1
                uniq_attr_vals[row[max_inf_gain_attr_index]]["pure_class"] = False
            else:
                uniq_attr_vals[row[max_inf_gain_attr_index]]["count"] += 1
                uniq_attr_vals[row[max_inf_gain_attr_index]]["class_vals_count"][new_train_data[1][i]] += 1
                uniq_attr_vals[row[max_inf_gain_attr_index]]["attrs_vals"].append(row)
                uniq_attr_vals[row[max_inf_gain_attr_index]]["class_vals"].append(new_train_data[1][i])
        if max_depth == 1 or len(new_train_data[0]["attrs_index"]) == 1:
            for attr_val in uniq_attr_vals:
                the_best_class_val = None
                class_val_count = -1
                for class_val in uniq_attr_vals[attr_val]["class_vals_count"]:
                    if uniq_attr_vals[attr_val]["class_vals_count"][class_val] >= class_val_count:
                        the_best_class_val = class_val
                tree.add_branch(attr_val, Leaf(the_best_class_val))
        else:
            for attr_val in uniq_attr_vals.keys():
                new_data = [{"attrs_index": list(filter(lambda x: x != max_inf_gain_attr_index, new_train_data[0]["attrs_index"])), "attrs_vals": []}, []]
                for class_val in uniq_attr_vals[attr_val]["class_vals_count"]:
                    if uniq_attr_vals[attr_val]["class_vals_count"][class_val] == uniq_attr_vals[attr_val]["count"]:
                        tree.add_branch(attr_val, Leaf(class_val))  # pure class
                        uniq_attr_vals[attr_val]["pure_class"] = True
                        break
                if uniq_attr_vals[attr_val]["pure_class"] is False:
                    new_data[0]["attrs_vals"] = uniq_attr_vals[attr_val]["attrs_vals"]
                    new_data[1] = uniq_attr_vals[attr_val]["class_vals"]
                    node = self.genenerate_tree(tuple(new_data), max_depth - 1, method)
                    tree.add_branch(attr_val, node)
        return tree

    def predict_tree_decision(self, tree, input_data):
        """
        Predicts the decision for a given input data using a decision tree.

        Parameters:
        tree (Node): The decision tree to make predictions with.
        input_data (list): The input data for which the decision is to be predicted.

        Returns:
        str or None: The predicted decision for the input data, or None if the decision cannot be determined.
        """
        node = tree
        while isinstance(node, Node):
            attr_index = node.attr_index
            input_attr_val = input_data[attr_index]
            if input_attr_val not in node.branches:
                return None
            node = node.branches[input_attr_val]
        if isinstance(node, Leaf):
            return node.decision
        return None

    def predict_random_forest(self, input_data):
        """
        Predicts the output for the given input data using the random forest model.

        Parameters:
        - input_data: The input data for which the output is to be predicted.

        Returns:
        - The predicted output based on the random forest model.
        """
        decisions = {}
        for tree in self.trees:
            decision = self.predict_tree_decision(tree, input_data)
            if decisions is None:
                continue
            if decision not in decisions:
                decisions[decision] = 1
            else:
                decisions[decision] += 1
        the_best_decision = None
        number_of_occurance = 0
        for i in decisions:
            if decisions[i] >= number_of_occurance:
                the_best_decision = i
                number_of_occurance = decisions[i]
        return the_best_decision


class Node:
    """
    Represents a node in a decision tree.

    Attributes:
        attr_index (int): The index of the attribute associated with this node.
        branches (dict): A dictionary mapping attribute values to child nodes.
    """
    def __init__(self, attr_index):
        self.attr_index = attr_index
        self.branches = {}

    def add_branch(self, attr_val, children):
        self.branches[attr_val] = children


class Leaf:
    def __init__(self, decision):
        """
        Initialize a Leaf object.
        Args:
            decision (str): The decision made by the leaf.
        Returns:
            None
        """
        self.decision = decision
