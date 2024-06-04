import numpy as np
import math
import copy


class DecisionTree():
    """
    Decision Tree classifier.

    Parameters:
    - train_data (tuple): Tuple containing the training data, where the first element is a dictionary with attribute indices and values, and the second element is a list of class labels.
    - max_depth (int): Maximum depth of each decision tree.

    Methods:
    - calculate_acc(test_data): Calculates the accuracy of the Decision Tree on the given test data.
    - calculate_confusion_matrix(test_data, checked_class): Calculates the confusion matrix for a specific class on the given test data.
    - calculate_metrics_of_confusion_matrix(confusion_matrix): Calculates various metrics (True Positive Rate, False Positive Rate, Precision, Accuracy, F1 Score) based on the given confusion matrix.
    - predict_random_forest(input_data): Predicts the class label for the given input data using the Decision Tree.
    """
    tree = None
    train_data = None

    def __init__(self, train_data, max_depth, default_prediction, method="entropy", FEM=True):
        self.FEM = FEM
        self.train_data = copy.deepcopy(train_data)
        self.default_prediction = default_prediction
        tree = self.genenerate_tree(train_data, max_depth, method)
        self.tree = tree
        self.attr_value_freq = self.calculate_attr_value_freq(train_data)

    def calculate_entropy(self, class_vals, uniq_class_vals, weights):
        """
        Calculate the entropy of a given set of class values.

        Parameters:
            class_vals (list): A list of class values.
            uniq_class_vals (list): A list of uniq class values.

        Returns:
            float: The entropy value.
        """
        total_weight = sum(weights)
        entropy = 0
        for class_val in uniq_class_vals:
            weighted_class_sum = sum(weight for val, weight in zip(class_vals, weights) if val == class_val)
            if weighted_class_sum > 0:
                prob_of_class = weighted_class_sum / total_weight
                entropy -= prob_of_class * math.log(prob_of_class)
        return entropy

    def calculate_attr_value_freq(self, train_data):
        """
        Calculate the frequency of each attribute value in the training data.

        Args:
            train_data (tuple): Training data consisting of attribute values and class labels.

        Returns:
            dict: A dictionary where keys are attribute indices and values are dictionaries
                  mapping attribute values to their frequencies.
        """
        attr_value_freq = {i: {} for i in train_data[0]["attrs_index"]}
        for row in train_data[0]["attrs_vals"]:
            for attr_index, attr_val in enumerate(row):
                if attr_val not in attr_value_freq[attr_index]:
                    attr_value_freq[attr_index][attr_val] = 0
                attr_value_freq[attr_index][attr_val] += 1
        return attr_value_freq

    def calculate_gini_impurity(self, class_vals, uniq_class_vals, weights):
        """
        Calculate the weighted gini impurity of a given set of class values.

        Parameters:
            class_vals (list): A list of class values.
            uniq_class_vals (list): A list of unique class values.
            weights (list): A list of weights corresponding to each example.

        Returns:
            float: The weighted gini impurity value.
        """
        total_weight = sum(weights)
        impurity = 1
        for class_val in uniq_class_vals:
            weighted_class_sum = sum(weight for val, weight in zip(class_vals, weights) if val == class_val)
            if weighted_class_sum > 0:
                prob_of_class = weighted_class_sum / total_weight
                impurity -= (prob_of_class ** 2)
        return impurity

    def calculate_acc(self, test_data):
        """
        Calculate the accuracy of the Decision Tree model on the given test data.

        Parameters:
        - test_data (tuple): A tuple containing the test data, where the first element is a dictionary with the attribute values
                             and the second element is a list of corresponding labels.

        Returns:
        - float: The accuracy of the Decision Tree model on the test data, expressed as a value between 0 and 1.
        """
        correct_predictions = 0
        wrong_predictions = 0
        for i, row in enumerate(test_data[0]["attrs_vals"]):
            decision = self.predict_decision_tree(row)
            if decision == test_data[1][i]:
                correct_predictions += 1
            else:
                wrong_predictions += 1
        return correct_predictions / (correct_predictions + wrong_predictions)

    def calculate_majority_class(self, class_vals):
        """
        Calculate the majority class in the given class values.

        Parameters:
            class_vals (list): A list of class values.

        Returns:
            object: The majority class.
        """
        try:
            maj_class = max(set(class_vals), key=class_vals.count)
        except Exception:
            maj_class = self.default_prediction
        return maj_class

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
            decision = self.predict_decision_tree(row)
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

    def inf_gain(self, attr_index, new_train_data, method, filtered_weights):
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
            total_entropy = self.calculate_gini_impurity(new_train_data[1], uniq_class_vals, filtered_weights)
        else:
            total_entropy = self.calculate_entropy(new_train_data[1], uniq_class_vals, filtered_weights)
        info = 0
        for attr_val in uniq_attr_vals:
            filtered_class_vals = []
            filtered_weights_for_attr_val = []
            for index, val in enumerate(attr_vals):
                if val == attr_val:
                    filtered_class_vals.append(new_train_data[1][index])
                    filtered_weights_for_attr_val.append(filtered_weights[index])
            attr_vals_prob = len(filtered_class_vals)/number_of_rows
            if method == "gini":
                entropy = self.calculate_gini_impurity(filtered_class_vals, uniq_class_vals, filtered_weights_for_attr_val)
            else:
                entropy = self.calculate_entropy(filtered_class_vals, uniq_class_vals, filtered_weights_for_attr_val)
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
            method (str): Method to calculate impurity ("entropy" or "gini").
        Returns:
            Node: The root node of the generated decision tree.
        """
        def calculate_filtered_weights():
            return [len([x for x in attrs_vals if x != 'missing']) / len(attrs_vals) for attrs_vals in new_train_data[0]['attrs_vals']] if self.FEM else [1] * len(new_train_data[0]['attrs_vals'])

        def find_best_attribute():
            max_inf_gain = -1
            max_inf_gain_attr_index = None
            for attr_index in new_train_data[0]["attrs_index"]:
                info_gain = self.inf_gain(attr_index, new_train_data, method, filtered_weights)
                if info_gain > max_inf_gain:
                    max_inf_gain = info_gain
                    max_inf_gain_attr_index = attr_index
            return max_inf_gain_attr_index

        def split_data_on_attribute(attr_index):
            uniq_attr_vals = {}
            for i, row in enumerate(new_train_data[0]["attrs_vals"]):
                attr_val = row[attr_index]
                if attr_val not in uniq_attr_vals:
                    uniq_attr_vals[attr_val] = {"attrs_vals": [], "class_vals": [], "class_vals_count": {k: 0 for k in uniq_class_vals}, "count": 0, "pure_class": False}
                uniq_attr_vals[attr_val]["attrs_vals"].append(row)
                uniq_attr_vals[attr_val]["class_vals"].append(new_train_data[1][i])
                uniq_attr_vals[attr_val]["class_vals_count"][new_train_data[1][i]] += 1
                uniq_attr_vals[attr_val]["count"] += 1
            return uniq_attr_vals

        def add_branches_or_leaves(tree, uniq_attr_vals, remaining_depth):
            for attr_val, val_data in uniq_attr_vals.items():
                if remaining_depth == 1 or len(new_train_data[0]["attrs_index"]) == 1:
                    best_class_val = max(val_data["class_vals_count"], key=val_data["class_vals_count"].get)
                    tree.add_branch(attr_val, Leaf(best_class_val))
                else:
                    if any(val_data["class_vals_count"][class_val] == val_data["count"] for class_val in val_data["class_vals_count"]):
                        pure_class = next(class_val for class_val in val_data["class_vals_count"] if val_data["class_vals_count"][class_val] == val_data["count"])
                        tree.add_branch(attr_val, Leaf(pure_class))
                        val_data["pure_class"] = True
                    if not val_data["pure_class"]:
                        new_data = [{"attrs_index": [i for i in new_train_data[0]["attrs_index"] if i != max_inf_gain_attr_index], "attrs_vals": val_data["attrs_vals"]}, val_data["class_vals"]]
                        node = self.genenerate_tree(tuple(new_data), remaining_depth - 1, method)
                        tree.add_branch(attr_val, node)

        uniq_class_vals = np.unique(new_train_data[1])
        filtered_weights = calculate_filtered_weights()
        max_inf_gain_attr_index = find_best_attribute()
        majority_class = self.calculate_majority_class(new_train_data[1])
        tree = Node(max_inf_gain_attr_index, majority_class)
        uniq_attr_vals = split_data_on_attribute(max_inf_gain_attr_index)
        add_branches_or_leaves(tree, uniq_attr_vals, max_depth)

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
            if input_attr_val == 'missing' and self.FEM is True:
                weighted_votes = {}
                total_weight = sum(self.attr_value_freq[attr_index].values())
                for attr_val, freq in self.attr_value_freq[attr_index].items():
                    if attr_val in node.branches:
                        decision = self.predict_tree_decision(node.branches[attr_val], input_data)
                        if decision not in weighted_votes:
                            weighted_votes[decision] = 0
                        weighted_votes[decision] += freq / total_weight
                if weighted_votes:
                    return max(weighted_votes, key=weighted_votes.get)
                return node.default_prediction
            if input_attr_val not in node.branches:
                return node.default_prediction
            node = node.branches[input_attr_val]
        if isinstance(node, Leaf):
            return node.decision
        return node.default_prediction

    def predict_decision_tree(self, input_data):
        """
        Predicts the output for the given input data using the Decision Tree model.

        Parameters:
        - input_data: The input data for which the output is to be predicted.

        Returns:
        - The predicted output based on the Decision Tree model.
        """
        return self.predict_tree_decision(self.tree, input_data)


class Node:
    """
    Represents a node in a decision tree.

    Attributes:
        attr_index (int): The index of the attribute associated with this node.
        branches (dict): A dictionary mapping attribute values to child nodes.
    """
    def __init__(self, attr_index, default_prediction):
        self.attr_index = attr_index
        self.branches = {}
        self.default_prediction = default_prediction

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
