import numpy as np
import math
import copy


class DecisionTree():
    tree = None
    train_data = None

    def __init__(self, train_data, max_depth, default_prediction, method="entropy", FEM=False):
        self.FEM = FEM
        self.train_data = copy.deepcopy(train_data)
        self.default_prediction = default_prediction
        tree = self.generate_tree(train_data, max_depth, method)
        self.tree = tree
        self.attr_value_freq = self.calculate_attr_value_freq(train_data)

    def calculate_entropy(self, class_vals, uniq_class_vals, weights):
        total_weight = sum(weights)
        entropy = 0
        for class_val in uniq_class_vals:
            weighted_class_sum = sum(weight for val, weight in zip(class_vals, weights) if val == class_val)
            if weighted_class_sum > 0:
                prob_of_class = weighted_class_sum / total_weight
                entropy -= prob_of_class * math.log(prob_of_class)
        return entropy

    def calculate_attr_value_freq(self, train_data):
        attr_value_freq = {i: {} for i in train_data[0]["attrs_index"]}
        for row in train_data[0]["attrs_vals"]:
            for attr_index, attr_val in enumerate(row):
                if attr_val not in attr_value_freq[attr_index]:
                    attr_value_freq[attr_index][attr_val] = 0
                attr_value_freq[attr_index][attr_val] += 1
        return attr_value_freq

    def calculate_gini_impurity(self, class_vals, uniq_class_vals, weights):
        total_weight = sum(weights)
        impurity = 1
        for class_val in uniq_class_vals:
            weighted_class_sum = sum(weight for val, weight in zip(class_vals, weights) if val == class_val)
            if weighted_class_sum > 0:
                prob_of_class = weighted_class_sum / total_weight
                impurity -= (prob_of_class ** 2)
        return impurity

    def calculate_majority_class(self, class_vals):
        try:
            maj_class = max(set(class_vals), key=class_vals.count)
        except Exception:
            maj_class = self.default_prediction
        return maj_class

    def calculate_confusion_matrix(self, test_data, checked_class):
        confusion_matrix = {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0
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

    def generate_tree(self, new_train_data, max_depth, method):
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
                        node = self.generate_tree(tuple(new_data), remaining_depth - 1, method)
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
        return self.predict_tree_decision(self.tree, input_data)


class Node:
    def __init__(self, attr_index, default_prediction):
        self.attr_index = attr_index
        self.branches = {}
        self.default_prediction = default_prediction

    def add_branch(self, attr_val, children):
        self.branches[attr_val] = children


class Leaf:
    def __init__(self, decision):
        self.decision = decision
