from anytree import Node
from anytree.exporter import UniqueDotExporter, DotExporter
from random_forest import Leaf
import matplotlib.pyplot as plt
import numpy as np


def visualize_tree(tree, attrs_names, output_name):
    """
    Visualizes a decision tree.

    Args:
        tree (DecisionTree): The decision tree to visualize.
        attrs_names (list): The names of the attributes in the tree.
        output_name (str): The name of the output picture file.

    Returns:
        None
    """
    main_node = Node(attrs_names[tree.attr_index], node_type="node")

    def create_children(tree, parrent_node):
        """
        Creates child nodes for a given tree and parent node.

        Args:
            tree (Tree): The tree object.
            parrent_node (Node): The parent node object.

        Returns:
            None
        """
        if isinstance(tree, Leaf):
            return
        for branch in tree.branches:
            if isinstance(tree.branches[branch], Leaf):
                node = Node(str(tree.branches[branch].decision), parent=parrent_node, node_type="leaf", parrent_branch=branch)
            else:
                node = Node(attrs_names[tree.branches[branch].attr_index], parent=parrent_node, node_type="node", parrent_branch=branch)
                create_children(tree.branches[branch], node)
    create_children(tree, main_node)

    def edgeattrfunc(node, child):
        return 'label=%s' % (str(child.parrent_branch))

    def nodeattrfunc(node):
        if node.node_type == "leaf":
            return "fixedsize=true"
        else:
            return "shape=diamond"
    print(main_node.height)
    #dot_filename = output_name.replace('.png', '.dot')
    #UniqueDotExporter(main_node, edgeattrfunc=edgeattrfunc).to_dotfile(dot_filename)

    UniqueDotExporter(main_node, edgeattrfunc=edgeattrfunc).to_picture(output_name)


def visualize_metrics_of_confusion_matrix(metrics, x_labels):
    """
    Visualizes the metrics of a confusion matrix.

    Args:
        metrics (list): A list of dictionaries containing the metrics for each class.
            Each dictionary should have the following keys: "Ttr", "Ffr", "Ppv", "Acc", "F1".
        x_labels (list): A list of labels for the x-axis.

    Returns:
        None
    """
    ttr = []
    ffr = []
    ppv = []
    acc = []
    f1 = []
    for i in metrics:
        ttr.append(i["Ttr"] or 0)
        ffr.append(i["Ffr"] or 0)
        ppv.append(i["Ppv"] or 0)
        acc.append(i["Acc"] or 0)
        f1.append(i["F1"] or 0)
    x = np.arange(len(x_labels))
    width = 0.10
    fig, ax = plt.subplots()
    ax.bar(x - 2*width, ttr, width, label='Ttr')
    ax.bar(x - width, ffr, width, label="Ffr")
    ax.bar(x, ppv, width, label="Ppv")
    ax.bar(x + width, acc, width, label="Acc")
    ax.bar(x + width*2, f1, width, label="F1")
    ax.set_xticks(x, x_labels)
    ax.legend()
    fig.tight_layout()
    fig.savefig("confusion_matrixs.png")


def visualize_class_counter(class_counter, nr):
    """
    Visualizes the class counter data.

    Args:
        class_counter (dict): A dictionary containing the class counter data.

    Returns:
        None
    """
    fig, axs = plt.subplots(2, 3)
    for attr, ax in zip(class_counter, axs.flat):
        x_label = []
        bars = {}
        for attr_val in class_counter[attr]:
            x_label.append(attr_val)
            for class_val in class_counter[attr][attr_val]:
                if class_val not in bars:
                    bars[class_val] = []
                bars[class_val].append(class_counter[attr][attr_val][class_val])
        x = np.arange(len(x_label))
        width = 0.10
        width_padding = [i for i in range(-2, 3)]
        for i, class_val in enumerate(bars):
            rects = ax.bar(x + width_padding[i]*width, bars[class_val], width, label=class_val)
            if class_val == "recommend":
                ax.bar_label(rects, padding=1)
        ax.set_xticks(x)
        ax.set_xticklabels(x_label, rotation=45)
        ax.set_title(attr)
    fig.tight_layout()
    fig.savefig(f"class_counter{nr}.png")


def visualize_acc(list_of_acc, x):
    """
    Visualizes the accuracy values using a bar graph.

    Parameters:
    list_of_acc (list): A list of accuracy values.
    x (list): A list of x-axis values.

    Returns:
    None
    """
    plt.bar(x, list_of_acc)
    plt.ylabel("Acc")
    plt.xticks(rotation=45)
    plt.savefig("acc_graph.png")


def visulate_acc_per_input_method(list_of_acc, labels):
    """
    Visualizes the accuracy values using a bar graph for two input methods.

    Parameters:
    list_of_acc (list): A list of accuracy values.
    x (list): A list of x-axis values.

    Returns:
    None
    """
    method1_acc = list_of_acc[0::2]
    method2_acc = list_of_acc[1::2]
    bar_width = 0.35
    index = np.arange(len(labels))
    fig, ax = plt.subplots()
    _ = ax.bar(index, method1_acc, bar_width, label='entropy')
    _ = ax.bar(index + bar_width, method2_acc, bar_width, label='gini')
    ax.set_xlabel('Percentage of Training Data')
    ax.set_ylabel('Accuracy')
    ax.set_title('Comparison of Two Input Methods')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.savefig("entropia_test.png")
