from anytree import Node
from anytree.exporter import UniqueDotExporter, DotExporter
from decision_tree import Leaf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils import CLASS_VALUES


def visualize_tree(tree, attrs_names, output_name):
    print(attrs_names[tree.attr_index])
    main_node = Node(attrs_names[tree.attr_index], node_type="node")

    def create_children(tree, parrent_node):
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

    UniqueDotExporter(main_node, edgeattrfunc=edgeattrfunc).to_picture(output_name)


def visualize_metrics_of_confusion_matrix(metrics, x_labels):
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
    plt.bar(x, list_of_acc)
    plt.ylabel("Acc")
    plt.xticks(rotation=45)
    plt.savefig("acc_graph.png")


def visualize_acc_per_input_method(list_of_acc, labels):
    method1_acc = list_of_acc[0::2]
    method2_acc = list_of_acc[1::2]
    bar_width = 0.35
    index = np.arange(len(labels))
    fig, ax = plt.subplots()
    _ = ax.bar(index, method1_acc, bar_width, label='entropia')
    _ = ax.bar(index + bar_width, method2_acc, bar_width, label='gini')
    ax.set_xlabel('Procent danych wziętych do treningu')
    ax.set_ylabel('Dokładność')
    ax.set_title('Porównanie dwóch metod obliczania przyrostu informacji')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.savefig("entropia_test.png")

def visualize_f1_per_input_method(list_of_f1, labels):
    method1_f1 = list_of_f1[0::2]
    method2_f1 = list_of_f1[1::2]
    bar_width = 0.35
    index = np.arange(len(labels))
    fig, ax = plt.subplots()
    _ = ax.bar(index, method1_f1, bar_width, label='entropia')
    _ = ax.bar(index + bar_width, method2_f1, bar_width, label='gini')
    ax.set_xlabel('Procent danych wziętych do treningu')
    ax.set_ylabel('Miara F1')
    ax.set_title('Porównanie dwóch metod obliczania przyrostu informacji')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.savefig("entropia_test1.png")


def visualize_acc_per_replacement_method(list_of_acc, labels):
    method1_acc = list_of_acc[0::4]
    method2_acc = list_of_acc[1::4]
    method3_acc = list_of_acc[2::4]
    method4_acc = list_of_acc[3::4]
    
    bar_width = 0.2
    index = np.arange(len(labels))
    
    fig, ax = plt.subplots()
    _ = ax.bar(index, method1_acc, bar_width, label='nowa wartość')
    _ = ax.bar(index + bar_width, method2_acc, bar_width, label='dominanta')
    _ = ax.bar(index + 2 * bar_width, method3_acc, bar_width, label='przykładów ułamkowych')
    _ = ax.bar(index + 3 * bar_width, method4_acc, bar_width, label='pominięcie')
    
    ax.set_xlabel('Procent danych brakujących')
    ax.set_ylabel('Dokładność')
    ax.set_title('Porównanie czterech metod')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(labels)
    ax.legend(loc='lower right')
    
    plt.savefig("metody_test.png")
    plt.show()


def visualize_f1_per_replacement_method(list_of_f1, labels):
    method1_f1 = list_of_f1[0::4]
    method2_f1 = list_of_f1[1::4]
    method3_f1 = list_of_f1[2::4]
    method4_f1 = list_of_f1[3::4]

    
    bar_width = 0.2
    index = np.arange(len(labels))
    
    fig, ax = plt.subplots()
    _ = ax.bar(index, method1_f1, bar_width, label='nowa wartość')
    _ = ax.bar(index + bar_width, method2_f1, bar_width, label='dominanta')
    _ = ax.bar(index + 2 * bar_width, method3_f1, bar_width, label='przykładów ułamkowych')
    _ = ax.bar(index + 3 * bar_width, method4_f1, bar_width, label='pominięcie')

    ax.set_xlabel('Procent danych brakujących')
    ax.set_ylabel('Miara F1')
    ax.set_title('Porównanie czterech metod')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper right')
    
    plt.savefig("metody_test_f1.png")
    plt.show()


def visualize_f1_and_acc(list_of_acc, list_of_f1, labels):
    method1_f1 = list_of_f1[0::1]
    method1_acc = list_of_acc[0::1]
    bar_width = 0.35
    index = np.arange(len(labels))
    fig, ax = plt.subplots()
    _ = ax.bar(index, method1_acc, bar_width, label='Model uznający brakujące dane za nowe')
    _ = ax.bar(index+bar_width, method1_f1, bar_width, label='Model używających przykładów ułamkowych')
    ax.set_xlabel('Procent danych brakujących')
    ax.set_ylabel('Miara F1')
    ax.set_title('Uznawanie brakujących dane za nowe i metoda przykładów ułamkowych')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.savefig("new_data1.png")

def visualize_confusion_matrix(cm, title):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                cbar=False, xticklabels=CLASS_VALUES, yticklabels=CLASS_VALUES)
    plt.title(title)
    plt.savefig(f"cm_{title}.png")
    plt.show()
