import random
from collections import Counter
import copy
import sys

def calculate_mode(attributes):
    columns = list(zip(*attributes))
    mode_values = []
    for column in columns:
        most_common = Counter(x for x in column if x != 'missing').most_common(1)
        mode_value = most_common[0][0] if most_common else 'missing'
        mode_values.append(mode_value)
    return mode_values

def replace_missing_with_mode(data, mode_values):
    new_data = []
    for row in data:
        new_row=[]*len(row)
        for i, value in enumerate(row):
            if value == 'missing':
                new_row.append(mode_values[i])
            else:
                new_row.append(value)
        new_data.append(new_row)
    return new_data

def mask_random_attributes(attrs_vals, masking_rate):
    masked_attrs_vals = []
    for values in attrs_vals:
        masked_values = ['missing' if random.random() < masking_rate else value for value in values]
        masked_attrs_vals.append(masked_values)
    return masked_attrs_vals

def skip_missing_inputs(data):
    new_data = []
    for row in data:
        if 'missing' in row[0]:
            continue
        new_data.append(row)
    return new_data

def split_random_to_train_and_test_data(attrs_vals, class_vals, percent_of_train_data, masking_rate=0.1):
    attrs_vals = mask_random_attributes(attrs_vals, masking_rate)

    combined_data = list(zip(attrs_vals, class_vals))
    random.shuffle(combined_data)
    split_index = int(len(combined_data) * (percent_of_train_data / 100))
    train_data = combined_data[:split_index]
    test_data = combined_data[split_index:]

    train_attrs_vals, train_class_vals = zip(*train_data) if train_data else ([], [])
    test_attrs_vals, test_class_vals = zip(*test_data) if test_data else ([], [])

    return ({"attrs_index": list(range(len(attrs_vals[0]))), "attrs_vals": train_attrs_vals}, train_class_vals), \
           ({"attrs_index": list(range(len(attrs_vals[0]))), "attrs_vals": test_attrs_vals}, test_class_vals)

def split_random_to_train_and_test_data_diff_methods(attrs_vals, class_vals, percent_of_train_data, masking_rate=10):
    masking_rate=masking_rate/100
    rand_seed = random.randint(-sys.maxsize-1, sys.maxsize)

    attrs_vals = mask_random_attributes(attrs_vals, masking_rate)
    attrs_vals_mode = copy.deepcopy(attrs_vals)
    attrs_vals_distrib = copy.deepcopy(attrs_vals)
    
    # skipped data 
    combined_data_skip = list(zip(attrs_vals, class_vals))
    random.seed(rand_seed)
    random.shuffle(combined_data_skip)
    split_index = int(len(combined_data_skip) * (percent_of_train_data / 100))
    train_data_skip = combined_data_skip[:split_index]
    skipped_train_data = skip_missing_inputs(train_data_skip)
    train_attrs_vals_skip, train_class_vals_skip = zip(*skipped_train_data) if skipped_train_data else ([], [])
    
    # masked data
    combined_data = list(zip(attrs_vals, class_vals))
    random.seed(rand_seed)
    random.shuffle(combined_data)
    split_index = int(len(combined_data) * (percent_of_train_data / 100))
    train_data = combined_data[:split_index]
    test_data = combined_data[split_index:]
    train_attrs_vals, train_class_vals = zip(*train_data) if train_data else ([], [])
    test_attrs_vals, test_class_vals = zip(*test_data) if test_data else ([], [])

    # masked data with mode replacement
    mode_values = calculate_mode(attrs_vals_mode)
    attrs_vals_mode = replace_missing_with_mode(attrs_vals_mode, mode_values)
    combined_data = list(zip(attrs_vals_mode, class_vals))
    random.seed(rand_seed)
    random.shuffle(combined_data)
    split_index = int(len(combined_data) * (percent_of_train_data / 100))
    train_data = combined_data[:split_index]
    test_data = combined_data[split_index:]
    train_attrs_vals_mode, train_class_vals_mode = zip(*train_data) if train_data else ([], [])
    test_attrs_vals_mode, test_class_vals_mode = zip(*test_data) if test_data else ([], [])

    # masked data with distribution replacement
    combined_data = list(zip(attrs_vals_distrib, class_vals))
    random.seed(rand_seed)
    random.shuffle(combined_data)
    distributions = calculate_attribute_distribution([x for x,_ in combined_data])
    attrs_vals_distrib = replace_missing_with_distribution([x for x,_ in combined_data], distributions)
    split_index = int(len(combined_data) * (percent_of_train_data / 100))
    train_data = combined_data[:split_index]
    test_data = combined_data[split_index:]
    train_attrs_vals_distrib, train_class_vals_distrib = zip(*train_data) if train_data else ([], [])
    test_attrs_vals_distrib, test_class_vals_distrib = zip(*test_data) if test_data else ([], [])

    return ({"attrs_index": list(range(len(attrs_vals[0]))), "attrs_vals": train_attrs_vals}, train_class_vals), \
           ({"attrs_index": list(range(len(attrs_vals[0]))), "attrs_vals": test_attrs_vals}, test_class_vals), \
           ({"attrs_index": list(range(len(attrs_vals_mode[0]))), "attrs_vals": train_attrs_vals_mode}, train_class_vals_mode), \
           ({"attrs_index": list(range(len(attrs_vals_mode[0]))), "attrs_vals": test_attrs_vals_mode}, test_class_vals_mode), \
           ({"attrs_index": list(range(len(attrs_vals_distrib[0]))), "attrs_vals": train_attrs_vals_distrib}, train_class_vals_distrib), \
           ({"attrs_index": list(range(len(attrs_vals_distrib[0]))), "attrs_vals": test_attrs_vals_distrib}, test_class_vals_distrib), \
           ({"attrs_index": list(range(len(attrs_vals[0]))), "attrs_vals": train_attrs_vals_skip}, train_class_vals_skip), \
           ({"attrs_index": list(range(len(attrs_vals[0]))), "attrs_vals": test_attrs_vals}, test_class_vals)
           

def get_data(name):
    row_attrs_vals = []
    row_class_vals = []

    file = open(name)
    for line in file:
        vals = line.strip().replace(" ", "").split(",")
        if len(vals) < 2:
            continue
        row_attrs_vals.append(vals[:-1])
        row_class_vals.append(vals[-1])
    file.close()
    return row_attrs_vals, row_class_vals

def calculate_attribute_distribution(data, index):
    total = len(data)
    value_counts = Counter(row[index] for row in data if row[index] != 'missing')
    distribution = {val: count / total for val, count in value_counts.items()}
    return distribution

def calculate_attribute_distribution(data):
    columns = list(zip(*data))
    distributions = []
    for column in columns:
        total = len([x for x in column if x != 'missing'])
        count = Counter(x for x in column if x != 'missing')
        distribution = {k: v / total for k, v in count.items()}
        distributions.append(distribution)
    return distributions

def replace_missing_with_distribution(data, distributions):
    new_data = []
    for row in data:
        new_row = []
        for i, value in enumerate(row):
            if value == 'missing':
                if distributions[i]:
                    new_row.append(random.choices(list(distributions[i].keys()), weights=distributions[i].values())[0])
                else:
                    new_row.append('missing')
            else:
                new_row.append(value)
        new_data.append(new_row)
    return new_data