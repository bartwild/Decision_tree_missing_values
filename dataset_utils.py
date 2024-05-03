import random
from collections import Counter

def calculate_mode(attributes):
    """
    Calculate the mode of each column in the dataset, ignoring 'missing' entries.
    """
    columns = list(zip(*attributes))
    mode_values = []
    for column in columns:
        most_common = Counter(x for x in column if x != 'missing').most_common(1)
        mode_value = most_common[0][0] if most_common else 'missing'
        mode_values.append(mode_value)
    return mode_values

def replace_missing_with_mode(data, mode_values):
    """
    Replace 'missing' entries in the dataset with the mode of the corresponding attribute column.
    """
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
    """
    Randomly masks individual attribute values within the dataset.
    """
    masked_attrs_vals = []
    for values in attrs_vals:
        masked_values = ['missing' if random.random() < masking_rate else value for value in values]
        masked_attrs_vals.append(masked_values)
    return masked_attrs_vals

def calculate_weights(data):
    weights = []
    for row in data:
        non_missing_count = sum(1 for value in row if value != 'missing')
        total_count = len(row)
        weight = non_missing_count / total_count
        weights.append(weight)
    return weights

def split_random_to_train_and_test_data(attrs_vals, class_vals, percent_of_train_data, masking_rate=0.1):
    """
    Splits the given dataset into training and testing data randomly after applying random attribute masking.
    """
    attrs_vals = mask_random_attributes(attrs_vals, masking_rate)
    weights = calculate_weights(attrs_vals)  # Calculate weights for each row
    #mode_values = calculate_mode(attrs_vals)
    #attrs_vals = replace_missing_with_mode(attrs_vals, mode_values)

    combined_data = list(zip(attrs_vals, class_vals, weights))  # Include weights in the combined data
    random.shuffle(combined_data)
    #distributions = calculate_attribute_distribution([x for x,_ in combined_data])
    #attrs_vals = replace_missing_with_distribution([x for x,_ in combined_data], distributions)
    split_index = int(len(combined_data) * (percent_of_train_data / 100))
    train_data = combined_data[:split_index]
    test_data = combined_data[split_index:]

    train_attrs_vals, train_class_vals, train_weights = zip(*train_data) if train_data else ([], [], [])
    test_attrs_vals, test_class_vals, test_weights = zip(*test_data) if test_data else ([], [], [])

    return ({"attrs_index": list(range(len(attrs_vals[0]))), "attrs_vals": train_attrs_vals}, train_class_vals, train_weights),\
            ({"attrs_index": list(range(len(attrs_vals[0]))), "attrs_vals": test_attrs_vals}, test_class_vals, test_weights)

def get_data(name):
    """
    Reads a file and extracts attribute values and class labels.
    """
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
    """ Oblicza rozkład wartości dla określonego atrybutu. """
    total = len(data)
    value_counts = Counter(row[index] for row in data if row[index] != 'missing')
    distribution = {val: count / total for val, count in value_counts.items()}
    return distribution

def calculate_attribute_distribution(data):
    """
    Calculates the distribution of values for each attribute in the dataset,
    ignoring 'missing' entries.
    """
    columns = list(zip(*data))
    distributions = []
    for column in columns:
        total = len([x for x in column if x != 'missing'])
        count = Counter(x for x in column if x != 'missing')
        distribution = {k: v / total for k, v in count.items()}
        distributions.append(distribution)
    return distributions

def replace_missing_with_distribution(data, distributions):
    """
    Replace 'missing' entries in the dataset based on the distribution of
    the corresponding attribute column.
    """
    new_data = []
    for row in data:
        new_row = []
        for i, value in enumerate(row):
            if value == 'missing':
                if distributions[i]:  # Check if there is a distribution to sample from
                    new_row.append(random.choices(list(distributions[i].keys()), weights=distributions[i].values())[0])
                else:
                    new_row.append('missing')  # In case there are no known values to base the distribution on
            else:
                new_row.append(value)
        new_data.append(new_row)
    return new_data