import random

random.seed(42137)

ATTRS_NAMES = ["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health"]
CLASS_VALUES = ["not_recom", "recommend", "very_recom", "priority", "spec_prior"]
#ATTRS_NAMES=["buying", "maint", "doors", "persons", "lug_boot", "safety"]
#CLASS_VALUES = ["unacc", "acc", "good", "vgood"]
MAX_DEPTH = 4
PERCENT_OF_TRAIN_DATA = 5
ATTR_TO_INDEX = {ATTRS_NAMES[i]: i for i in range(len(ATTRS_NAMES))}
