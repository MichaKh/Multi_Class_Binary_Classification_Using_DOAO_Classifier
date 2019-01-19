from collections import OrderedDict

import numpy as np
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB

from DataLoader import encode_categorical_features

"""
Kang, Seokho, Sungzoon Cho, and Pilsung Kang. "Constructing a multi-class classifier using one-against-one approach with
 different binary classifiers." Neurocomputing 149 (2015): 677-682.‏
 
The Diversified One-Against-One (DOAO) classifier finds the best classification algorithm for each class pair when
applying the one-against-one approach to multi-class classification problems.

Procedure DOAO:
1: C <- Init classifier set
2: For each class pair (i, j) do:
    2.1: Init the set of datapoints whose class labels are i or j
    2.2: Train cnadidate classifiers, each of which is trained using a different algorithm.
    2.3: Obtain validation error for each candidate classifier
    2.4: cl <- Find the calassifier that corresponds to the minimum valdation error.
    2.5  Add cl to the classifiers set C
3:	end procedure
"""


def build_pair_classifiers(data_set):
    """
    Build pair classifiers with each pair of class labels
    :param data_set: Data points
    :return: Set of pair classifiers
    """
    classifiers_dict = get_classifiers()
    class_labels = list(data_set.iloc[:, -1].unique())
    domains = len(class_labels)
    pair_classifiers_set = []
    for i in range(domains):
        for j in range(i + 1, domains):
            curr_pair_data_points = instances_selection(class_labels[i], class_labels[j], data_set)
            curr_pair_classifiers = build_subset_pair_classifiers(curr_pair_data_points, classifiers_dict)
            name, curr_best_classifier = choose_best_classifier(pair_classifiers=curr_pair_classifiers)
            pair_classifiers_set.append(curr_best_classifier)
    return pair_classifiers_set


def classify_new_instance(instance, pair_classifiers):
    """
    Classify new test instance using each of the pair classifiers.
    The classification corresponds to the majority vote of the classifiers.
    :param instance: Instance to be classified
    :param pair_classifiers: Set of pair classifiers
    :return: Class label
    """
    list_votes = []
    instance = np.array(instance)[:-1].reshape(1, -1)
    for classifier in pair_classifiers:
        class_value = classifier.predict(instance)
        list_votes.append(class_value)
    return max(list_votes, key=list_votes.count)


def get_classifiers():
    """
    chooses the models that will be tested
    :return:
    """
    classifiers = {'DecisionTree': tree.DecisionTreeClassifier(),
                   'NaiveBais': MultinomialNB()}
    return classifiers


def instances_selection(c1, c2, data_set):
    """
    choose only the instances which classification is c1 or c2
    :param c1: First class label
    :param c2: Second class label
    :param data_set: Data points
    :return: Slices data points
    """
    class_column = data_set.columns[-1]
    res = data_set.loc[data_set[class_column].isin([c1, c2])]
    return res


def build_subset_pair_classifiers(data_set, classifiers_dict):
    """
    Train m models form classifiers_set of size m on dataset data_set
    :param data_set: Data points
    :param classifiers_dict: dict: {classifier_name: classifier_instance}
    :return:
    """
    classifiers_model_dict = {}
    X = data_set.iloc[:, :-1]
    y = data_set.iloc[:, -1]
    X = encode_categorical_features(X)  # check if it is by ref
    for name, classifier in classifiers_dict.items():
        trained_classifier = classifier.fit(X, y)
        eval_scores = cross_val_score(classifier, X, y, cv=10)
        classifiers_model_dict[name] = (trained_classifier, np.mean(eval_scores))
    return classifiers_model_dict


def choose_best_classifier(pair_classifiers):
    """
    choose the best classifer based on accuracy from pair_classifiers which contain m trained models
    :param pair_classifiers: Set of pair classifiers
    :return: Classification accuracy dict: {classifier_name: accuracy}
    """
    sorted_dict = sorted(pair_classifiers.items(), key=lambda x: x[1][1], reverse=True)
    name, model_score = list(OrderedDict(sorted_dict).items())[0]
    return name, model_score[0]
