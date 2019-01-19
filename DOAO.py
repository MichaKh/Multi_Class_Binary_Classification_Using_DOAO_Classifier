from collections import OrderedDict

import numpy as np
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder


# from operator import itemgetter


#  [2, 3, 5, , , , ]
# [yellow, yellow, blue, green, blue]
# [yellow, blue, green]

def build_pair_classifiers(data_set):
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


# public
def classify_new_instance(instance, pair_classifiers):
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


'''
choose only the instances which classification is c1 or c2
'''


def instances_selection(c1, c2, data_set):
    class_column = data_set.columns[-1]
    res = data_set.loc[data_set[class_column].isin([c1, c2])]
    return res


'''
train m models form classifiers_set of size m on dataset data_set
'''


def build_subset_pair_classifiers(data_set, classifiers_dict, eval_function='accuracy'):
    classifiers_model_dict = {}
    X = data_set.iloc[:, :-1]
    y = data_set.iloc[:, -1]
    X = encode_categorical_features(X)  # check if it is by ref
    for name, classifier in classifiers_dict.items():
        trained_classifier = classifier.fit(X, y)
        eval_scores = cross_val_score(classifier, X, y, cv=10)
        classifiers_model_dict[name] = (trained_classifier, np.mean(eval_scores))
    return classifiers_model_dict


def encode_categorical_features(X):
    encoded_X = X.copy()
    le = LabelEncoder()
    for col in encoded_X.columns:
        if encoded_X[col].dtype not in [np.float64, np.int64]:  # numeric
            encoded_X[col] = le.fit_transform(X[col])
    return encoded_X


'''
choose the best classifer based on accuracy from pair_classifiers which contain m trained models
'''


# ("tree")-> (model, score)
def choose_best_classifier(pair_classifiers):
    sorted_dict = sorted(pair_classifiers.items(), key=lambda x: x[1][1], reverse=True)
    name, model_score = list(OrderedDict(sorted_dict).items())[0]
    return name, model_score[0]
